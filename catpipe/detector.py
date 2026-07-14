"""
detector.py  --  A.4, the anomaly detector. Four layers, one alert table.

  Layer 1    interval exceedance. For each scored day, conformal margins are
             computed from the TRAILING ledger window (calibrate.py's
             conformal_margins / apply_margins — the same functions, the
             promised reuse) and the day's actual is tested against its
             CALIBRATED band. Uncalibrated GBM intervals fire at ~2x design
             rate; calibrated ones fire at the rate the triage load was
             designed for. Margins are computed strictly from days BEFORE
             the scored day: an anomaly must never soften its own alarm.

  Layer 1.5  CUSUM drift. The December 2024 lesson: a five-week glide never
             breaches a daily interval, yet ends 47% below where it began.
             Residuals (actual - q50) are standardised by a trailing robust
             sigma (MAD-based) and fed to a two-sided CUSUM with standard
             reference k=0.5 and threshold h=5 (in sigma units): small
             persistent drift accumulates to an alarm that no single day
             would raise. The statistic resets after each alarm.

  Layer 2    job-level robust-z. Per (job_name, pool), today's attributed
             cost against the trailing history's median and MAD:
             z = (x - median) / (1.4826 * MAD). Median/MAD, never mean/std,
             because job cost history contains exactly the spikes the
             detector must not learn to expect. A job with insufficient
             history is a NEW-JOB informational alert, not a z-score — a
             brand-new expensive job is precisely what a cost owner wants
             to hear about. An absolute floor suppresses penny alerts on
             stable jobs whose MAD is near zero.

  Attribution  unknown_pct rule: a day whose attributed cost is largely
             Unknown/NULL-team (the column frame 2 already computes) is an
             attribution-health alert regardless of totals, because the
             completeness gate is blind to attribution failure.

Alert table: one row per alert with a stable alert_id (hash of layer, scope,
date, direction), severity, human-readable message, and a STATUS column
carried from day one ('new' by default; merge_alert_status preserves triage
states across re-scores). The doc's operational caution is the design
driver: an alert table nobody can triage decays into noise within a month,
so the table is born with its feedback loop.

Approved stack only: pandas + numpy (hashlib from stdlib).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import calibrate as C

MAD_SCALE = 1.4826  # MAD -> sigma under normality

ALERT_COLUMNS = [
    "alert_id", "run_date", "layer", "scope", "metric", "observed",
    "expected", "lo", "hi", "score", "direction", "severity", "message",
    "status",
]


@dataclass
class DetectorConfig:
    """Every threshold in one visible, versionable place. These are the
    knobs the triage feedback loop will tune; none of them hides in a
    function body.
    """
    target_coverage: float = 0.90
    margin_window_days: int = 90         # trailing ledger for Layer 1
    min_margin_scores: int = 30          # below: pooled margin fallback
    cusum_k: float = 0.5                 # reference value, sigma units
    cusum_h: float = 5.0                 # alarm threshold, sigma units
    cusum_sigma_window: int = 60         # trailing days for robust sigma
    cusum_min_history: int = 20
    z_threshold: float = 5.0             # Layer 2 robust-z alarm
    min_job_history: int = 10            # below: new-job alert instead
    job_abs_floor: float = 5.0           # suppress penny alerts (currency)
    unknown_pct_threshold: float = 0.20  # the A.3/A.4 attribution rule


# ---------------------------------------------------------------------------
# alert-table plumbing
# ---------------------------------------------------------------------------

def _alert_id(layer: str, scope: str, run_date, direction: str) -> str:
    raw = f"{layer}|{scope}|{run_date}|{direction}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _scope(row, group_keys: list[str]) -> str:
    return ";".join(f"{k}={row[k]}" for k in group_keys)


def _severity(score: float) -> str:
    """One monotone mapping for every layer: score is 'how far past the
    line', in the layer's own dimensionless units (scaled exceedance,
    CUSUM overshoot / h, |z| / threshold - 1).
    """
    if score >= 1.0:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _mk_alerts(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    out = pd.DataFrame(rows)
    out["status"] = "new"
    out["alert_id"] = [
        _alert_id(r["layer"], r["scope"], r["run_date"], r["direction"])
        for r in rows
    ]
    return out[ALERT_COLUMNS]


def merge_alert_status(new: pd.DataFrame,
                       previous: pd.DataFrame | None) -> pd.DataFrame:
    """Re-scoring must never reset triage: an alert already acknowledged,
    marked expected, or under investigation keeps its status when the same
    alert_id reappears. New ids arrive as 'new'.
    """
    if previous is None or previous.empty:
        return new
    keep = previous.set_index("alert_id")["status"]
    out = new.copy()
    out["status"] = out["alert_id"].map(keep).fillna("new")
    return out


# ---------------------------------------------------------------------------
# Layer 1: calibrated interval exceedance
# ---------------------------------------------------------------------------

def layer1_interval_alerts(
    ledger: pd.DataFrame,
    group_keys: list[str],
    config: DetectorConfig = DetectorConfig(),
    score_from: date | None = None,
) -> pd.DataFrame:
    """Replay the nightly Layer 1 rule over a prediction ledger (columns:
    group_keys, run_date, y_true, q05, q50, q95 — RAW model intervals).

    For each scored day d: margins from the ledger rows in
    [d - margin_window_days, d), applied to d's rows, exceedance tested on
    the calibrated band. Strictly-past margins are the point: the spike
    being scored contributes nothing to the width of the band that scores
    it. Days whose trailing window is empty are skipped (nothing honest to
    calibrate against), which in production is only the burn-in fortnight.

    score_from limits the replay (production scores one day; back-testing
    the detector replays a span).
    """
    led = ledger.dropna(subset=["y_true", "q05", "q50", "q95"]).copy()
    led["run_date"] = pd.to_datetime(led["run_date"]).dt.date
    days = sorted(led["run_date"].unique())
    if score_from is not None:
        days = [d for d in days if d >= score_from]

    rows = []
    for d in days:
        lo_bound = (pd.Timestamp(d)
                    - pd.Timedelta(days=config.margin_window_days)).date()
        trailing = led[(led["run_date"] < d)
                       & (led["run_date"] >= lo_bound)]
        if trailing.empty:
            continue
        margins = C.conformal_margins(
            trailing, group_keys,
            target_coverage=config.target_coverage,
            min_group_scores=config.min_margin_scores)
        today = led[led["run_date"] == d]
        cal = C.apply_margins(today, margins, group_keys)
        for _, r in cal.iterrows():
            width = max(r["q95"] - r["q05"], C.WIDTH_FLOOR)
            if r["y_true"] > r["q95"]:
                direction, exceed = "above", r["y_true"] - r["q95"]
            elif r["y_true"] < r["q05"]:
                direction, exceed = "below", r["q05"] - r["y_true"]
            else:
                continue
            score = exceed / width
            scope = _scope(r, group_keys)
            rows.append(dict(
                run_date=d, layer="L1_interval", scope=scope,
                metric="daily_cost", observed=float(r["y_true"]),
                expected=float(r["q50"]), lo=float(r["q05"]),
                hi=float(r["q95"]), score=float(score),
                direction=direction, severity=_severity(score),
                message=(f"{scope}: actual {r['y_true']:.2f} {direction} the "
                         f"calibrated {int(config.target_coverage*100)}% band "
                         f"[{r['q05']:.2f}, {r['q95']:.2f}] "
                         f"(median {r['q50']:.2f})"),
            ))
    return _mk_alerts(rows)


# ---------------------------------------------------------------------------
# Layer 1.5: CUSUM drift on standardised residuals
# ---------------------------------------------------------------------------

def layer15_cusum_alerts(
    ledger: pd.DataFrame,
    group_keys: list[str],
    config: DetectorConfig = DetectorConfig(),
) -> pd.DataFrame:
    """Two-sided CUSUM per group on z_t = (y_t - q50_t) / sigma_t, sigma
    the MAD-based robust scale of the trailing residual window (strictly
    past). S+ accumulates upward drift, S- downward:

        S+_t = max(0, S+_{t-1} + z_t - k)
        S-_t = max(0, S-_{t-1} - z_t - k)

    Alarm when either exceeds h; the statistic resets to zero after an
    alarm so one long glide produces a sequence of alarms rather than a
    single ever-louder one — each is a fresh 'still drifting' signal.
    """
    led = ledger.dropna(subset=["y_true", "q50"]).copy()
    led["run_date"] = pd.to_datetime(led["run_date"]).dt.date
    led["_resid"] = led["y_true"] - led["q50"]

    rows = []
    for keys, grp in led.groupby(group_keys, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        grp = grp.sort_values("run_date")
        resid = grp["_resid"].to_numpy(float)
        dates = grp["run_date"].to_list()
        s_pos = s_neg = 0.0
        for i in range(len(resid)):
            hist = resid[max(0, i - config.cusum_sigma_window):i]
            if len(hist) < config.cusum_min_history:
                continue
            mad = np.median(np.abs(hist - np.median(hist)))
            sigma = max(MAD_SCALE * mad, C.WIDTH_FLOOR)
            z = resid[i] / sigma
            s_pos = max(0.0, s_pos + z - config.cusum_k)
            s_neg = max(0.0, s_neg - z - config.cusum_k)
            stat, direction = ((s_pos, "above") if s_pos >= s_neg
                               else (s_neg, "below"))
            if stat > config.cusum_h:
                scope = ";".join(f"{k}={v}"
                                 for k, v in zip(group_keys, keys))
                score = (stat - config.cusum_h) / config.cusum_h
                rows.append(dict(
                    run_date=dates[i], layer="L1.5_cusum", scope=scope,
                    metric="residual_drift", observed=float(resid[i]),
                    expected=0.0, lo=np.nan, hi=np.nan,
                    score=float(score), direction=direction,
                    severity=_severity(score),
                    message=(f"{scope}: sustained {direction}-forecast drift; "
                             f"CUSUM {stat:.1f} exceeded h={config.cusum_h} "
                             f"(sigma {sigma:.2f}). Glide-type change: no "
                             "single day breached its interval."),
                ))
                s_pos = s_neg = 0.0
    return _mk_alerts(rows)


# ---------------------------------------------------------------------------
# Layer 2: job-level robust-z
# ---------------------------------------------------------------------------

JOB_PROFILE_KEYS = ["subscription_id", "batch_account_name", "pool_name",
                    "job_name"]


def layer2_job_alerts(
    job_cost: pd.DataFrame,
    config: DetectorConfig = DetectorConfig(),
    score_from: date | None = None,
) -> pd.DataFrame:
    """Per (job_name, pool) profile: today's cost against the trailing
    history's median and MAD, strictly past. Insufficient history is a
    new-job informational alert on the job's FIRST appearance only. The
    absolute floor keeps a stable job's near-zero MAD from turning pennies
    into pages.
    """
    jc = job_cost.copy()
    jc["run_date"] = pd.to_datetime(jc["run_date"]).dt.date
    daily = (jc.groupby(JOB_PROFILE_KEYS + ["run_date"], observed=True)
             ["cost"].sum().reset_index())

    rows = []
    for keys, grp in daily.groupby(JOB_PROFILE_KEYS, observed=True,
                                   sort=False):
        grp = grp.sort_values("run_date")
        costs = grp["cost"].to_numpy(float)
        dates = grp["run_date"].to_list()
        scope = ";".join(f"{k}={v}"
                         for k, v in zip(JOB_PROFILE_KEYS, keys))
        for i in range(len(costs)):
            d = dates[i]
            if score_from is not None and d < score_from:
                continue
            hist = costs[:i]
            if len(hist) < config.min_job_history:
                if i == 0:
                    rows.append(dict(
                        run_date=d, layer="L2_job", scope=scope,
                        metric="new_job", observed=float(costs[i]),
                        expected=np.nan, lo=np.nan, hi=np.nan,
                        score=0.0, direction="new",
                        severity="low",
                        message=(f"{scope}: first appearance, cost "
                                 f"{costs[i]:.2f}. No profile yet."),
                    ))
                continue
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med)))
            sigma = max(MAD_SCALE * mad, C.WIDTH_FLOOR)
            dev = costs[i] - med
            if abs(dev) < config.job_abs_floor:
                continue
            z = dev / sigma
            if abs(z) <= config.z_threshold:
                continue
            score = abs(z) / config.z_threshold - 1.0
            direction = "above" if dev > 0 else "below"
            rows.append(dict(
                run_date=d, layer="L2_job", scope=scope,
                metric="job_daily_cost", observed=float(costs[i]),
                expected=med, lo=np.nan, hi=np.nan,
                score=float(score), direction=direction,
                severity=_severity(score),
                message=(f"{scope}: cost {costs[i]:.2f} vs profile median "
                         f"{med:.2f} (robust z = {z:+.1f}, threshold "
                         f"{config.z_threshold:.0f})"),
            ))
    return _mk_alerts(rows)


# ---------------------------------------------------------------------------
# attribution health: the unknown_pct rule
# ---------------------------------------------------------------------------

def attribution_alerts(
    frame_2: pd.DataFrame,
    config: DetectorConfig = DetectorConfig(),
) -> pd.DataFrame:
    """One alert per day whose unknown_pct (already computed by
    build_frame_2: Unknown + __NULL_TEAM__ cost share) exceeds the
    threshold. The completeness gate is blind to this failure mode; the
    detector is not allowed to be.
    """
    daily = (frame_2[["run_date", "unknown_pct"]].dropna()
             .drop_duplicates("run_date"))
    daily["run_date"] = pd.to_datetime(daily["run_date"]).dt.date
    rows = []
    for _, r in daily.iterrows():
        if r["unknown_pct"] <= config.unknown_pct_threshold:
            continue
        score = (r["unknown_pct"] / config.unknown_pct_threshold) - 1.0
        rows.append(dict(
            run_date=r["run_date"], layer="attribution", scope="estate",
            metric="unknown_pct", observed=float(r["unknown_pct"]),
            expected=config.unknown_pct_threshold, lo=np.nan, hi=np.nan,
            score=float(score), direction="above",
            severity=_severity(score),
            message=(f"{r['unknown_pct']:.0%} of attributed cost sits in "
                     "Unknown/NULL team (threshold "
                     f"{config.unknown_pct_threshold:.0%}). Attribution "
                     "health, not spend: totals may look normal."),
        ))
    return _mk_alerts(rows)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def run_detector(
    ledgers: dict[str, tuple[pd.DataFrame, list[str]]],
    job_cost: pd.DataFrame | None = None,
    frame_2: pd.DataFrame | None = None,
    config: DetectorConfig = DetectorConfig(),
    previous_alerts: pd.DataFrame | None = None,
    score_from: date | None = None,
) -> pd.DataFrame:
    """All layers, one table. ledgers maps a label (e.g. 'frame_1a') to
    (prediction ledger, group_keys); Layers 1 and 1.5 run per ledger, Layer
    2 on job_cost, the attribution rule on frame_2. Statuses from
    previous_alerts survive re-scoring. Sorted most-severe-first, then
    newest-first: the top of the table is the morning's triage order.
    """
    pieces = []
    for label, (ledger, group_keys) in ledgers.items():
        l1 = layer1_interval_alerts(ledger, group_keys, config,
                                    score_from=score_from)
        l15 = layer15_cusum_alerts(ledger, group_keys, config)
        if score_from is not None and len(l15):
            l15 = l15[l15["run_date"] >= score_from]
        for piece in (l1, l15):
            if len(piece):
                piece = piece.copy()
                piece["scope"] = label + ":" + piece["scope"]
                # scope changed => recompute stable ids
                piece["alert_id"] = [
                    _alert_id(r["layer"], r["scope"], r["run_date"],
                              r["direction"])
                    for _, r in piece.iterrows()
                ]
                pieces.append(piece)
    if job_cost is not None:
        pieces.append(layer2_job_alerts(job_cost, config,
                                        score_from=score_from))
    if frame_2 is not None:
        att = attribution_alerts(frame_2, config)
        if score_from is not None and len(att):
            att = att[att["run_date"] >= score_from]
        pieces.append(att)

    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return pd.DataFrame(columns=ALERT_COLUMNS)
    table = pd.concat(pieces, ignore_index=True)
    table = merge_alert_status(table, previous_alerts)
    sev_rank = table["severity"].map({"high": 0, "medium": 1, "low": 2})
    table = (table.assign(_s=sev_rank)
             .sort_values(["_s", "run_date"], ascending=[True, False])
             .drop(columns="_s").reset_index(drop=True))
    return table
