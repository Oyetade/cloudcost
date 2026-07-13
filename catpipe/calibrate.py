"""
calibrate.py  --  conformalised quantile calibration (CQR, Romano et al.
2019): measure how wrong the model's interval claims are on data it has not
seen, and widen (or shrink) future intervals by exactly that much. The
split-conformal guarantee: on exchangeable data the corrected interval
achieves at least the target coverage in finite samples, REGARDLESS of how
miscalibrated the underlying model is. This is the same move as the gate and
the partition identity — assert the property, never assume it — applied to
probability claims.

Why it exists: quantile GBMs systematically under-cover (observed ~0.82
against the 0.90 target). A.4 Layer 1 alerts when an actual escapes the
5-95 band; uncorrected, it would fire nearly twice its design rate and the
alert table would decay into ignored noise within a month.

Two entry points, one implementation:

  ConformalWrapper   wraps any fitted-or-fittable harness forecaster (the
                     GBM, a baseline, anything with fit/predict). At fit
                     time it holds out a CHRONOLOGICAL calibration tail,
                     scores the inner model's intervals on it, and stores
                     margins; at predict time it widens. Slots into
                     harness.run_models unchanged.

  conformal_margins  pure functions over a prediction ledger — the harness
  apply_margins      ledger today, the production alert ledger tomorrow.
                     A.4 Layer 1 REUSES THESE: nightly, recompute margins
                     from the trailing ledger window and widen the day's
                     intervals before the exceedance rule runs.

The nonconformity score is the CQR score max(q05 - y, y - q95): positive
when the actual escapes the band, negative when it sits comfortably inside,
so the calibrated margin can be negative and SHRINK an over-wide model —
calibration runs in both directions. The scaled variant divides by the
interval's own width, so a volatile wide-band day receives more absolute
widening than a quiet one; this matters for CAT because pool-day dispersion
varies by orders of magnitude across pools.

Margins are per group where the calibration window holds enough scores,
pooled across groups where it does not — the same thin-group fallback the
baselines' residual intervals use.

Approved stack only: pandas + numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .baselines import FrameSpec

WIDTH_FLOOR = 1e-9  # a zero-width interval cannot scale a score
POOLED = "__POOLED__"


# ---------------------------------------------------------------------------
# the maths, once: scores, finite-sample quantile, margins, application
# ---------------------------------------------------------------------------

def cqr_scores(y: pd.Series, q05: pd.Series, q95: pd.Series,
               scaled: bool = True) -> pd.Series:
    """CQR nonconformity: max(q05 - y, y - q95). Positive = escaped the
    band, negative = inside with room. Scaled divides by the interval's own
    width so the margin adapts to per-day dispersion.
    """
    lo = (q05 - y).to_numpy(float)
    hi = (y - q95).to_numpy(float)
    s = np.maximum(lo, hi)
    if scaled:
        w = np.maximum((q95 - q05).to_numpy(float), WIDTH_FLOOR)
        s = s / w
    return pd.Series(s, index=y.index)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """The conformal quantile with the (n+1) correction: the k-th smallest
    score where k = ceil((n+1) * coverage). This, not the plain empirical
    quantile, is what carries the finite-sample guarantee. When k exceeds n
    (calibration set too small for the requested coverage), the guarantee is
    unattainable; return the max score and let the caller report it, rather
    than inventing a margin.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return 0.0
    k = math.ceil((n + 1) * coverage)
    if k > n:
        return float(s[-1])
    return float(s[k - 1])


def conformal_margins(
    ledger: pd.DataFrame,
    group_keys: list[str],
    target_coverage: float = 0.90,
    scaled: bool = True,
    min_group_scores: int = 30,
) -> dict:
    """Margins from a prediction ledger (columns: group_keys, y_true, q05,
    q95). One pooled margin always; a per-group margin wherever the ledger
    holds at least min_group_scores rows for that group. Returns a dict —
    kept JSON-friendly deliberately, because A.4 will persist it beside the
    alert table so every alert is auditable back to the margin that shaped
    its interval:

        {"target_coverage", "scaled", "n", "guaranteed",
         "pooled": float, "groups": {group_tuple: float}}

    guaranteed is False when the pooled calibration set is too small for the
    (n+1) correction to reach the target — the margin is still the best
    available (the max score) but the theorem no longer applies, and the
    caller should say so rather than claim coverage it cannot have.
    """
    scored = ledger.dropna(subset=["y_true", "q05", "q95"])
    s_all = cqr_scores(scored["y_true"], scored["q05"], scored["q95"],
                       scaled=scaled)
    n = len(s_all)
    k = math.ceil((n + 1) * target_coverage) if n else 1
    margins = {
        "target_coverage": target_coverage,
        "scaled": scaled,
        "n": n,
        "guaranteed": bool(n and k <= n),
        "pooled": finite_sample_quantile(s_all.to_numpy(), target_coverage),
        "groups": {},
    }
    for keys, grp in scored.groupby(group_keys, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(grp) >= min_group_scores:
            s = cqr_scores(grp["y_true"], grp["q05"], grp["q95"],
                           scaled=scaled)
            margins["groups"][keys] = finite_sample_quantile(
                s.to_numpy(), target_coverage)
    return margins


def apply_margins(
    preds: pd.DataFrame,
    margins: dict,
    group_keys: list[str],
) -> pd.DataFrame:
    """Widen (or shrink) q05/q95 by the group's margin, pooled fallback.
    Scaled margins multiply the interval's own width. The median is never
    touched; the band is re-clamped around it (a negative margin may not
    shrink an interval past its median) and floored at zero because cost is
    non-negative.
    """
    out = preds.copy()
    m = np.array([
        margins["groups"].get(tuple(row[k] for k in group_keys),
                              margins["pooled"])
        for _, row in out.iterrows()
    ], dtype=float)
    if margins["scaled"]:
        w = np.maximum((out["q95"] - out["q05"]).to_numpy(float),
                       WIDTH_FLOOR)
        m = m * w
    out["q05"] = out["q05"] - m
    out["q95"] = out["q95"] + m
    out["q05"] = np.minimum(out["q05"], out["q50"])
    out["q95"] = np.maximum(out["q95"], out["q50"])
    out[["q05", "q50", "q95"]] = out[["q05", "q50", "q95"]].clip(lower=0)
    return out


# ---------------------------------------------------------------------------
# the harness-facing wrapper
# ---------------------------------------------------------------------------

class ConformalWrapper:
    """Any harness forecaster, conformally calibrated.

    fit: split the training window CHRONOLOGICALLY — never randomly, since
    shuffling time leaks the future into calibration — into a proper
    training slice and a calibration tail (default 90 days). Fit the inner
    model on the proper slice, score its intervals on the tail, store
    margins. Then, by default, refit the inner model on the full window so
    the tail's information is not thrown away at predict time; this trades
    the strict exchangeability of split-CQR for a practically better model,
    a standard and disclosed compromise (`refit_on_full=False` restores the
    textbook procedure). If the tail is too thin to calibrate, margins are
    zero and calibration_report says so — the wrapper degrades to the inner
    model, loudly, never silently.

    predict: inner predictions widened by apply_margins. The
    calibration_report (margins dict plus raw-coverage-on-tail) is kept on
    the instance for the audit trail A.4 will persist.
    """

    def __init__(
        self,
        inner,
        target_coverage: float = 0.90,
        calib_tail_days: int = 90,
        scaled: bool = True,
        min_group_scores: int = 30,
        min_calib_rows: int = 30,
        refit_on_full: bool = True,
    ):
        self.inner = inner
        self.target_coverage = target_coverage
        self.calib_tail_days = calib_tail_days
        self.scaled = scaled
        self.min_group_scores = min_group_scores
        self.min_calib_rows = min_calib_rows
        self.refit_on_full = refit_on_full
        self.name = f"conformal_{getattr(inner, 'name', type(inner).__name__)}"

    def fit(self, train: pd.DataFrame, spec: FrameSpec) -> None:
        dates = pd.to_datetime(train[spec.date_col])
        cutoff = dates.max() - pd.Timedelta(days=self.calib_tail_days - 1)
        calib_mask = dates >= cutoff
        proper = train[~calib_mask]
        calib = train[calib_mask]

        if len(calib) < self.min_calib_rows or proper.empty:
            self.inner.fit(train, spec)
            self._margins = {"target_coverage": self.target_coverage,
                             "scaled": self.scaled, "n": 0,
                             "guaranteed": False, "pooled": 0.0,
                             "groups": {}}
            self.calibration_report = {
                "calibrated": False,
                "reason": (f"calibration tail has {len(calib)} rows, "
                           f"needs {self.min_calib_rows}"),
                "margins": self._margins,
            }
            return

        self.inner.fit(proper, spec)
        preds = self.inner.predict(calib, spec)
        led = calib[spec.group_keys + [spec.date_col, spec.target]].rename(
            columns={spec.target: "y_true"}).reset_index(drop=True)
        led = led.merge(preds, on=spec.group_keys + [spec.date_col],
                        how="left")

        self._margins = conformal_margins(
            led, spec.group_keys,
            target_coverage=self.target_coverage,
            scaled=self.scaled,
            min_group_scores=self.min_group_scores)

        scored = led.dropna(subset=["q05", "q50", "q95"])
        raw_cov = float(((scored["y_true"] >= scored["q05"])
                         & (scored["y_true"] <= scored["q95"])).mean()) \
            if len(scored) else float("nan")
        self.calibration_report = {
            "calibrated": True,
            "raw_coverage_on_tail": raw_cov,
            "margins": self._margins,
        }

        if self.refit_on_full:
            self.inner.fit(train, spec)

    def predict(self, test: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
        preds = self.inner.predict(test, spec)
        return apply_margins(preds, self._margins, spec.group_keys)
