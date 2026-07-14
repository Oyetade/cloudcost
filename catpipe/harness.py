"""
harness.py  --  the single walk-forward back-testing harness of 5.4: every
model, baseline or GBM, is evaluated through the same folds via the same
narrow interface (fit / predict-quantiles), so no comparison is ever
apples-to-oranges.

Fold structure: monthly rolled origins. At each origin o, the model trains
on rows strictly before o and is scored on the calendar month starting at o.
Direct horizon by construction — the model never sees inside the test month.

Origin honesty (5.7): the earliest origin is derived from the data the frame
actually holds AFTER regime filtering, never hardcoded. An origin whose
training window would reach before the frame's honest start is REFUSED with
an error, not silently accepted: a back-test that trains a gated model on
ungated history is the dishonesty the regime machinery exists to prevent.
Per-product windows come out of exactly this: frame 1a filtered to
featured_gated origins around mid-2024; frame 1b masked to post_glide
origins from mid-2025; the cost-only baselines, given the wider regime set,
may originate earlier.

Metrics: daily MAE on the median; monthly aggregate percentage error per
group (sum of daily medians vs sum of actuals — the number stakeholders
compare against the incumbent); pinball loss at 5/50/95; and empirical
coverage of the 5-95 interval (target 0.90). Pinball and coverage make the
harness quantile-shaped now, so A.4 Layer 1 falls out of the same numbers
when the GBM arrives.

Approved stack only: pandas + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .baselines import FrameSpec, spec_for_frame

DEFAULT_REGIMES = ("featured_gated",)
DEFAULT_MIN_TRAIN_DAYS = 180


class BacktestError(ValueError):
    """Raised when a fold would be dishonest (origin too early, empty test
    month, no training data) rather than producing a quietly wrong number.
    """


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else \
        date(d.year, d.month + 1, 1)


def prepare(frame: pd.DataFrame,
            regimes: tuple[str, ...] = DEFAULT_REGIMES,
            mask: pd.Series | None = None) -> pd.DataFrame:
    """Select the honest evaluation slice: filter by data_regime (the model
    layer's job, per the transform-labels/model-filters split) and by an
    optional extra boolean mask (frame 1b's post_glide, frame 2's
    unknown_pct rule). Returns a copy with run_date normalised to date.
    """
    out = frame
    if "data_regime" in out.columns and regimes:
        out = out[out["data_regime"].isin(regimes)]
        if out.empty:
            raise BacktestError(
                f"no rows survive regime filter {regimes}; the frame does "
                "not support this evaluation window")
    if mask is not None:
        out = out[mask.reindex(out.index).fillna(False)]
        if out.empty:
            raise BacktestError(
                f"rows survive regime filter {regimes} but none survive the "
                "extra mask (post_glide / unknown_pct); the honest window "
                "is empty")
    out = out.copy()
    out["run_date"] = pd.to_datetime(out["run_date"]).dt.date
    return out


def monthly_origins(prepared: pd.DataFrame,
                    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS
                    ) -> list[date]:
    """Every month-start origin the prepared frame can honestly support:
    at least min_train_days of history before the origin, and a complete
    test month after it (a partial final month would flatter whoever is
    lucky with its missing days).
    """
    lo, hi = prepared["run_date"].min(), prepared["run_date"].max()
    earliest = lo + pd.Timedelta(days=min_train_days)
    # round UP to the next month start: rounding down would emit an origin
    # with fewer than min_train_days of history, which _validate_origin
    # (rightly) refuses. The generator and the validator must agree.
    first = _month_start(earliest)
    if pd.Timestamp(first) < pd.Timestamp(earliest):
        first = _next_month(first)
    origins = []
    o = first
    while _next_month(o) <= hi + pd.Timedelta(days=1):
        origins.append(o)
        o = _next_month(o)
    return origins


def _validate_origin(origin: date, prepared: pd.DataFrame,
                     min_train_days: int) -> None:
    lo = prepared["run_date"].min()
    earliest_honest = lo + pd.Timedelta(days=min_train_days)
    if pd.Timestamp(origin) < pd.Timestamp(earliest_honest):
        raise BacktestError(
            f"origin {origin} refused: only "
            f"{(pd.Timestamp(origin) - pd.Timestamp(lo)).days} days of "
            f"honest history precede it (frame starts {lo}, "
            f"min_train_days={min_train_days}). A gated model cannot "
            "originate before its regime has accumulated enough history "
            "(5.7).")


def walk_forward(
    frame: pd.DataFrame,
    model,
    spec: FrameSpec | None = None,
    origins: list[date] | None = None,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
    mask: pd.Series | None = None,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
) -> pd.DataFrame:
    """Run one model through the monthly folds. Returns the prediction
    ledger: one row per (group, day, origin) with y_true, q05, q50, q95 and
    the model's name — the raw material every metric is computed from, and
    the artefact to keep, because metrics can be recomputed from the ledger
    but not the reverse.
    """
    spec = spec or spec_for_frame(frame)
    prepared = prepare(frame, regimes=regimes, mask=mask)

    if origins is None:
        origins = monthly_origins(prepared, min_train_days)
        if not origins:
            raise BacktestError(
                "no honest monthly origin exists: the prepared frame spans "
                f"{prepared['run_date'].min()} to "
                f"{prepared['run_date'].max()}, shorter than "
                f"min_train_days={min_train_days} plus one test month")
    else:
        for o in origins:
            _validate_origin(o, prepared, min_train_days)

    ledger = []
    for origin in origins:
        train = prepared[prepared["run_date"] < origin]
        test_end = _next_month(origin)
        test = prepared[(prepared["run_date"] >= origin)
                        & (prepared["run_date"] < test_end)]
        if train.empty or test.empty:
            raise BacktestError(
                f"origin {origin}: empty {'train' if train.empty else 'test'}"
                " slice; refuse rather than score a hollow fold")
        model.fit(train, spec)
        preds = model.predict(test, spec)
        fold = test[spec.group_keys + ["run_date", spec.target]].rename(
            columns={spec.target: "y_true"}).reset_index(drop=True)
        fold = fold.merge(
            preds, on=spec.group_keys + ["run_date"], how="left")
        fold["origin"] = origin
        fold["model"] = getattr(model, "name", type(model).__name__)
        ledger.append(fold)

    return pd.concat(ledger, ignore_index=True)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def pinball_loss(y: pd.Series, q: pd.Series, tau: float) -> float:
    """Quantile (pinball) loss: the score a tau-quantile forecast is trained
    to minimise, and therefore the fair way to compare interval forecasts.
    """
    d = (y - q).to_numpy(float)
    return float(np.mean(np.where(d >= 0, tau * d, (tau - 1) * d)))


def evaluate(ledger: pd.DataFrame,
             group_keys: list[str] | None = None) -> pd.DataFrame:
    """Roll the prediction ledger up to one row per model:

      n / n_scored     rows, and rows the model actually predicted (a new
                       group in a test month yields NaN — reported, not
                       hidden)
      mae_daily        mean |y - q50| on scored rows

      Three monthly errors, three questions:
      monthly_pct_err_estate  per month: |sum of ALL predictions - sum of
                       ALL actuals| / total actuals, averaged over months.
                       THE incumbent-comparison number: what finance sees
                       when the estate's monthly bill lands. Offsetting
                       group errors legitimately cancel here.
      monthly_wape     spend-weighted: sum over (group, month) of
                       |pred - actual|, divided by total actual spend.
                       Attribution accuracy: offsetting errors do NOT
                       cancel. The gap between this and the estate number
                       is exactly how much the model relies on cancellation.
      monthly_bias     SIGNED estate error: mean over months of
                       (sum predictions - sum actuals) / actuals. Negative
                       means the model systematically under-forecasts the
                       monthly total (the asinh-median signature); the
                       absolute estate error above says how big the miss
                       is, this says which way and whether it is one-signed.
      monthly_pct_err  unweighted mean over (group, month) ratios. A
                       diagnostic for small groups — a tiny pool with a
                       near-zero month inflates it enormously, so it must
                       never be quoted as the headline (the 206% lesson).

      pinball_05/50/95 quantile losses
      coverage_90      share of actuals inside [q05, q95]; target 0.90.
                       Above target means intervals too wide to be useful,
                       below means A.4 Layer 1 would over-alert.
    """
    rows = []
    for name, m in ledger.groupby("model", observed=True):
        scored = m.dropna(subset=["q50"])
        row = {"model": name, "n": len(m), "n_scored": len(scored)}
        if len(scored):
            row["mae_daily"] = float(
                (scored["y_true"] - scored["q50"]).abs().mean())
            agg_keys = (group_keys or []) + ["origin"]
            monthly = scored.groupby(agg_keys, observed=True).agg(
                y=("y_true", "sum"), p=("q50", "sum"))
            nonzero = monthly[monthly["y"].abs() > 1e-9]
            row["monthly_pct_err"] = float(
                ((nonzero["p"] - nonzero["y"]).abs()
                 / nonzero["y"].abs()).mean()) if len(nonzero) else np.nan

            estate = scored.groupby("origin", observed=True).agg(
                y=("y_true", "sum"), p=("q50", "sum"))
            nz = estate[estate["y"].abs() > 1e-9]
            row["monthly_pct_err_estate"] = float(
                ((nz["p"] - nz["y"]).abs()
                 / nz["y"].abs()).mean()) if len(nz) else np.nan
            row["monthly_bias"] = float(
                ((nz["p"] - nz["y"])
                 / nz["y"].abs()).mean()) if len(nz) else np.nan

            total_y = float(monthly["y"].abs().sum())
            row["monthly_wape"] = float(
                (monthly["p"] - monthly["y"]).abs().sum() / total_y
            ) if total_y > 1e-9 else np.nan
            row["pinball_05"] = pinball_loss(scored["y_true"],
                                             scored["q05"], 0.05)
            row["pinball_50"] = pinball_loss(scored["y_true"],
                                             scored["q50"], 0.50)
            row["pinball_95"] = pinball_loss(scored["y_true"],
                                             scored["q95"], 0.95)
            row["coverage_90"] = float(
                ((scored["y_true"] >= scored["q05"])
                 & (scored["y_true"] <= scored["q95"])).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_daily").reset_index(drop=True)


def run_models(
    frame: pd.DataFrame,
    models: list,
    spec: FrameSpec | None = None,
    regimes: tuple[str, ...] = DEFAULT_REGIMES,
    mask: pd.Series | None = None,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every model through identical folds; returns (summary, ledger). The
    shared-origin discipline is enforced by computing origins once and
    passing the same list to every model.
    """
    spec = spec or spec_for_frame(frame)
    prepared = prepare(frame, regimes=regimes, mask=mask)
    origins = monthly_origins(prepared, min_train_days)
    if not origins:
        raise BacktestError(
            "no honest monthly origin exists for this frame/regime/mask "
            f"combination (span {prepared['run_date'].min()} to "
            f"{prepared['run_date'].max()}, min_train_days={min_train_days})")
    ledgers = [
        walk_forward(frame, m, spec=spec, origins=origins, regimes=regimes,
                     mask=mask, min_train_days=min_train_days)
        for m in models
    ]
    ledger = pd.concat(ledgers, ignore_index=True)
    return evaluate(ledger, group_keys=spec.group_keys), ledger
