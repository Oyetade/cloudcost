"""
feature_factory.py  --  build-order item 1: lags, rolls, calendar leads,
daily padding, and the effective-price drift feature. The gate to everything;
nothing in Appendix A trains without it.

Leakage discipline (7.5 / assert_no_same_day_cost): every feature produced
here is either lagged by at least one day or calendar-known. Rolling windows
END AT t-1, never at t: `add_rolling` shifts by one before rolling, so a
28-day roll on day t summarises days t-28..t-1. Same-day cost can therefore
never reach a feature through this module.

Continuity discipline (7.3): lags and rolls are positional, so they are only
honest on a daily-continuous series. Silent gaps corrupt rolling windows —
pad_daily() first, then lag. pad_daily is gate-aware: a missing day becomes
an explicit zero only where run_status confirms the load completed; a missing
day that cannot be verified is excluded, never invented.

Price-drift discipline (A.2, redesigned July 2026): the v18 repr_30d flag
assumed a repricing is a step with a date. The December 2024 event was a
five-week glide. effective_price_drift replaces the flag with a continuous
measure — per meter, log(mean effective price over the last 14 days / mean
over the prior 28) — under which a step registers as a spike and a glide as
a sustained elevation. One feature covers both.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import assertions as A

# ---------------------------------------------------------------------------
# calendar
# ---------------------------------------------------------------------------

CALENDAR_COLS = ["dow", "day_of_month", "month",
                 "d_to_month_end", "d_to_quarter_end", "is_weekend"]


def add_calendar(frame: pd.DataFrame, date_col: str = "run_date") -> pd.DataFrame:
    """Calendar features (5.5 / 7.4). The one feature family where future
    values are legitimate, because the future is known. dow is an integer
    (Monday=0) so tree models can split on it directly; d_to_month_end and
    d_to_quarter_end are the known-future leads that let the model expect the
    quarter-end spike (the point of Appendix A.1's final example row).
    """
    out = frame.copy()
    ts = pd.to_datetime(out[date_col])
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["day_of_month"] = ts.dt.day.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    month_end = ts + pd.offsets.MonthEnd(0)
    quarter_end = ts + pd.offsets.QuarterEnd(0)
    out["d_to_month_end"] = (month_end - ts).dt.days.astype("int16")
    out["d_to_quarter_end"] = (quarter_end - ts).dt.days.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5)
    return out


# ---------------------------------------------------------------------------
# lags and rolling windows
# ---------------------------------------------------------------------------

def add_lags(
    frame: pd.DataFrame,
    group_keys: list[str],
    cols: list[str],
    lags: list[int],
    date_col: str = "run_date",
) -> pd.DataFrame:
    """Per-group lagged columns: <col>_lag<k>. Positional shift, so the frame
    must be daily-continuous within each group (pad_daily first). Sorting is
    done here, per group by date, so callers cannot get it wrong silently.
    NaN lags at each group's start are left as NaN — LightGBM handles them
    natively, and imputing would fabricate history.
    """
    out = frame.sort_values(group_keys + [date_col]).reset_index(drop=True)
    g = out.groupby(group_keys, observed=True, sort=False)
    for c in cols:
        for k in lags:
            out[f"{c}_lag{k}"] = g[c].shift(k)
    return out


def add_rolling(
    frame: pd.DataFrame,
    group_keys: list[str],
    cols: list[str],
    windows: list[int],
    date_col: str = "run_date",
    stats: tuple[str, ...] = ("mean",),
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Per-group rolling stats over windows ENDING AT t-1: <col>_roll<w> for
    mean (other stats get a _<stat> suffix). The shift(1)-then-roll order is
    the leakage guard — a window that included day t would be the target
    arriving through a side door.

    min_periods defaults to the full window (strict): a partial window is a
    different, biased statistic, and NaN is more honest than a mean of three
    days pretending to be a month. Relax deliberately, not by default.
    """
    out = frame.sort_values(group_keys + [date_col]).reset_index(drop=True)
    g = out.groupby(group_keys, observed=True, sort=False)
    for c in cols:
        for w in windows:
            mp = w if min_periods is None else min_periods
            for stat in stats:
                # transform preserves index alignment per group, so the
                # rolled values land on the right rows by construction.
                out_col = (
                    g[c].transform(
                        lambda s, w=w, mp=mp, stat=stat:
                        s.shift(1).rolling(w, min_periods=mp).agg(stat)
                    )
                )
                suffix = f"roll{w}" if stat == "mean" else f"roll{w}_{stat}"
                out[f"{c}_{suffix}"] = out_col
    return out


# ---------------------------------------------------------------------------
# daily padding (7.3)
# ---------------------------------------------------------------------------

def pad_daily(
    frame: pd.DataFrame,
    group_keys: list[str],
    zero_cols: list[str],
    date_col: str = "run_date",
    gate: pd.DataFrame | None = None,
    subscription_col: str = "subscription_id",
    spine: tuple[date, date] | None = None,
) -> pd.DataFrame:
    """Make each group's series daily-continuous, per 7.3: a missing day
    becomes an explicit zero WHERE the gate confirms the load completed, and
    an excluded observation where it did not. Silent gaps corrupt rolling
    windows; invented zeros corrupt the target. This function does neither.

    Spine scope: by default each group is padded between its OWN first and
    last observed date — right for pools, which appear and retire, where a
    shared spine would invent zeros before a pool existed. Pass `spine`
    (start, end) to pad every group over a shared range instead — right for
    the 1b frame, whose identity (pool + non-pool = total, per day) needs
    every segment present on every day the estate has rows.

      - zero_cols are filled with 0.0 on padded rows (a gated day with no
        rows for this group is a true zero: the group genuinely cost/ran
        nothing that day). All other columns are left NaN.
      - padded rows carry padded=True; original rows padded=False.
      - if `gate` is given (run_date, subscription_id, gate_complete), padded
        rows survive only where gate_complete is True for that
        subscription-day. A padded day the gate cannot verify — including
        every day before the run_status era — is dropped, because "the load
        completed and there was nothing" cannot be distinguished from "the
        load never ran". Original (observed) rows are never dropped here;
        gating observed rows is apply_gate's job, not padding's.
    """
    if frame.empty:
        out = frame.copy()
        out["padded"] = pd.Series(dtype=bool)
        return out

    out = frame.copy()
    out["padded"] = False

    pieces = [out]
    for keys, grp in out.groupby(group_keys, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        have = set(pd.to_datetime(grp[date_col]))
        if spine is not None:
            lo, hi = pd.Timestamp(spine[0]), pd.Timestamp(spine[1])
        else:
            lo, hi = min(have), max(have)
        full = pd.date_range(lo, hi, freq="D")
        missing = [d for d in full if d not in have]
        if not missing:
            continue
        pad = pd.DataFrame({date_col: [d.date() for d in missing]})
        for k, v in zip(group_keys, keys):
            pad[k] = v
        for c in zero_cols:
            pad[c] = 0.0
        pad["padded"] = True
        pieces.append(pad)

    result = pd.concat(pieces, ignore_index=True)

    if gate is not None:
        merged = result.merge(
            gate[["run_date", subscription_col, "gate_complete"]],
            left_on=[date_col, subscription_col],
            right_on=["run_date", subscription_col],
            how="left",
            suffixes=("", "_gate"),
        )
        verified = merged["gate_complete"].fillna(False).astype(bool)
        keep = (~merged["padded"]) | verified
        result = merged.loc[keep, result.columns].copy()

    return (
        result.sort_values(group_keys + [date_col]).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# effective-price drift (A.2 redesign: steps AND glides)
# ---------------------------------------------------------------------------

def effective_price_drift(
    lines: pd.DataFrame,
    group_keys: list[str],
    date_col: str = "run_date",
    meter_col: str = "meter",
    cost_col: str = "pre_tax_cost",
    usage_col: str = "usage_quantity",
    recent_window: int = 14,
    prior_window: int = 28,
) -> pd.DataFrame:
    """Continuous repricing measure per (group_keys..., day), replacing the
    v18 repr_30d step flag.

    Per meter within each group: daily effective price = sum(cost)/sum(usage)
    under the priceable mask (both strictly positive — 5.6; effective prices
    are only meaningful within a meter and are never averaged across meters
    with different units, 5.5). Then
        drift = log( mean(eff price, last `recent_window` days)
                   / mean(eff price, prior `prior_window` days) )
    computed on a daily-reindexed series so the windows are windows of TIME,
    not of billing rows. A step repricing registers as a spike in drift; the
    December 2024 five-week glide registers as a sustained elevation. One
    measure covers both.

    Meters are then combined to the group grain as a weighted average, the
    weight being each meter's trailing `prior_window`-day cost, so a large
    meter's repricing moves the feature and a trivial meter's noise does not.
    Trailing cost as the weight (not same-day cost) keeps the feature clear
    of the target.

    Returns one row per (group_keys..., date_col) with `price_drift`. The
    CALLER must lag this by one day before it enters a training frame; it is
    a same-day summary of strictly-past prices, but lagging it keeps every
    frame's feature set uniformly t-1.
    """
    df = lines.copy()
    mask = (df[usage_col] > 0) & (df[cost_col] > 0)
    df = df[mask]
    if df.empty:
        return pd.DataFrame(columns=group_keys + [date_col, "price_drift"])

    daily = (
        df.groupby(group_keys + [meter_col, date_col], observed=True)
        .agg(cost=(cost_col, "sum"), usage=(usage_col, "sum"))
        .reset_index()
    )
    daily["eff_price"] = daily["cost"] / daily["usage"]

    pieces = []
    for keys, grp in daily.groupby(group_keys + [meter_col], observed=True,
                                   sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        idx = pd.to_datetime(grp[date_col])
        s = pd.Series(grp["eff_price"].to_numpy(), index=idx).sort_index()
        w = pd.Series(grp["cost"].to_numpy(), index=idx).sort_index()
        full = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s = s.reindex(full)
        w = w.reindex(full)

        recent = s.rolling(recent_window,
                           min_periods=max(2, recent_window // 2)).mean()
        prior = s.shift(recent_window).rolling(
            prior_window, min_periods=max(2, prior_window // 2)).mean()
        drift = np.log(recent / prior)
        weight = w.fillna(0.0).rolling(prior_window, min_periods=1).sum()

        piece = pd.DataFrame({
            date_col: [d.date() for d in full],
            "meter_drift": drift.to_numpy(),
            "meter_weight": weight.to_numpy(),
        })
        for k, v in zip(group_keys + [meter_col], keys):
            piece[k] = v
        pieces.append(piece)

    per_meter = pd.concat(pieces, ignore_index=True)
    per_meter = per_meter[per_meter["meter_drift"].notna()]
    if per_meter.empty:
        return pd.DataFrame(columns=group_keys + [date_col, "price_drift"])

    per_meter["wx"] = per_meter["meter_drift"] * per_meter["meter_weight"]
    agg = (
        per_meter.groupby(group_keys + [date_col], observed=True)
        .agg(wx=("wx", "sum"), w=("meter_weight", "sum"))
        .reset_index()
    )
    agg["price_drift"] = np.where(agg["w"] > 0, agg["wx"] / agg["w"], np.nan)
    return agg[group_keys + [date_col, "price_drift"]]


# ---------------------------------------------------------------------------
# feature-set assembly
# ---------------------------------------------------------------------------

def feature_columns(frame: pd.DataFrame,
                    exclude: list[str]) -> list[str]:
    """The frame's feature list: every column not excluded (keys, target,
    labels), checked through assert_no_same_day_cost so an unlagged cost
    column can never be declared a feature. Returns the list; also the place
    a reviewer looks to see exactly what the model is allowed to know.
    """
    cols = [c for c in frame.columns if c not in exclude]
    A.assert_no_same_day_cost(cols, "feature_columns")
    return cols
