"""
baselines.py  --  the F3 baselines: seasonal naive, 3-month trend (the
incumbent), rolling 28-day median. The models every real model must beat,
on the same folds, before it earns production.

The forecaster interface (shared with the future GBMs):

    model.fit(train, spec)              # train: rows strictly before origin
    model.predict(test, spec) -> frame  # one row per test row, with columns
                                        # q05, q50, q95

Honesty rule: these baselines FREEZE AT THE ORIGIN. A monthly walk-forward
scores a whole month ahead, so a baseline that peeked at the test month's
actuals (e.g. seasonal naive reading t-7 from inside the month) would be a
recursive cheat the doc rules out (5.4: direct horizons, features lagged at
least the horizon). Each rule therefore predicts every day of the test month
from training data alone: the seasonal naive repeats the last observed value
per weekday; the trend extrapolates a line fitted before the origin; the
median holds the last 28 training days' median flat.

Intervals: baselines are point rules, but the harness scores quantiles, so
each baseline carries empirical intervals — the 5th and 95th percentiles of
its OWN in-training residuals (per group, pooled fallback when a group is
thin), added to the point. This is the honest cheap interval: if the GBM's
intervals are no sharper than residual-quantile bands around a naive rule,
the quantile machinery has earned nothing.

Approved stack only: pandas + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_GROUP_RESIDUALS = 20  # below this, a group borrows the pooled residuals


@dataclass
class FrameSpec:
    """What the harness and models need to know about a frame's shape.
    Built from frame.attrs by spec_for_frame; carried explicitly so a model
    can never guess the grain.
    """
    group_keys: list[str]
    target: str = "cost"
    date_col: str = "run_date"


def spec_for_frame(frame: pd.DataFrame) -> FrameSpec:
    """Read the spec the frame declares about itself (frames.py sets
    group_keys/target in attrs). Raises if the frame does not declare them:
    guessing grain is how fanouts happen.
    """
    gk = frame.attrs.get("group_keys")
    if not gk:
        raise ValueError(
            "frame declares no group_keys in .attrs; build it with "
            "frames.build_frame_* or pass an explicit FrameSpec")
    return FrameSpec(group_keys=list(gk),
                     target=frame.attrs.get("target", "cost"))


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s)


def _row_key(row, group_keys: list[str]) -> tuple:
    return tuple(row[k] for k in group_keys)


class _ResidualIntervals:
    """Shared interval machinery: store per-group and pooled residual
    quantiles at fit time; wrap a point prediction into q05/q50/q95.
    Residuals come from each rule's own in-sample errors (subclass supplies
    them), so the interval width reflects how wrong THIS rule tends to be.
    """

    def _store_residuals(self, resid: pd.DataFrame,
                         spec: FrameSpec) -> None:
        resid = resid.dropna(subset=["resid"])
        self._pooled_q = (
            (float(resid["resid"].quantile(0.05)),
             float(resid["resid"].quantile(0.95)))
            if len(resid) else (0.0, 0.0)
        )
        self._group_q = {}
        for keys, grp in resid.groupby(spec.group_keys, observed=True,
                                       sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            if len(grp) >= MIN_GROUP_RESIDUALS:
                self._group_q[keys] = (
                    float(grp["resid"].quantile(0.05)),
                    float(grp["resid"].quantile(0.95)))

    def _wrap(self, test: pd.DataFrame, point: pd.Series,
              spec: FrameSpec) -> pd.DataFrame:
        out = test[spec.group_keys + [spec.date_col]].copy()
        out["q50"] = point.to_numpy()
        lo, hi = [], []
        for _, row in out.iterrows():
            ql, qh = self._group_q.get(_row_key(row, spec.group_keys),
                                       self._pooled_q)
            lo.append(ql)
            hi.append(qh)
        out["q05"] = out["q50"] + np.array(lo)
        out["q95"] = out["q50"] + np.array(hi)
        # cost is non-negative; an interval below zero is noise, not a claim
        out[["q05", "q50", "q95"]] = out[["q05", "q50", "q95"]].clip(lower=0)
        return out


class SeasonalNaive(_ResidualIntervals):
    """F3 baseline: same weekday last week. Frozen at the origin: for each
    group and each weekday, the LAST training observation of that weekday is
    repeated across the test month. Residuals: cost minus cost seven days
    earlier, within training.
    """

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame, spec: FrameSpec) -> None:
        t = train.copy()
        t["_dow"] = _dates(t[spec.date_col]).dt.dayofweek
        t = t.sort_values(spec.date_col)
        last = t.groupby(spec.group_keys + ["_dow"], observed=True,
                         sort=False)[spec.target].last()
        self._lookup = last.to_dict()

        t = t.sort_values(spec.group_keys + [spec.date_col])
        t["resid"] = (t[spec.target]
                      - t.groupby(spec.group_keys, observed=True,
                                  sort=False)[spec.target].shift(7))
        self._store_residuals(t[spec.group_keys + ["resid"]], spec)

    def predict(self, test: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
        dow = _dates(test[spec.date_col]).dt.dayofweek
        vals = []
        for (_, row), d in zip(test.iterrows(), dow):
            vals.append(self._lookup.get(
                _row_key(row, spec.group_keys) + (d,), np.nan))
        point = pd.Series(vals, index=test.index, dtype=float)
        return self._wrap(test, point, spec)


class ThreeMonthTrend(_ResidualIntervals):
    """F3 baseline reproducing the incumbent: a linear trend fitted per
    group on the trailing window (default 90 days) before the origin,
    extrapolated across the test month, floored at zero because a negative
    cost forecast is not a forecast. Residuals: the fit's own in-window
    errors. This is the model the charter says is 'maintained by hand';
    beating it is the programme's first claim.
    """

    name = "trend_3m"

    def __init__(self, window_days: int = 90):
        self.window_days = window_days

    def fit(self, train: pd.DataFrame, spec: FrameSpec) -> None:
        t = train.copy()
        t["_ts"] = _dates(t[spec.date_col])
        cutoff = t["_ts"].max() - pd.Timedelta(days=self.window_days - 1)
        t = t[t["_ts"] >= cutoff]

        self._fits = {}
        resid_rows = []
        for keys, grp in t.groupby(spec.group_keys, observed=True,
                                   sort=False):
            key = keys if isinstance(keys, tuple) else (keys,)
            x = grp["_ts"].map(pd.Timestamp.toordinal).to_numpy(float)
            y = grp[spec.target].to_numpy(float)
            if len(grp) >= 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope = 0.0
                intercept = float(y[0]) if len(y) else np.nan
            self._fits[key] = (slope, intercept)
            fitted = slope * x + intercept
            r = grp[spec.group_keys].copy()
            r["resid"] = y - fitted
            resid_rows.append(r)
        self._store_residuals(
            pd.concat(resid_rows, ignore_index=True) if resid_rows
            else pd.DataFrame(columns=spec.group_keys + ["resid"]),
            spec)

    def predict(self, test: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
        x = _dates(test[spec.date_col]).map(pd.Timestamp.toordinal)
        vals = []
        for (_, row), xo in zip(test.iterrows(), x):
            key = _row_key(row, spec.group_keys)
            if key in self._fits:
                slope, intercept = self._fits[key]
                vals.append(slope * xo + intercept)
            else:
                vals.append(np.nan)
        point = pd.Series(vals, index=test.index, dtype=float).clip(lower=0)
        return self._wrap(test, point, spec)


class RollingMedian(_ResidualIntervals):
    """F3 baseline: the median of the last `window` training days per group,
    held flat across the test month. Robust to spikes by construction, which
    is exactly why it is a fair floor for a spiky target. Residuals: cost
    minus the trailing median at each training day.
    """

    name = "rolling_median_28"

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, train: pd.DataFrame, spec: FrameSpec) -> None:
        t = train.sort_values(spec.group_keys + [spec.date_col]).copy()
        g = t.groupby(spec.group_keys, observed=True, sort=False)
        self._level = {
            (keys if isinstance(keys, tuple) else (keys,)):
            float(grp[spec.target].tail(self.window).median())
            for keys, grp in g
        }
        trailing = g[spec.target].transform(
            lambda s: s.shift(1).rolling(
                self.window, min_periods=max(2, self.window // 4)).median())
        t["resid"] = t[spec.target] - trailing
        self._store_residuals(t[spec.group_keys + ["resid"]], spec)

    def predict(self, test: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
        vals = [self._level.get(_row_key(row, spec.group_keys), np.nan)
                for _, row in test.iterrows()]
        point = pd.Series(vals, index=test.index, dtype=float)
        return self._wrap(test, point, spec)


def all_baselines() -> list:
    """The F3 trio, fresh instances, in charter order."""
    return [ThreeMonthTrend(), SeasonalNaive(), RollingMedian()]
