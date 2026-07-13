"""QuantileGBM (build-order item 5). Every test here skips cleanly when
lightgbm is not installed, because the library is pending software approval
and the rest of catpipe must not depend on it. The module import itself is
tested unconditionally — lazy import is the contract.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from catpipe import baselines as B
from catpipe import harness as H
from catpipe import models as M  # must import WITHOUT lightgbm installed

lgb = pytest.importorskip("lightgbm")

FAST = dict(num_boost_round=300, early_stopping_rounds=25,
            params={"learning_rate": 0.1})


def _make_frame(days=420, groups=("eodpool",), noise=5.0, seed=7,
                weekend_lift=0.0, group_level=None):
    """Synthetic frame with honestly lagged features: dow, cost_lag1,
    cost_lag7 computed from the series itself, one row per group-day.
    """
    rng = np.random.default_rng(seed)
    start = date(2024, 6, 2)  # a Sunday
    rows = []
    for g in groups:
        level = (group_level or {}).get(g, 400.0)
        series = []
        for i in range(days):
            d = start + timedelta(days=i)
            mu = level + weekend_lift * (d.weekday() >= 5)
            series.append((d, mu + rng.normal(0, noise)))
        for i, (d, y) in enumerate(series):
            rows.append(dict(
                run_date=d, pool=g, cost=y,
                dow=d.weekday(),
                cost_lag1=series[i - 1][1] if i >= 1 else np.nan,
                cost_lag7=series[i - 7][1] if i >= 7 else np.nan,
                data_regime="featured_gated",
            ))
    f = pd.DataFrame(rows)
    f.attrs["group_keys"] = ["pool"]
    f.attrs["target"] = "cost"
    f.attrs["feature_cols"] = ["dow", "cost_lag1", "cost_lag7", "pool"]
    f.attrs["categorical_cols"] = ["pool"]
    return f


def _spec():
    return B.FrameSpec(group_keys=["pool"])


class TestConstruction:
    def test_for_frame_reads_the_frame_declaration(self):
        f = _make_frame(days=30)
        m = M.QuantileGBM.for_frame(f, **FAST)
        assert m.features == ["dow", "cost_lag1", "cost_lag7", "pool"]
        assert m.categoricals == ["pool"]

    def test_undeclared_frame_refused(self):
        f = pd.DataFrame({"run_date": [date(2025, 1, 1)], "cost": [1.0]})
        with pytest.raises(ValueError, match="feature_cols"):
            M.QuantileGBM.for_frame(f)

    def test_median_is_mandatory(self):
        with pytest.raises(ValueError, match="median"):
            M.QuantileGBM(features=["dow"], quantiles=(0.05, 0.95))


class TestLearning:
    def test_learns_weekly_pattern_beating_flat_median(self):
        # weekend lift of 100 with noise 5: a model that sees dow must beat
        # a flat 28-day median by a wide margin on the same folds
        f = _make_frame(days=420, weekend_lift=100.0, noise=5.0)
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        med = B.RollingMedian()
        summary, _ = H.run_models(f, [gbm, med])
        s = summary.set_index("model")
        assert s.loc["quantile_gbm", "mae_daily"] < \
            0.5 * s.loc["rolling_median_28", "mae_daily"]

    def test_quantiles_never_cross_and_cover(self):
        f = _make_frame(days=420, weekend_lift=50.0, noise=20.0)
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        _, ledger = H.run_models(f, [gbm])
        assert (ledger["q05"] <= ledger["q50"] + 1e-9).all()
        assert (ledger["q50"] <= ledger["q95"] + 1e-9).all()
        cov = ((ledger["y_true"] >= ledger["q05"])
               & (ledger["y_true"] <= ledger["q95"])).mean()
        assert 0.75 <= cov <= 0.99  # near the 0.90 target, not degenerate

    def test_pool_categorical_separates_levels(self):
        f = _make_frame(days=300, groups=("small", "big"), noise=2.0,
                        group_level={"small": 100.0, "big": 1000.0})
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        spec = _spec()
        train = f[pd.to_datetime(f["run_date"]).dt.date < date(2025, 3, 1)]
        test = f[pd.to_datetime(f["run_date"]).dt.date >= date(2025, 3, 1)]
        # drop the lag features so ONLY the categorical can explain the gap
        for frame in (train, test):
            frame.attrs.update(f.attrs)
        gbm = M.QuantileGBM(features=["dow", "pool"],
                            categoricals=["pool"], **FAST)
        gbm.fit(train, spec)
        p = gbm.predict(test, spec).groupby("pool")["q50"].mean()
        assert p["big"] > 5 * p["small"]


class TestRobustness:
    def test_unseen_category_predicts_not_crashes(self):
        f = _make_frame(days=300)
        spec = _spec()
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        gbm.fit(f, spec)
        new = f.tail(3).copy()
        new["pool"] = "brand_new_pool"
        p = gbm.predict(new, spec)
        assert p["q50"].notna().all()  # mapped to missing, still scored

    def test_zero_cost_padded_days_safe_under_asinh(self):
        f = _make_frame(days=300, noise=1.0)
        f.loc[f.index[-40:], "cost"] = 0.0  # a padded-zero tail
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        gbm.fit(f, _spec())
        p = gbm.predict(f.tail(20), _spec())
        assert np.isfinite(p[["q05", "q50", "q95"]].to_numpy()).all()
        assert (p["q05"] >= 0).all()

    def test_missing_declared_feature_refused(self):
        f = _make_frame(days=60)
        gbm = M.QuantileGBM(features=["dow", "not_a_column"], **FAST)
        with pytest.raises(ValueError, match="missing declared"):
            gbm.fit(f, _spec())

    def test_feature_importance_available_after_fit(self):
        f = _make_frame(days=300, weekend_lift=80.0)
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        gbm.fit(f, _spec())
        imp = gbm.feature_importance()
        assert set(imp.columns) == {"feature", "gain"}
        assert imp.iloc[0]["gain"] > 0


class TestHarnessIntegration:
    def test_gbm_and_baselines_share_identical_folds(self):
        f = _make_frame(days=420, weekend_lift=60.0)
        gbm = M.QuantileGBM.for_frame(f, **FAST)
        summary, ledger = H.run_models(f, [gbm] + B.all_baselines())
        assert set(summary["model"]) == {
            "quantile_gbm", "seasonal_naive", "trend_3m",
            "rolling_median_28"}
        # identical folds: every model scored on exactly the same
        # (day, origin) set
        per_model = ledger.groupby("model").apply(
            lambda m: set(zip(m["run_date"], m["origin"])),
            include_groups=False)
        assert len(set(map(frozenset, per_model))) == 1
