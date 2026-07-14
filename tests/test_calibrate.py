"""Conformal calibration (CQR). The load-bearing tests: the (n+1)
finite-sample quantile computed by hand, an under-covering model restored to
target coverage on folds the margin never saw, an over-covering model
shrunk not just widened, the scaled variant adapting to per-day dispersion,
and the thin-group pooled fallback.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from catpipe import baselines as B
from catpipe import calibrate as C
from catpipe import harness as H


def _spec():
    return B.FrameSpec(group_keys=["grp"])


class _FixedBandModel:
    """A deliberately miscalibrated forecaster: predicts the true mean as
    the median with a FIXED half-width band. Too narrow => under-covers;
    too wide => over-covers. The perfect test subject.
    """

    def __init__(self, level: float, half_width: float, name="fixed"):
        self.level = level
        self.half_width = half_width
        self.name = name

    def fit(self, train, spec):
        pass

    def predict(self, test, spec):
        out = test[spec.group_keys + [spec.date_col]].copy()
        out["q50"] = self.level
        out["q05"] = self.level - self.half_width
        out["q95"] = self.level + self.half_width
        return out


def _noise_frame(days=500, sigma=10.0, level=100.0, grp="a", seed=3):
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    return pd.DataFrame({
        "run_date": [start + timedelta(days=i) for i in range(days)],
        "grp": grp,
        "cost": level + rng.normal(0, sigma, days),
        "data_regime": "featured_gated",
    })


class TestFiniteSampleQuantile:
    def test_n_plus_one_correction_by_hand(self):
        # n=9 scores 1..9, coverage 0.9: k = ceil(10 * 0.9) = 9 -> 9th
        # smallest = 9. The PLAIN 0.9 quantile of 1..9 is 8.2; the conformal
        # one is deliberately more conservative.
        assert C.finite_sample_quantile(np.arange(1.0, 10.0), 0.9) == 9.0

    def test_k_exceeding_n_returns_max(self):
        # n=5, coverage 0.9: k = ceil(6*0.9) = 6 > 5 -> guarantee
        # unattainable; the max score is returned, and conformal_margins
        # flags guaranteed=False
        assert C.finite_sample_quantile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                                        0.9) == 5.0

    def test_empty_scores_zero_margin(self):
        assert C.finite_sample_quantile(np.array([]), 0.9) == 0.0


class TestScores:
    def test_sign_convention(self):
        y = pd.Series([10.0, 10.0, 10.0])
        q05 = pd.Series([8.0, 12.0, 9.0])
        q95 = pd.Series([12.0, 14.0, 9.5])
        s = C.cqr_scores(y, q05, q95, scaled=False)
        assert s.iloc[0] == -2.0   # inside with 2 of room
        assert s.iloc[1] == 2.0    # escaped below by 2
        assert s.iloc[2] == 0.5    # escaped above by 0.5


class TestWrapperCoverage:
    def _run(self, half_width, sigma=10.0):
        f = _noise_frame(sigma=sigma)
        wrapped = C.ConformalWrapper(
            _FixedBandModel(level=100.0, half_width=half_width),
            refit_on_full=False)
        ledger = H.walk_forward(f, wrapped, spec=_spec())
        cov = ((ledger["y_true"] >= ledger["q05"])
               & (ledger["y_true"] <= ledger["q95"])).mean()
        return cov, ledger

    def test_undercovering_model_restored_to_target(self):
        # half-width 5 on sigma-10 noise covers ~38% raw; calibrated
        # coverage on UNSEEN folds must reach ~0.90
        cov, _ = self._run(half_width=5.0)
        assert cov >= 0.85

    def test_overcovering_model_shrunk_not_left_bloated(self):
        # half-width 50 on sigma-10 noise covers ~100% raw with a uselessly
        # wide band; the negative CQR margin must shrink it while holding
        # the target
        cov, ledger = self._run(half_width=50.0)
        assert cov >= 0.85
        width = (ledger["q95"] - ledger["q05"]).mean()
        assert width < 80.0  # was 100 uncalibrated

    def test_median_never_touched(self):
        f = _noise_frame()
        wrapped = C.ConformalWrapper(
            _FixedBandModel(level=100.0, half_width=5.0),
            refit_on_full=False)
        ledger = H.walk_forward(f, wrapped, spec=_spec())
        assert (ledger["q50"] == 100.0).all()
        assert (ledger["q05"] <= ledger["q50"]).all()
        assert (ledger["q95"] >= ledger["q50"]).all()

    def test_thin_tail_degrades_loudly_to_inner(self):
        f = _noise_frame(days=250)
        wrapped = C.ConformalWrapper(
            _FixedBandModel(level=100.0, half_width=5.0),
            calib_tail_days=90, min_calib_rows=10_000,  # unattainable
            refit_on_full=False)
        wrapped.fit(f, _spec())
        assert wrapped.calibration_report["calibrated"] is False
        p = wrapped.predict(f.tail(5), _spec())
        assert ((p["q95"] - p["q05"]) == 10.0).all()  # inner band unchanged


class TestScaledVariant:
    def test_wide_days_widened_more_in_absolute_terms(self):
        preds = pd.DataFrame({
            "grp": ["a", "a"], "run_date": [date(2025, 1, 1),
                                            date(2025, 1, 2)],
            "q05": [90.0, 50.0], "q50": [100.0, 100.0],
            "q95": [110.0, 150.0],   # widths 20 and 100
        })
        margins = {"target_coverage": 0.9, "scaled": True, "n": 100,
                   "guaranteed": True, "pooled": 0.5, "groups": {}}
        out = C.apply_margins(preds, margins, ["grp"])
        w = out["q95"] - out["q05"]
        # scaled margin 0.5: widths grow by 2*0.5*width -> doubled
        assert w.iloc[0] == pytest.approx(40.0)
        assert w.iloc[1] == pytest.approx(200.0)


class TestMarginsFromLedger:
    def _ledger(self, n_a=200, n_b=10, seed=5):
        rng = np.random.default_rng(seed)
        rows = []
        for grp, n, sigma in (("a", n_a, 10.0), ("b", n_b, 40.0)):
            y = 100 + rng.normal(0, sigma, n)
            rows.append(pd.DataFrame({
                "grp": grp, "y_true": y,
                "q05": 95.0, "q50": 100.0, "q95": 105.0,
            }))
        return pd.concat(rows, ignore_index=True)

    def test_per_group_where_dense_pooled_where_thin(self):
        m = C.conformal_margins(self._ledger(), ["grp"],
                                min_group_scores=30)
        assert ("a",) in m["groups"]
        assert ("b",) not in m["groups"]   # 10 rows: pooled fallback
        assert m["guaranteed"] is True

    def test_guarantee_flag_false_on_tiny_ledger(self):
        m = C.conformal_margins(self._ledger(n_a=5, n_b=0), ["grp"])
        assert m["guaranteed"] is False

    def test_apply_from_ledger_restores_future_coverage(self):
        # margins fitted on one ledger, applied to fresh draws from the
        # same process: the A.4 nightly pattern in miniature
        rng = np.random.default_rng(9)
        m = C.conformal_margins(self._ledger(n_a=500, n_b=0), ["grp"],
                                scaled=False)
        future_y = 100 + rng.normal(0, 10.0, 500)
        preds = pd.DataFrame({
            "grp": "a", "run_date": date(2025, 6, 1),
            "q05": 95.0, "q50": 100.0, "q95": 105.0,
            "y_true": future_y,
        })
        out = C.apply_margins(preds, m, ["grp"])
        cov = ((out["y_true"] >= out["q05"])
               & (out["y_true"] <= out["q95"])).mean()
        assert cov >= 0.85


class TestWithGBM:
    def test_calibrated_gbm_coverage_improves_toward_target(self):
        from catpipe import models as M
        from tests.test_models import _make_frame, FAST

        f = _make_frame(days=460, weekend_lift=50.0, noise=20.0, seed=13)
        raw = M.QuantileGBM.for_frame(f, **FAST)
        cal = C.ConformalWrapper(M.QuantileGBM.for_frame(f, **FAST))
        summary, _ = H.run_models(f, [raw, cal])
        s = summary.set_index("model")
        raw_cov = s.loc["quantile_gbm", "coverage_90"]
        cal_cov = s.loc["conformal_quantile_gbm", "coverage_90"]
        assert cal_cov >= raw_cov
        assert cal_cov >= 0.85
        assert abs(cal_cov - 0.90) <= abs(raw_cov - 0.90)
