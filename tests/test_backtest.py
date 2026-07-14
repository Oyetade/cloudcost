"""Baselines and the walk-forward harness (build-order item 4).

The tests that matter: the fold boundary (nothing from the test month can
reach training), the honest-origin refusal (a gated model cannot originate
before its regime has history), exact reproduction of each baseline's rule
on hand-computable series, and metric values checked by hand.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from catpipe import baselines as B
from catpipe import harness as H


def _frame(start: date, values: list[float], grp: str = "a",
           regime: str = "featured_gated") -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range(len(values))]
    f = pd.DataFrame({"run_date": days, "grp": [grp] * len(values),
                      "cost": values,
                      "data_regime": [regime] * len(values)})
    return f


def _spec():
    return B.FrameSpec(group_keys=["grp"])


class TestSeasonalNaive:
    def test_repeats_last_observed_weekday(self):
        # 21 days of a weekly pattern 0..6 by weekday index
        start = date(2025, 6, 2)  # a Monday
        vals = [float((start + timedelta(days=i)).weekday())
                for i in range(21)]
        train = _frame(start, vals)
        test = _frame(date(2025, 6, 23), [0.0] * 7)  # next Monday onward
        m = B.SeasonalNaive()
        m.fit(train, _spec())
        p = m.predict(test, _spec())
        expected = [float(d.weekday()) for d in test["run_date"]]
        assert p["q50"].tolist() == expected

    def test_perfect_weekly_pattern_gives_zero_width_intervals(self):
        start = date(2025, 6, 2)
        vals = [float((start + timedelta(days=i)).weekday())
                for i in range(28)]
        m = B.SeasonalNaive()
        m.fit(_frame(start, vals), _spec())
        p = m.predict(_frame(date(2025, 6, 30), [0.0] * 7), _spec())
        # residuals of a perfect weekly pattern are all zero
        assert (p["q95"] - p["q05"]).abs().max() == pytest.approx(0.0)

    def test_unseen_group_yields_nan_not_invention(self):
        m = B.SeasonalNaive()
        m.fit(_frame(date(2025, 6, 2), [1.0] * 14, grp="a"), _spec())
        p = m.predict(_frame(date(2025, 6, 16), [0.0], grp="b"), _spec())
        assert p["q50"].isna().all()


class TestThreeMonthTrend:
    def test_extrapolates_linear_series_exactly(self):
        # cost = 10 + 2*i: the trend must recover slope 2 and continue it
        train = _frame(date(2025, 1, 1), [10.0 + 2 * i for i in range(90)])
        test = _frame(date(2025, 4, 1), [0.0] * 5)
        m = B.ThreeMonthTrend()
        m.fit(train, _spec())
        p = m.predict(test, _spec())
        expected = [10.0 + 2 * (90 + i) for i in range(5)]
        assert p["q50"].to_numpy() == pytest.approx(expected)
        # a perfect fit has zero residuals, hence zero-width intervals
        assert (p["q95"] - p["q05"]).abs().max() == pytest.approx(0.0)

    def test_declining_trend_floors_at_zero(self):
        train = _frame(date(2025, 1, 1), [90.0 - i for i in range(90)])
        far = _frame(date(2025, 8, 1), [0.0] * 3)  # extrapolates negative
        m = B.ThreeMonthTrend()
        m.fit(train, _spec())
        p = m.predict(far, _spec())
        assert (p["q50"] >= 0).all()

    def test_uses_only_trailing_window(self):
        # 90 flat days at 100 after 90 rising days: the 90-day window must
        # see only the flat regime and predict ~100, not a blended slope
        vals = [float(i) for i in range(90)] + [100.0] * 90
        m = B.ThreeMonthTrend(window_days=90)
        m.fit(_frame(date(2025, 1, 1), vals), _spec())
        p = m.predict(_frame(date(2025, 7, 1), [0.0] * 3), _spec())
        assert p["q50"].to_numpy() == pytest.approx([100.0] * 3, abs=1e-6)


class TestRollingMedian:
    def test_holds_last_28_day_median_flat(self):
        vals = [1.0] * 30 + [5.0] * 28  # last 28 days are all 5
        m = B.RollingMedian()
        m.fit(_frame(date(2025, 1, 1), vals), _spec())
        p = m.predict(_frame(date(2025, 3, 1), [0.0] * 4), _spec())
        assert (p["q50"] == 5.0).all()

    def test_robust_to_a_spike(self):
        vals = [10.0] * 55 + [10_000.0]  # one closing spike
        m = B.RollingMedian()
        m.fit(_frame(date(2025, 1, 1), vals), _spec())
        p = m.predict(_frame(date(2025, 2, 26), [0.0]), _spec())
        assert p["q50"].iloc[0] == 10.0


class TestHarnessFolds:
    def _long_frame(self, days=400, start=date(2024, 6, 1)):
        rng = np.random.default_rng(7)
        vals = list(100 + rng.normal(0, 1, days))
        return _frame(start, vals)

    def test_no_test_month_row_reaches_training(self):
        # a model that records what it was trained on: the leak detector
        class Spy:
            name = "spy"

            def fit(self, train, spec):
                self.train_max = max(train["run_date"])

            def predict(self, test, spec):
                assert self.train_max < min(test["run_date"])
                out = test[spec.group_keys + ["run_date"]].copy()
                out[["q05", "q50", "q95"]] = 0.0
                return out

        f = self._long_frame()
        H.walk_forward(f, Spy(), spec=_spec())  # the assert inside predict is the test

    def test_dishonest_origin_refused(self):
        f = self._long_frame(days=400)
        # frame starts 2024-06-01; an origin 60 days in cannot carry a
        # 180-day training window
        with pytest.raises(H.BacktestError, match="refused"):
            H.walk_forward(f, B.RollingMedian(), spec=_spec(),
                           origins=[date(2024, 8, 1)])

    def test_regime_filter_moves_the_honest_start(self):
        # 200 ungated days then 400 gated days: with the default regime
        # filter, origins must be computed from the GATED start, not the
        # frame's first row
        ungated = _frame(date(2023, 6, 1), [50.0] * 200,
                         regime="featured_ungated")
        gated = _frame(date(2023, 12, 18), [100.0] * 400)
        f = pd.concat([ungated, gated], ignore_index=True)
        prepared = H.prepare(f)
        origins = H.monthly_origins(prepared)
        assert min(origins) >= date(2024, 6, 1)

    def test_short_frame_yields_no_origin_and_says_so(self):
        f = self._long_frame(days=100)
        with pytest.raises(H.BacktestError, match="no honest"):
            H.run_models(f, [B.RollingMedian()], spec=_spec())

    def test_mask_restricts_the_window(self):
        f = self._long_frame(days=400)
        mask = pd.to_datetime(f["run_date"]).dt.date >= date(2024, 9, 1)
        prepared = H.prepare(f, mask=mask)
        assert prepared["run_date"].min() == date(2024, 9, 1)


class TestMetrics:
    def test_pinball_by_hand(self):
        y = pd.Series([10.0, 10.0])
        q = pd.Series([8.0, 12.0])
        # tau=0.5: mean(0.5*2, 0.5*2) = 1.0
        assert H.pinball_loss(y, q, 0.5) == pytest.approx(1.0)
        # tau=0.05: under-forecast costs tau, over costs (1-tau):
        # mean(0.05*2, 0.95*2) = 1.0
        assert H.pinball_loss(y, q, 0.05) == pytest.approx(1.0)
        # tau=0.95 asymmetry: mean(0.95*2, 0.05*2) = 1.0
        assert H.pinball_loss(y, q, 0.95) == pytest.approx(1.0)
        # and a q that sits ON y scores zero
        assert H.pinball_loss(y, y, 0.5) == pytest.approx(0.0)

    def test_evaluate_by_hand(self):
        ledger = pd.DataFrame({
            "model": "m", "grp": "a",
            "run_date": [date(2025, 1, 1), date(2025, 1, 2)],
            "origin": date(2025, 1, 1),
            "y_true": [100.0, 100.0],
            "q05": [90.0, 90.0],
            "q50": [110.0, 100.0],
            "q95": [120.0, 95.0],   # second actual ABOVE q95
        })
        e = H.evaluate(ledger, group_keys=["grp"]).iloc[0]
        assert e["mae_daily"] == pytest.approx(5.0)
        # monthly: |210 - 200| / 200
        assert e["monthly_pct_err"] == pytest.approx(0.05)
        assert e["coverage_90"] == pytest.approx(0.5)

    def test_unscored_rows_reported_not_hidden(self):
        ledger = pd.DataFrame({
            "model": "m", "grp": ["a", "b"],
            "run_date": date(2025, 1, 1), "origin": date(2025, 1, 1),
            "y_true": [1.0, 2.0],
            "q05": [1.0, np.nan], "q50": [1.0, np.nan],
            "q95": [1.0, np.nan],
        })
        e = H.evaluate(ledger, group_keys=["grp"]).iloc[0]
        assert e["n"] == 2 and e["n_scored"] == 1


class TestEndToEnd:
    def test_baselines_ranked_on_a_seasonal_series(self):
        # a strong weekly pattern: the seasonal naive must beat the flat
        # median and the trend on daily MAE, and the summary must carry
        # every metric column
        start = date(2024, 1, 1)
        days = 550
        vals = [100.0 + 30.0 * ((start + timedelta(days=i)).weekday() >= 5)
                for i in range(days)]
        f = _frame(start, vals)
        summary, ledger = H.run_models(f, B.all_baselines(), spec=_spec())
        assert set(summary["model"]) == {
            "seasonal_naive", "trend_3m", "rolling_median_28"}
        best = summary.iloc[0]
        assert best["model"] == "seasonal_naive"
        assert best["mae_daily"] == pytest.approx(0.0, abs=1e-9)
        for col in ["monthly_pct_err", "pinball_05", "pinball_95",
                    "coverage_90"]:
            assert col in summary.columns
        # the ledger is per (group, day, origin, model), reconstructible
        assert {"y_true", "q05", "q50", "q95", "origin",
                "model"} <= set(ledger.columns)

    def test_two_groups_scored_independently(self):
        f = pd.concat([
            _frame(date(2024, 1, 1), [100.0] * 400, grp="a"),
            _frame(date(2024, 1, 1), [500.0] * 400, grp="b"),
        ], ignore_index=True)
        f.attrs["group_keys"] = ["grp"]
        summary, ledger = H.run_models(f, [B.RollingMedian()])
        assert summary.iloc[0]["mae_daily"] == pytest.approx(0.0, abs=1e-9)
        got = ledger.groupby("grp")["q50"].first()
        assert got["a"] == 100.0 and got["b"] == 500.0


class TestMonthlyErrorVariants:
    def _ledger(self, rows):
        df = pd.DataFrame(rows)
        df["model"] = "m"
        df["run_date"] = date(2025, 1, 15)
        df["origin"] = date(2025, 1, 1)
        for q in ("q05", "q95"):
            df[q] = df["q50"]
        return df

    def test_offsetting_errors_cancel_at_estate_not_in_wape(self):
        # two groups, equal and opposite 10% errors: finance sees a perfect
        # month (estate 0), attribution sees 10% misallocated (wape 0.10)
        led = self._ledger([
            dict(grp="a", y_true=100.0, q50=110.0),
            dict(grp="b", y_true=100.0, q50=90.0),
        ])
        e = H.evaluate(led, group_keys=["grp"]).iloc[0]
        assert e["monthly_pct_err_estate"] == pytest.approx(0.0)
        assert e["monthly_wape"] == pytest.approx(0.10)
        assert e["monthly_pct_err"] == pytest.approx(0.10)

    def test_tiny_group_inflates_only_the_unweighted_mean(self):
        # the 206% lesson in miniature: a 1000-spend pool at 1% error and
        # a 1-spend pool at 100% error. The unweighted mean screams 50.5%;
        # the numbers finance and attribution care about both say ~1%.
        led = self._ledger([
            dict(grp="big", y_true=1000.0, q50=1010.0),
            dict(grp="tiny", y_true=1.0, q50=2.0),
        ])
        e = H.evaluate(led, group_keys=["grp"]).iloc[0]
        assert e["monthly_pct_err"] == pytest.approx(0.505)
        assert e["monthly_wape"] == pytest.approx(11.0 / 1001.0)
        assert e["monthly_pct_err_estate"] == pytest.approx(11.0 / 1001.0)

    def test_single_group_all_three_agree(self):
        led = self._ledger([dict(grp="a", y_true=200.0, q50=210.0)])
        e = H.evaluate(led, group_keys=["grp"]).iloc[0]
        assert e["monthly_pct_err"] == pytest.approx(0.05)
        assert e["monthly_wape"] == pytest.approx(0.05)
        assert e["monthly_pct_err_estate"] == pytest.approx(0.05)

    def test_signed_bias_shows_direction_and_cancels_when_unbiased(self):
        # biased-low model: bias negative, magnitude = estate error
        low = self._ledger([
            dict(grp="a", y_true=100.0, q50=90.0),
            dict(grp="b", y_true=100.0, q50=95.0),
        ])
        e = H.evaluate(low, group_keys=["grp"]).iloc[0]
        assert e["monthly_bias"] == pytest.approx(-0.075)
        assert e["monthly_pct_err_estate"] == pytest.approx(0.075)
        # unbiased offsetting model: bias zero while wape is not
        off = self._ledger([
            dict(grp="a", y_true=100.0, q50=110.0),
            dict(grp="b", y_true=100.0, q50=90.0),
        ])
        e = H.evaluate(off, group_keys=["grp"]).iloc[0]
        assert e["monthly_bias"] == pytest.approx(0.0)
