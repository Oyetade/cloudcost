"""Feature factory: the leakage and continuity disciplines, enforced.

The tests that matter most here are negative-space tests: a rolling window
must NOT see day t, a lag must NOT bleed across groups, a padded day must
NOT survive without a gate to vouch for it, and a price glide must register
where a step flag would stay silent.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from catpipe import assertions as A
from catpipe import feature_factory as FF


def _daily(group: str, start: date, costs: list[float]) -> pd.DataFrame:
    days = pd.date_range(start, periods=len(costs), freq="D")
    return pd.DataFrame({
        "run_date": [d.date() for d in days],
        "grp": group,
        "cost": costs,
    })


class TestLags:
    def test_lag_shifts_within_group(self):
        f = _daily("a", date(2025, 1, 1), [1.0, 2.0, 3.0])
        out = FF.add_lags(f, ["grp"], ["cost"], [1])
        assert out["cost_lag1"].tolist()[1:] == [1.0, 2.0]
        assert pd.isna(out["cost_lag1"].iloc[0])

    def test_no_bleed_across_groups(self):
        f = pd.concat([
            _daily("a", date(2025, 1, 1), [1.0, 2.0]),
            _daily("b", date(2025, 1, 1), [10.0, 20.0]),
        ], ignore_index=True)
        out = FF.add_lags(f, ["grp"], ["cost"], [1])
        b_first = out[(out["grp"] == "b")].sort_values("run_date").iloc[0]
        # b's first lag must be NaN, not a's last value
        assert pd.isna(b_first["cost_lag1"])

    def test_unsorted_input_is_sorted_internally(self):
        f = _daily("a", date(2025, 1, 1), [1.0, 2.0, 3.0]).iloc[::-1]
        out = FF.add_lags(f, ["grp"], ["cost"], [1])
        row = out[out["run_date"] == date(2025, 1, 2)].iloc[0]
        assert row["cost_lag1"] == 1.0


class TestRolling:
    def test_window_ends_at_t_minus_1(self):
        # THE leakage test: a window-1 roll must equal lag 1, proving day t
        # is outside its own window.
        f = _daily("a", date(2025, 1, 1), [1.0, 2.0, 3.0])
        out = FF.add_rolling(f, ["grp"], ["cost"], [1])
        lag = FF.add_lags(f, ["grp"], ["cost"], [1])
        assert out["cost_roll1"].fillna(-1).tolist() == \
            lag["cost_lag1"].fillna(-1).tolist()

    def test_mean_excludes_today(self):
        f = _daily("a", date(2025, 1, 1), [10.0, 20.0, 999.0])
        out = FF.add_rolling(f, ["grp"], ["cost"], [2])
        # on day 3 the 2-day window is days 1-2, untouched by the 999 spike
        assert out["cost_roll2"].iloc[2] == 15.0

    def test_strict_min_periods_by_default(self):
        f = _daily("a", date(2025, 1, 1), [1.0, 2.0, 3.0, 4.0])
        out = FF.add_rolling(f, ["grp"], ["cost"], [3])
        # first full 3-day trailing window exists only on day 4
        assert out["cost_roll3"].isna().tolist() == [True, True, True, False]
        assert out["cost_roll3"].iloc[3] == 2.0


class TestCalendar:
    def test_known_future_leads(self):
        f = pd.DataFrame({"run_date": [date(2025, 12, 31), date(2025, 12, 3),
                                       date(2025, 11, 30)]})
        out = FF.add_calendar(f)
        row = out[out["run_date"] == date(2025, 12, 31)].iloc[0]
        assert row["d_to_month_end"] == 0
        assert row["d_to_quarter_end"] == 0
        row = out[out["run_date"] == date(2025, 12, 3)].iloc[0]
        assert row["d_to_month_end"] == 28
        assert row["dow"] == 2  # Wednesday, as in the A.1 example rows
        row = out[out["run_date"] == date(2025, 11, 30)].iloc[0]
        assert row["d_to_month_end"] == 0
        assert row["d_to_quarter_end"] == 31
        assert bool(row["is_weekend"])  # a Sunday


class TestPadDaily:
    def _gate(self, days, complete=True, sub="s"):
        return pd.DataFrame({
            "run_date": days,
            "subscription_id": sub,
            "gate_complete": complete,
        })

    def _frame(self):
        # day 2 missing
        return pd.DataFrame({
            "run_date": [date(2025, 1, 1), date(2025, 1, 3)],
            "subscription_id": ["s", "s"],
            "grp": ["a", "a"],
            "cost": [10.0, 30.0],
        })

    def test_gap_filled_with_zero_and_flag(self):
        gate = self._gate([date(2025, 1, 1), date(2025, 1, 2),
                           date(2025, 1, 3)])
        out = FF.pad_daily(self._frame(), ["subscription_id", "grp"],
                           zero_cols=["cost"], gate=gate)
        assert len(out) == 3
        pad = out[out["run_date"] == date(2025, 1, 2)].iloc[0]
        assert pad["cost"] == 0.0
        assert bool(pad["padded"])
        assert not out[out["run_date"] == date(2025, 1, 1)]["padded"].iloc[0]

    def test_unverifiable_padded_day_is_excluded(self):
        # gate has no row for the missing day: it cannot be distinguished
        # from a failed load, so it must not be invented as a zero
        gate = self._gate([date(2025, 1, 1), date(2025, 1, 3)])
        out = FF.pad_daily(self._frame(), ["subscription_id", "grp"],
                           zero_cols=["cost"], gate=gate)
        assert len(out) == 2
        assert date(2025, 1, 2) not in set(out["run_date"])

    def test_gate_failed_padded_day_is_excluded(self):
        gate = pd.DataFrame({
            "run_date": [date(2025, 1, 1), date(2025, 1, 2),
                         date(2025, 1, 3)],
            "subscription_id": "s",
            "gate_complete": [True, False, True],
        })
        out = FF.pad_daily(self._frame(), ["subscription_id", "grp"],
                           zero_cols=["cost"], gate=gate)
        assert date(2025, 1, 2) not in set(out["run_date"])

    def test_observed_rows_never_dropped_by_padding(self):
        # gating OBSERVED rows is apply_gate's job; padding must not
        # anticipate it
        gate = self._gate([date(2025, 1, 1), date(2025, 1, 3)],
                          complete=False)
        out = FF.pad_daily(self._frame(), ["subscription_id", "grp"],
                           zero_cols=["cost"], gate=gate)
        assert len(out) == 2  # both observed rows survive

    def test_no_gate_pads_everything(self):
        out = FF.pad_daily(self._frame(), ["subscription_id", "grp"],
                           zero_cols=["cost"])
        assert len(out) == 3


class TestEffectivePriceDrift:
    def _lines(self, prices, start=date(2025, 1, 1), usage=10.0,
               sub="s", meter="m"):
        days = pd.date_range(start, periods=len(prices), freq="D")
        return pd.DataFrame({
            "run_date": [d.date() for d in days],
            "subscription_id": sub,
            "meter": meter,
            "usage_quantity": usage,
            "pre_tax_cost": [p * usage for p in prices],
        })

    def test_step_registers_as_spike(self):
        prices = [1.0] * 42 + [0.5] * 21
        drift = FF.effective_price_drift(
            self._lines(prices), ["subscription_id"])
        d = drift.set_index("run_date")["price_drift"]
        assert d.loc[date(2025, 1, 30)] == pytest.approx(0.0, abs=1e-9)
        # after the halving, the 14d/28d ratio goes decisively negative
        assert d.iloc[-1] < -0.3

    def test_glide_registers_as_sustained_elevation(self):
        # the December-2024 counterexample to a step flag: no single date,
        # a smooth five-week decline. The drift measure must go and STAY
        # negative through the glide.
        glide = list(np.linspace(1.0, 0.35, 35))
        prices = [1.0] * 42 + glide
        drift = FF.effective_price_drift(
            self._lines(prices), ["subscription_id"])
        d = drift.set_index("run_date")["price_drift"]
        tail = d.iloc[-14:]
        assert (tail < -0.1).all()

    def test_stable_prices_drift_zero(self):
        drift = FF.effective_price_drift(
            self._lines([2.0] * 60), ["subscription_id"])
        assert drift["price_drift"].abs().max() == pytest.approx(0.0,
                                                                 abs=1e-9)

    def test_unpriceable_rows_ignored(self):
        lines = self._lines([1.0] * 60)
        free = lines.copy()
        free["pre_tax_cost"] = 0.0  # free-tier rows: no effective price
        both = pd.concat([lines, free], ignore_index=True)
        drift = FF.effective_price_drift(both, ["subscription_id"])
        assert drift["price_drift"].abs().max() == pytest.approx(0.0,
                                                                 abs=1e-9)

    def test_large_meter_outweighs_trivial_one(self):
        big = self._lines([1.0] * 42 + [0.5] * 21, meter="big", usage=1000.0)
        small = self._lines([1.0] * 63, meter="small", usage=0.1)
        drift = FF.effective_price_drift(
            pd.concat([big, small], ignore_index=True), ["subscription_id"])
        assert drift.set_index("run_date")["price_drift"].iloc[-1] < -0.3


class TestFeatureColumns:
    def test_unlagged_cost_cannot_be_a_feature(self):
        f = pd.DataFrame(columns=["run_date", "cost", "cost_lag1", "dow"])
        with pytest.raises(A.DataQualityError, match="side door"):
            FF.feature_columns(f, exclude=["run_date"])

    def test_lagged_cost_passes(self):
        f = pd.DataFrame(columns=["run_date", "cost", "cost_lag1", "dow"])
        cols = FF.feature_columns(f, exclude=["run_date", "cost"])
        assert set(cols) == {"cost_lag1", "dow"}
