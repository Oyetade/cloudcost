"""Tests for min_history.py and ledger.py, plus the ragged-window lag
hazard pinned as a demonstration to port against build_pool_frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catpipe.ledger import LedgerError, PredictionLedger
from catpipe.min_history import (
    feature_history_days, min_history_days, recommended_extract_days,
)


class TestMinHistory:
    def test_feature_history_parsing(self):
        assert feature_history_days("cost_lag1") == 1
        assert feature_history_days("cost_lag_7") == 7
        assert feature_history_days("cost_roll28") == 29      # shifted-then-rolled
        assert feature_history_days("job_seconds_rolling_28") == 29
        assert feature_history_days("price_drift_lag1") == 42  # 14d vs prior 28d
        assert feature_history_days("effective_price_drift") == 42
        assert feature_history_days("price_drift_lag7") == 48
        assert feature_history_days("dow") == 0
        assert feature_history_days("pool_name") == 0

    def test_current_frame_features_need_42_days_at_h1(self):
        features = ["cost_lag1", "cost_lag7", "cost_roll28",
                    "price_drift_lag1", "dow", "d_to_month_end", "pool_name"]
        assert min_history_days(features, horizon_days=1) == 42
        assert min_history_days(features, horizon_days=30) == 71

    def test_deeper_roll_raises_the_demand(self):
        """The reason this is computed, not configured: a 90-day roll added
        next quarter must raise the pipeline's history demand with it."""
        assert min_history_days(["cost_lag1", "cost_roll90"]) == 91

    def test_recommended_pull_covers_detector_and_load_lag(self):
        features = ["cost_lag1", "cost_roll28", "price_drift_lag1"]
        pull = recommended_extract_days(features)
        assert pull >= 42 + 10   # feature floor + load lag
        assert pull >= 90 + 10   # detector trailing windows + load lag


def _preds(dates, pools, value=100.0):
    rows = [{"run_date": pd.Timestamp(d),
             "subscription_id": "sub-001",
             "batch_account_name": "batchacct1",
             "pool_name": p,
             "q05": value * 0.8, "q50": value, "q95": value * 1.2,
             "pred_mean": value * 1.05}
            for d in dates for p in pools]
    return pd.DataFrame(rows)


GROUPS = ["subscription_id", "batch_account_name", "pool_name"]


class TestLedger:
    def test_append_and_current_view(self, tmp_path):
        led = PredictionLedger(tmp_path, "pool", GROUPS)
        led.append(_preds(["2026-07-01", "2026-07-02"], ["pool_0"]), "v1")
        view = led.current_view()
        assert len(view) == 2
        assert set(view["model_version"]) == {"v1"}

    def test_rescore_resolves_to_latest_and_rewrites_nothing(self, tmp_path):
        led = PredictionLedger(tmp_path, "pool", GROUPS)
        led.append(_preds(["2026-07-01"], ["pool_0"], value=100.0), "v1")
        led.append(_preds(["2026-07-01"], ["pool_0"], value=150.0), "v1")
        view = led.current_view()
        assert len(view) == 1
        assert view["q50"].iloc[0] == 150.0
        assert len(led.read_all()) == 2  # full history still on disk

    def test_mixed_versions_refuse_to_blend(self, tmp_path):
        """A residual series must never silently span a retrain boundary."""
        led = PredictionLedger(tmp_path, "pool", GROUPS)
        led.append(_preds(["2026-07-01"], ["pool_0"]), "v1")
        led.append(_preds(["2026-07-02"], ["pool_0"]), "v2")
        with pytest.raises(LedgerError, match="multiple model versions"):
            led.current_view()
        assert len(led.current_view("v2")) == 1

    def test_watermark_advances_and_never_reverses(self, tmp_path):
        led = PredictionLedger(tmp_path, "pool", GROUPS)
        assert led.watermark("v1") is None
        led.advance_watermark("v1", "2026-07-10")
        assert led.watermark("v1") == pd.Timestamp("2026-07-10")
        with pytest.raises(LedgerError, match="backwards"):
            led.advance_watermark("v1", "2026-07-05")
        assert led.watermark("v2") is None  # per version

    def test_duplicate_grain_within_one_append_raises(self, tmp_path):
        led = PredictionLedger(tmp_path, "pool", GROUPS)
        df = pd.concat([_preds(["2026-07-01"], ["pool_0"])] * 2,
                       ignore_index=True)
        with pytest.raises(LedgerError, match="grain"):
            led.append(df, "v1")


class TestRaggedWindowLagHazard:
    def test_gate_before_lag_reads_two_days_back(self):
        """With load lag, day T-4 can be incomplete while T-3 and T-5 are
        complete. If the gate EXCLUDES the incomplete day from the frame
        before lags are computed, shift(1) silently reindexes T-3's
        cost_lag1 to T-5's value: a wrong feature served with no error.

        Pinned here as a demonstration; port against build_pool_frame with
        a hole punched in run_status, asserting the frame either excludes
        the affected day or carries NaN there, never the value from two
        days back. The right order: lag on the full calendar spine first,
        then gate; the wrong order produces a value, the right order
        produces NaN, which _check_novel_nulls converts into a refusal.
        """
        days = pd.date_range("2026-07-01", periods=6, freq="D")
        df = pd.DataFrame({"run_date": days,
                           "cost": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                           "complete": [True, True, True, False, True, True]})

        # WRONG: gate first, then lag. Day index 4 inherits index 2's cost.
        gated_first = df[df["complete"]].copy()
        gated_first["cost_lag1"] = gated_first["cost"].shift(1)
        wrong = gated_first.loc[gated_first["run_date"] == days[4],
                                "cost_lag1"].iloc[0]
        assert wrong == 30.0  # silently two days back: the hazard

        # RIGHT: lag on the full spine, then gate.
        spined = df.copy()
        spined.loc[~spined["complete"], "cost"] = np.nan
        spined["cost_lag1"] = spined["cost"].shift(1)
        gated_after = spined[spined["complete"]]
        right = gated_after.loc[gated_after["run_date"] == days[4],
                                "cost_lag1"].iloc[0]
        assert np.isnan(right)
