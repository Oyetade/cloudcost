"""Tests for the pivot-column reconciliation and the card-level leakage
assertion, both driven by the real frame_1a feature list: 19 of its 40
features are share_*_lag1 pivot columns that only exist when their category
was observed in the window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catpipe import transform as T
from catpipe.persistence import (
    PersistenceError, assert_no_unlagged_features, load_bundle,
    reindex_pivot_features, save_bundle,
)
from tests import helpers_ops as H


@pytest.fixture(scope="module")
def featured():
    return H.featurize_pool(T.build_pool_frame(H.make_tables()))


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, featured):
    boosters = H.fit_boosters(featured)
    card = H.make_card(featured)
    root = tmp_path_factory.mktemp("models")
    return load_bundle(save_bundle(root / "v1", boosters, card))


class TestPivotReindex:
    def test_quiet_window_zero_fills_and_reports(self, featured, loaded):
        """A window in which no Audit jobs ran has no share_Audit_lag1
        column at all. The card knows its true value is 0.0; the run must
        score, not refuse, and must say what it filled."""
        quiet = featured.drop(columns=["share_Audit_lag1"])
        preds = loaded.predict(quiet)
        assert preds.attrs["pivot_report"]["zero_filled"] == ["share_Audit_lag1"]
        assert len(preds) == len(quiet)

    def test_zero_fill_equals_genuinely_zero_share(self, featured, loaded):
        """Filling an absent column with 0.0 must give the same predictions
        as the column being present and zero: absence of the category and a
        zero share are the same fact."""
        quiet = featured.drop(columns=["share_Audit_lag1"])
        explicit = featured.copy()
        explicit["share_Audit_lag1"] = 0.0
        np.testing.assert_array_equal(
            loaded.predict(quiet)["q50"].to_numpy(),
            loaded.predict(explicit)["q50"].to_numpy(),
        )

    def test_novel_category_is_reported_not_absorbed(self, featured, loaded):
        """A new job category since training is the column-space mirror of
        an unseen pool level: not fed to the model, but surfaced."""
        grown = featured.copy()
        grown["share_NewCategory_lag1"] = 0.5
        preds = loaded.predict(grown)
        assert preds.attrs["pivot_report"]["novel_pivot_columns"] == [
            "share_NewCategory_lag1"
        ]

    def test_non_pivot_missing_feature_still_refuses(self, featured, loaded):
        """The zero-fill is scoped to the declared prefixes; any other
        missing feature is still a schema break."""
        with pytest.raises(PersistenceError, match="missing features"):
            loaded.predict(featured.drop(columns=["cost_lag7"]))

    def test_unlagged_pivot_column_is_not_flagged_novel(self, featured, loaded):
        """The frame's unlagged share_Audit raw material must not be
        reported as a novel pivot column; only lagged/rolled forms count."""
        preds = loaded.predict(featured)
        assert preds.attrs["pivot_report"]["novel_pivot_columns"] == []


class TestUnlaggedFeatureAssertion:
    COLUMNS = ["run_date", "cost", "job_seconds", "job_seconds_lag1",
               "share_Audit", "share_Audit_lag1", "price_drift",
               "price_drift_lag1", "cost_lag1", "dow", "pool_name"]

    def test_real_frame_1a_feature_list_passes(self):
        """The list the model actually sees: every cost/activity feature
        lagged or rolled, calendar and statics untouched."""
        features = ["cost_lag1", "cost_lag7", "cost_roll28",
                    "job_seconds_lag1", "share_Audit_lag1",
                    "price_drift_lag1", "job_seconds_roll7",
                    "dow", "is_weekend", "environment_tier", "pool_name"]
        cols = self.COLUMNS + ["job_seconds_roll7", "is_weekend",
                               "environment_tier"]
        assert_no_unlagged_features(features, cols, target="cost")  # no raise

    def test_target_is_caught(self):
        with pytest.raises(PersistenceError, match="unlagged same-day"):
            assert_no_unlagged_features(["cost", "dow"], self.COLUMNS, "cost")

    def test_same_day_activity_twin_is_caught(self):
        """assert_no_same_day_cost only catches literal cost columns; the
        charter's rule covers activity too. job_seconds has a lagged twin
        in the frame, so its unlagged form in the card is a leak."""
        with pytest.raises(PersistenceError, match="job_seconds"):
            assert_no_unlagged_features(
                ["job_seconds", "dow"], self.COLUMNS, "cost")

    def test_same_day_pivot_share_is_caught(self):
        with pytest.raises(PersistenceError, match="share_Audit"):
            assert_no_unlagged_features(
                ["share_Audit"], self.COLUMNS, "cost")

    def test_same_day_price_drift_is_caught(self):
        with pytest.raises(PersistenceError, match="price_drift"):
            assert_no_unlagged_features(
                ["price_drift"], self.COLUMNS, "cost")

    def test_calendar_and_statics_pass(self):
        assert_no_unlagged_features(
            ["dow", "pool_name"], self.COLUMNS, "cost")  # no raise
