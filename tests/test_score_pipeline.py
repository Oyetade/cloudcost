"""Tests for score_pipeline.py, end to end: a synthetic snapshot on disk is
read by transform.load_snapshot, built by the REAL build_pool_frame,
featurized, scored by a reloaded bundle, and written to the ledger. The
same path production takes, minus Postgres."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from catpipe import transform as T
from catpipe.persistence import save_bundle
from catpipe.score_pipeline import ScoringAborted, score_snapshot
from tests import helpers_ops as H


@pytest.fixture(scope="module")
def tables():
    return H.make_tables()


@pytest.fixture(scope="module")
def featured(tables):
    return H.featurize_pool(T.build_pool_frame(tables))


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory, featured):
    boosters = H.fit_boosters(featured)
    card = H.make_card(featured)
    root = tmp_path_factory.mktemp("models")
    return save_bundle(root / "v2026-08-01", boosters, card)


@pytest.fixture()
def snapshot_dir(tmp_path, tables):
    d = tmp_path / "snapshot"
    H.write_snapshot(tables, d)
    return d


class TestScoreSnapshot:
    def test_happy_path_scores_ledgers_and_manifests(
        self, tmp_path, snapshot_dir, bundle_dir, featured
    ):
        m = score_snapshot(snapshot_dir, bundle_dir, tmp_path / "ledger",
                           "pool", featurize=H.featurize_pool)
        assert m["aborted"] is False
        assert m["rows_scored"] == len(featured)
        assert m["point_col"] == "pred_mean"
        assert Path(m["ledger_file"]).exists()
        ledger_df = pd.read_parquet(m["ledger_file"])
        for col in ("run_date", "pool_name", "q05", "q50", "q95",
                    "pred_mean", "model_version", "scored_at"):
            assert col in ledger_df.columns

    def test_watermark_makes_rerun_a_soft_noop(
        self, tmp_path, snapshot_dir, bundle_dir
    ):
        score_snapshot(snapshot_dir, bundle_dir, tmp_path / "ledger",
                       "pool", featurize=H.featurize_pool)
        second = score_snapshot(snapshot_dir, bundle_dir, tmp_path / "ledger",
                                "pool", featurize=H.featurize_pool)
        assert second["aborted"] is True
        assert "watermark" in second["abort_reason"]

    def test_new_days_beyond_watermark_are_scored(
        self, tmp_path, snapshot_dir, bundle_dir, tables
    ):
        first = score_snapshot(snapshot_dir, bundle_dir, tmp_path / "ledger",
                               "pool", featurize=H.featurize_pool)
        extended = H.make_tables(n_days=127)  # one more week
        d2 = tmp_path / "snapshot2"
        H.write_snapshot(extended, d2)
        second = score_snapshot(d2, bundle_dir, tmp_path / "ledger",
                                "pool", featurize=H.featurize_pool)
        assert second["aborted"] is False
        assert second["rows_scored"] == 7 * 3  # 7 new days x 3 pools
        assert second["max_run_date"] > first["max_run_date"]

    def test_data_quality_failure_aborts_and_manifests(
        self, tmp_path, tables, bundle_dir
    ):
        """A duplicated batch row trips assert_no_duplicates inside the real
        build_pool_frame; scoring must abort loudly, with the manifest
        surviving for the audit trail."""
        broken = {k: v.copy() for k, v in tables.items()}
        broken["raw_cost"] = pd.concat(
            [broken["raw_cost"], broken["raw_cost"].head(1)],
            ignore_index=True,
        )
        d = tmp_path / "snapshot_broken"
        H.write_snapshot(broken, d)
        with pytest.raises(ScoringAborted, match="data-quality"):
            score_snapshot(d, bundle_dir, tmp_path / "ledger", "pool",
                           featurize=H.featurize_pool)
        manifests = list(
            (tmp_path / "ledger" / "frame=pool" / "manifests").glob("*.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text())["aborted"] is True

    def test_short_snapshot_is_refused(self, tmp_path, bundle_dir):
        short = H.make_tables(n_days=20)  # below the 29 the card's rolls need
        d = tmp_path / "snapshot_short"
        H.write_snapshot(short, d)
        with pytest.raises(ScoringAborted, match="edge-degraded"):
            score_snapshot(d, bundle_dir, tmp_path / "ledger", "pool",
                           featurize=H.featurize_pool)

    def test_wrong_frame_bundle_is_refused(
        self, tmp_path, snapshot_dir, bundle_dir
    ):
        with pytest.raises(ScoringAborted, match="frame"):
            score_snapshot(snapshot_dir, bundle_dir, tmp_path / "ledger",
                           "team", featurize=H.featurize_pool)

    def test_incomplete_days_are_not_scored(self, tmp_path, bundle_dir):
        """C8 in code: a day whose three runs are not all Complete is not
        scoreable. Break the last day's Attribution run and the scored
        row count drops by exactly one day's pools."""
        tables = H.make_tables()
        rs = tables["run_status"]
        last_day = rs["run_date"].max()
        mask = (rs["run_date"] == last_day) & (rs["run_type"] == "Attribution")
        rs.loc[mask, "status"] = "Failed"
        d = tmp_path / "snapshot_incomplete"
        H.write_snapshot(tables, d)
        m = score_snapshot(d, bundle_dir, tmp_path / "ledger", "pool",
                           featurize=H.featurize_pool)
        full = H.featurize_pool(T.build_pool_frame(H.make_tables()))
        assert m["rows_scored"] == len(full) - 3  # 3 pools on the broken day
