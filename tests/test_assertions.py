"""Tests for assertions.py: the section-7 invariants."""

from datetime import date
import pandas as pd
import pytest

from catpipe import assertions as A


class TestDuplicates:
    def test_passes_on_unique(self):
        df = pd.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
        A.assert_no_duplicates(df, ["k"], "t")  # no raise

    def test_raises_on_duplicate(self):
        df = pd.DataFrame({"k": [1, 1, 2], "v": [10, 10, 20]})
        with pytest.raises(A.DataQualityError, match="duplicate"):
            A.assert_no_duplicates(df, ["k"], "raw_cost")


class TestRowCountIdentity:
    def test_passes_when_equal(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        A.assert_row_count_identity(df, 3, "join")

    def test_raises_when_grown(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        with pytest.raises(A.DataQualityError, match="not unique"):
            A.assert_row_count_identity(df, 3, "job_usage x job_cost")


class TestAntiJoin:
    def test_reports_orphans_both_ways(self):
        left = pd.DataFrame({"k": [1, 2, 3]})
        right = pd.DataFrame({"k": [2, 3, 4]})
        rep = A.report_anti_join(left, right, ["k"], "L", "R")
        assert rep["L_only"] == 1  # key 1
        assert rep["R_only"] == 1  # key 4
        assert rep["both"] == 2    # keys 2, 3

    def test_no_orphans(self):
        left = pd.DataFrame({"k": [1, 2]})
        right = pd.DataFrame({"k": [1, 2]})
        rep = A.report_anti_join(left, right, ["k"], "L", "R")
        assert rep["L_only"] == 0 and rep["R_only"] == 0


class TestNoFailedGate:
    def test_passes_when_only_complete_and_ungated(self):
        df = pd.DataFrame({"gate_state": ["gated_complete", "ungated"]})
        A.assert_no_failed_gate(df, "ctx")

    def test_raises_when_failed_survives(self):
        df = pd.DataFrame({"gate_state": ["gated_complete", "gated_failed"]})
        with pytest.raises(A.DataQualityError, match="gated_failed"):
            A.assert_no_failed_gate(df, "ctx")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(A.DataQualityError, match="missing"):
            A.assert_no_failed_gate(df, "ctx")


class TestNoSameDayCost:
    def test_lagged_cost_is_fine(self):
        A.assert_no_same_day_cost(["cost_lag_1", "job_seconds"], "ctx")

    def test_raw_cost_column_is_leak(self):
        with pytest.raises(A.DataQualityError, match="side door"):
            A.assert_no_same_day_cost(["cost", "job_seconds"], "ctx")

    def test_pre_tax_cost_is_leak(self):
        with pytest.raises(A.DataQualityError, match="side door"):
            A.assert_no_same_day_cost(["pre_tax_cost"], "ctx")


# --- Q22: the grain of the one-write check (14 July 2026) -------------------
#
# The check keyed on (run_date, subscription_id), which is not a row. A loader
# writing a subscription's many cost lines at different moments produces many
# update_times per bucket with nothing rewritten. The old check called that a
# violation; these tests pin the distinction.

def _row(update_time, cost=100.0, sub_cat="Files", meter="Read Operations",
         batch="ba-1", pool="pool-1", rg="rg-a"):
    return dict(
        run_date=date(2026, 1, 1), subscription_id="sub-1",
        resource_group_name=rg, resource_type="microsoft.storage/storageaccounts",
        meter=meter, meter_sub_category=sub_cat,
        batch_account_name=batch, pool_name=pool,
        pre_tax_cost=cost, update_time=pd.Timestamp(update_time),
    )


def test_one_write_allows_many_rows_written_at_different_times():
    """THE Q22 false-positive case: one subscription-day, several distinct
    rows, each written once, at different moments. The old coarse key called
    this a rewrite. It is a normal loader.
    """
    raw = pd.DataFrame([
        _row("2026-01-02T01:00:00", sub_cat="Files"),
        _row("2026-01-02T02:00:00", sub_cat="Tables"),
        _row("2026-01-02T03:00:00", rg="rg-b"),
        _row("2026-01-02T04:00:00", meter="Write Operations"),
    ])
    A.assert_one_write_per_slice(raw)  # must not raise


def test_one_write_still_catches_a_genuine_rewrite():
    """The same ROW written twice: zero first, real value later. This is the
    live pattern found on the dev spot VMs.
    """
    raw = pd.DataFrame([
        _row("2026-01-02T01:00:00", cost=0.0),
        _row("2026-03-15T09:00:00", cost=366.09),
    ])
    with pytest.raises(A.DataQualityError, match="more than one update_time"):
        A.assert_one_write_per_slice(raw)


def test_one_write_does_not_go_blind_on_null_keys():
    """dropna=False is load-bearing: batch_account_name and pool_name are
    nullable, and groupby's default would silently drop these rows, letting a
    real rewrite pass unseen. That default is what cost 17,585.93 in Q23.
    """
    raw = pd.DataFrame([
        _row("2026-01-02T01:00:00", cost=0.0, batch=None, pool=None),
        _row("2026-03-15T09:00:00", cost=366.09, batch=None, pool=None),
    ])
    with pytest.raises(A.DataQualityError):
        A.assert_one_write_per_slice(raw)


def test_one_write_coarse_key_would_have_false_positived():
    """Pins the diagnosis itself: the benign frame passes at row grain and
    fails at the old (run_date, subscription_id) grain. If this ever stops
    being true, the Q22 reasoning needs revisiting.
    """
    raw = pd.DataFrame([
        _row("2026-01-02T01:00:00", sub_cat="Files"),
        _row("2026-01-02T02:00:00", sub_cat="Tables"),
    ])
    A.assert_one_write_per_slice(raw)
    with pytest.raises(A.DataQualityError):
        A.assert_one_write_per_slice(
            raw, keys=["run_date", "subscription_id"])


def test_raw_cost_grain_includes_meter_sub_category():
    """Q24: the omission that flagged 142 legitimate rows as duplicates."""
    assert "meter_sub_category" in A.RAW_COST_GRAIN


def test_no_duplicates_accepts_files_and_tables_sub_categories():
    """The exact live rows from 2025-10-30: same meter, different sub-category,
    both zero-cost. Two rows, not one row twice.
    """
    raw = pd.DataFrame([
        _row("2025-11-01T07:04:59", cost=0.0, sub_cat="Files",
             batch=None, pool=None),
        _row("2025-11-01T07:04:59", cost=0.0, sub_cat="Tables",
             batch=None, pool=None),
    ])
    A.assert_no_duplicates(raw, A.RAW_COST_GRAIN, "raw_cost[batch]")
