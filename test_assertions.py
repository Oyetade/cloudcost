"""Tests for assertions.py: the section-7 invariants."""

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


class TestGateComplete:
    def test_passes_all_true(self):
        df = pd.DataFrame({"gate_complete": [True, True]})
        A.assert_gate_complete(df, "ctx")

    def test_raises_on_false_slipping_through(self):
        df = pd.DataFrame({"gate_complete": [True, False]})
        with pytest.raises(A.DataQualityError, match="passed the gate"):
            A.assert_gate_complete(df, "ctx")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(A.DataQualityError, match="missing"):
            A.assert_gate_complete(df, "ctx")


class TestNoSameDayCost:
    def test_lagged_cost_is_fine(self):
        A.assert_no_same_day_cost(["cost_lag_1", "job_seconds"], "ctx")

    def test_raw_cost_column_is_leak(self):
        with pytest.raises(A.DataQualityError, match="side door"):
            A.assert_no_same_day_cost(["cost", "job_seconds"], "ctx")

    def test_pre_tax_cost_is_leak(self):
        with pytest.raises(A.DataQualityError, match="side door"):
            A.assert_no_same_day_cost(["pre_tax_cost"], "ctx")
