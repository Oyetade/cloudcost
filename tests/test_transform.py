"""Tests for transform.py: gate, five-key join, mask, team frame."""

from datetime import date

import pandas as pd
import pytest

from catpipe import assertions as A
from catpipe import transform as T


def _rs(run_date, sub, rtype, status, update):
    return {
        "run_date": run_date, "subscription_id": sub, "run_type": rtype,
        "status": status, "update_time": pd.Timestamp(update, tz="UTC"),
    }


class TestStampRegime:
    def _frame(self, dates):
        return pd.DataFrame({"run_date": dates, "cost": [1.0] * len(dates)})

    def test_all_four_regimes_assigned_by_boundary(self):
        f = self._frame([
            date(2022, 1, 1),   # pre_coverage  (< 2022-08-01)
            date(2022, 8, 1),   # cost_only     (>= floor, < activity)
            date(2023, 8, 1),   # featured_ungated (>= activity, < run_status)
            date(2024, 1, 2),   # featured_gated (>= run_status)
        ])
        out = T.stamp_regime(f)
        assert list(out["data_regime"]) == [
            "pre_coverage", "cost_only", "featured_ungated", "featured_gated"
        ]

    def test_boundaries_are_half_open_on_the_left(self):
        # the day BEFORE a boundary belongs to the earlier regime
        f = self._frame([date(2022, 7, 31), date(2023, 7, 31),
                         date(2024, 1, 1)])
        out = T.stamp_regime(f)
        assert list(out["data_regime"]) == [
            "pre_coverage", "cost_only", "featured_ungated"
        ]

    def test_drop_pre_coverage_removes_only_ramp(self):
        f = T.stamp_regime(self._frame([
            date(2022, 1, 1), date(2022, 8, 1), date(2024, 6, 1)
        ]))
        kept = T.drop_pre_coverage(f)
        assert len(kept) == 2
        assert "pre_coverage" not in set(kept["data_regime"])


class TestGate:
    def test_all_three_complete_passes_gate(self):
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Attribution", "Complete", "2024-03-02"),
        ])
        gate = T.build_gate(rs)
        assert gate["gate_complete"].iloc[0]

    def test_one_missing_type_fails_gate(self):
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            # Attribution absent
        ])
        gate = T.build_gate(rs)
        assert not gate["gate_complete"].iloc[0]

    def test_one_errored_type_fails_gate(self):
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Attribution", "Error", "2024-03-02"),
        ])
        gate = T.build_gate(rs)
        assert not gate["gate_complete"].iloc[0]

    def test_unknown_status_fails_safe(self):
        # a status never seen before must NOT pass (fail-safe design)
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Attribution", "Weird", "2024-03-02"),
        ])
        gate = T.build_gate(rs)
        assert not gate["gate_complete"].iloc[0]

    def test_latest_run_wins_per_type(self):
        # earlier Error superseded by later Complete => gate passes
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Error", "2024-03-02 08:00"),
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02 10:00"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Attribution", "Complete", "2024-03-02"),
        ])
        gate = T.build_gate(rs)
        assert gate["gate_complete"].iloc[0]


class TestApplyGate:
    def test_missing_row_inside_era_fails_and_is_excluded(self):
        # A slice on/after run_status_start with no gate row is unverified
        # => excluded.
        frame = pd.DataFrame({
            "run_date": [date(2024, 3, 1), date(2024, 3, 2)],
            "subscription_id": ["s", "s"],
            "cost": [100.0, 200.0],
        })
        gate = pd.DataFrame({
            "run_date": [date(2024, 3, 1)],
            "subscription_id": ["s"],
            "gate_complete": [True],
        })
        kept = T.apply_gate(frame, gate, "ctx",
                            run_status_start=date(2024, 1, 2))
        assert len(kept) == 1
        assert kept["gate_state"].iloc[0] == "gated_complete"

    def test_missing_row_before_era_is_kept_as_ungated(self):
        # A slice BEFORE run_status_start with no gate row is not a failure:
        # run_status did not exist yet => kept, labelled ungated.
        frame = pd.DataFrame({
            "run_date": [date(2023, 9, 1)],
            "subscription_id": ["s"],
            "cost": [100.0],
        })
        gate = pd.DataFrame({
            "run_date": pd.Series([], dtype="object"),
            "subscription_id": pd.Series([], dtype="object"),
            "gate_complete": pd.Series([], dtype="bool"),
        })
        kept = T.apply_gate(frame, gate, "ctx",
                            run_status_start=date(2024, 1, 2))
        assert len(kept) == 1
        assert kept["gate_state"].iloc[0] == "ungated"

    def test_failed_row_before_era_still_excluded_if_gate_says_incomplete(self):
        # If a gate row exists before the era and says incomplete, honour it:
        # ungated only applies where there is NO row at all.
        frame = pd.DataFrame({
            "run_date": [date(2023, 9, 1)],
            "subscription_id": ["s"],
            "cost": [100.0],
        })
        gate = pd.DataFrame({
            "run_date": [date(2023, 9, 1)],
            "subscription_id": ["s"],
            "gate_complete": [False],
        })
        kept = T.apply_gate(frame, gate, "ctx",
                            run_status_start=date(2024, 1, 2))
        assert len(kept) == 0  # explicit incomplete => failed, excluded


class TestFiveKeyJoin:
    def _usage(self, jids):
        return pd.DataFrame([{
            "run_date": date(2024, 3, 1), "subscription_id": "s",
            "batch_account_name": "a", "pool_name": "p", "job_id": j,
            "job_seconds": 100.0, "task_count": 5,
        } for j in jids])

    def _cost(self, jids, team="Pillar1"):
        return pd.DataFrame([{
            "run_date": date(2024, 3, 1), "subscription_id": "s",
            "batch_account_name": "a", "pool_name": "p", "job_id": j,
            "job_name": j.split("-")[0], "job_category": "BT",
            "job_ownership": "Risk", "job_team": team, "cost": 50.0,
        } for j in jids])

    def test_clean_join_preserves_row_count(self):
        usage = self._usage(["j1", "j2"])
        cost = self._cost(["j1", "j2"])
        joined, orphans = T.join_job_attributes(usage, cost)
        assert len(joined) == 2
        assert orphans["both"] == 2

    def test_retried_job_on_usage_side_is_caught(self):
        # A retried job_id within a day on the usage side would inflate every
        # activity aggregate. Caught by the explicit left-side uniqueness
        # check, not by the row-count identity (which compares to len(usage)).
        usage = self._usage(["j1", "j1"])  # retry
        with pytest.raises(A.DataQualityError, match="duplicate"):
            T.join_job_attributes(usage, self._cost(["j1"]))

    def test_duplicate_on_cost_side_does_not_multiply_usage(self):
        # A duplicate key on the cost side is de-duplicated before the join,
        # so usage rows are not multiplied and the identity holds.
        usage = self._usage(["j1"])
        cost = self._cost(["j1", "j1"])  # duplicate attribute rows
        joined, _ = T.join_job_attributes(usage, cost)
        assert len(joined) == 1

    def test_usage_orphan_gets_unknown_not_dropped(self):
        usage = self._usage(["j1", "j2"])
        cost = self._cost(["j1"])  # j2 has no cost row
        joined, orphans = T.join_job_attributes(usage, cost)
        assert len(joined) == 2  # j2 kept, not discarded
        j2 = joined[joined["job_id"] == "j2"].iloc[0]
        assert j2["job_team"] == "Unknown"
        assert orphans["job_usage_only"] == 1

    def test_never_brings_same_day_cost(self):
        usage = self._usage(["j1"])
        cost = self._cost(["j1"])
        joined, _ = T.join_job_attributes(usage, cost)
        assert "cost" not in joined.columns


class TestPriceableMask:
    def test_masks_zero_cost_and_zero_usage(self):
        rc = pd.DataFrame({
            "pre_tax_cost": [10.0, 0.0, 5.0, 0.0],
            "usage_quantity": [2.0, 3.0, 0.0, 0.0],
        })
        mask = T.priceable_mask(rc)
        assert mask.tolist() == [True, False, False, False]


class TestTeamFrame:
    def test_null_team_distinct_from_unknown(self):
        jc = pd.DataFrame([
            {"run_date": date(2024, 3, 1), "subscription_id": "s",
             "job_team": "Unknown", "cost": 10.0},
            {"run_date": date(2024, 3, 1), "subscription_id": "s",
             "job_team": None, "cost": 20.0},
        ])
        rs = pd.DataFrame([
            _rs(date(2024, 3, 1), "s", "Cost", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Usage", "Complete", "2024-03-02"),
            _rs(date(2024, 3, 1), "s", "Attribution", "Complete", "2024-03-02"),
        ])
        frame = T.build_team_frame({"job_cost": jc, "run_status": rs})
        teams = set(frame["job_team"])
        assert "Unknown" in teams
        assert "__NULL_TEAM__" in teams
        assert len(frame) == 2  # kept distinct, not merged
