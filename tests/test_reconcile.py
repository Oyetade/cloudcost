"""Tests for reconcile.py: pool-day job_cost vs raw_cost divergence."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from catpipe import reconcile as R


def _raw(rows):
    """rows: list of (date, sub, acct, pool_or_None, cost)."""
    return pd.DataFrame([
        dict(run_date=d, subscription_id=s, batch_account_name=a,
             pool_name=p, resource_type="vmss", meter="D64",
             pre_tax_cost=c, usage_quantity=1.0)
        for (d, s, a, p, c) in rows
    ])


def _job(rows):
    """rows: list of (date, sub, acct, pool, cost)."""
    return pd.DataFrame([
        dict(run_date=d, subscription_id=s, batch_account_name=a,
             pool_name=p, job_id=f"j{i}", cost=c)
        for i, (d, s, a, p, c) in enumerate(rows)
    ])


class TestScope:
    def test_non_pool_raw_cost_excluded_from_comparison(self):
        # a non-batch ledger line (pool_name None) must not inflate raw_cost,
        # else raw is trivially larger than job everywhere.
        raw = _raw([
            (date(2024, 3, 1), "s", "a", "p", 100.0),
            (date(2024, 3, 1), "s", "a", None, 900.0),  # non-pool, excluded
        ])
        job = _job([(date(2024, 3, 1), "s", "a", "p", 100.0)])
        recon = R.reconcile_pool_day(raw, job)
        row = recon.iloc[0]
        assert row["raw_cost"] == 100.0   # not 1000.0
        assert row["coverage"] == "both"


class TestAgreement:
    def test_clean_agreement_within_tolerance(self):
        raw = _raw([
            (date(2024, 3, 1), "s", "a", "p", 100.0),
            (date(2024, 3, 2), "s", "a", "p", 200.0),
        ])
        job = _job([
            (date(2024, 3, 1), "s", "a", "p", 102.0),   # +2%
            (date(2024, 3, 2), "s", "a", "p", 197.0),   # -1.5%
        ])
        recon = R.reconcile_pool_day(raw, job, rel_tol=0.05)
        assert recon["within_tol"].all()
        summary = R.reconciliation_summary(recon)
        assert 0.97 <= summary["aggregate_job_over_raw"] <= 1.03
        # agree on overlap AND full coverage => interchangeable
        assert "interchangeable" in summary["verdict"]

    def test_rel_diff_computed_correctly(self):
        raw = _raw([(date(2024, 3, 1), "s", "a", "p", 100.0)])
        job = _job([(date(2024, 3, 1), "s", "a", "p", 110.0)])
        recon = R.reconcile_pool_day(raw, job)
        assert recon["rel_diff"].iloc[0] == pytest.approx(0.10)
        assert recon["abs_diff"].iloc[0] == pytest.approx(10.0)


class TestSubset:
    def test_exact_on_overlap_but_job_omits_pooldays(self):
        # The confirmed real-data shape: job_cost equals raw_cost EXACTLY on
        # shared pool-days, but raw_cost has extra pool-days job_cost lacks.
        # => job_cost is an exact SUBSET, not a distortion. raw_cost wins 1a.
        raw = _raw([
            (date(2024, 3, 1), "s", "a", "p", 100.0),
            (date(2024, 3, 2), "s", "a", "p", 100.0),
            (date(2024, 3, 3), "s", "a", "p", 100.0),
        ])
        job = _job([(date(2024, 3, 1), "s", "a", "p", 100.0)])  # exact, day 1
        recon = R.reconcile_pool_day(raw, job)
        summary = R.reconciliation_summary(recon)
        assert summary["coverage_counts"].get("raw_only", 0) == 2
        assert summary["pct_raw_cost_unattributed"] == pytest.approx(2 / 3)
        # exact on the shared day (rel_diff 0) but a subset in coverage
        assert summary["median_abs_rel_diff_both"] == pytest.approx(0.0)
        assert "EXACT SUBSET" in summary["verdict"]
        assert "raw_cost wins for the physical forecast" in summary["verdict"]

    def test_job_only_pool_day_is_flagged(self):
        # attribution with no matching ledger line: a data-quality signal.
        raw = _raw([(date(2024, 3, 1), "s", "a", "p", 100.0)])
        job = _job([
            (date(2024, 3, 1), "s", "a", "p", 100.0),
            (date(2024, 3, 2), "s", "a", "p", 50.0),   # no raw_cost row
        ])
        recon = R.reconcile_pool_day(raw, job)
        assert (recon["coverage"] == "job_only").sum() == 1
        jo = recon[recon["coverage"] == "job_only"].iloc[0]
        assert jo["raw_cost"] == 0.0
        assert np.isnan(jo["rel_diff"])   # undefined against zero billed


class TestBorderline:
    def test_close_but_loose_gives_caution(self):
        raw = _raw([
            (date(2024, 3, 1), "s", "a", "p", 100.0),
            (date(2024, 3, 2), "s", "a", "p", 100.0),
        ])
        job = _job([
            (date(2024, 3, 1), "s", "a", "p", 108.0),   # +8%, outside 5% tol
            (date(2024, 3, 2), "s", "a", "p", 93.0),    # -7%
        ])
        recon = R.reconcile_pool_day(raw, job, rel_tol=0.05)
        summary = R.reconciliation_summary(recon)
        # aggregate ratio ~1.005, but per-day dispersion breaches tol, and
        # coverage is full => "disagree on cost" branch
        assert summary["share_within_tol_of_both"] < 0.9
        assert "disagree on cost" in summary["verdict"]
