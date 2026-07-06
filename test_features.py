"""Tests for features.py: the concurrency sweep and job-mix."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from catpipe import features as F

POOL = ["subscription_id", "batch_account_name", "pool_name"]


def _job(sub, acct, pool, jid, start, end, cat="BT", secs=100.0):
    return {
        "subscription_id": sub, "batch_account_name": acct, "pool_name": pool,
        "job_id": jid, "start_time": pd.Timestamp(start, tz="UTC"),
        "end_time": pd.Timestamp(end, tz="UTC"), "job_category": cat,
        "job_seconds": secs, "run_date": pd.Timestamp(start).date(),
    }


class TestConcurrency:
    def test_two_overlapping_jobs_peak_two(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 11:00"),
            _job("s", "a", "p", "j2", "2024-03-01 10:00", "2024-03-01 12:00"),
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert out["peak_concurrency"].iloc[0] == 2

    def test_sequential_jobs_peak_one(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 10:00"),
            _job("s", "a", "p", "j2", "2024-03-01 11:00", "2024-03-01 12:00"),
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert out["peak_concurrency"].iloc[0] == 1

    def test_instantaneous_handover_does_not_inflate_peak(self):
        # j1 ends exactly when j2 starts: end applied before start => peak 1
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 10:00"),
            _job("s", "a", "p", "j2", "2024-03-01 10:00", "2024-03-01 11:00"),
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert out["peak_concurrency"].iloc[0] == 1

    def test_pools_counted_independently(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p1", "j1", "2024-03-01 09:00", "2024-03-01 12:00"),
            _job("s", "a", "p2", "j2", "2024-03-01 09:00", "2024-03-01 12:00"),
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert (out["peak_concurrency"] == 1).all()
        assert len(out) == 2

    def test_null_times_dropped(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 11:00"),
            {**_job("s", "a", "p", "j2", "2024-03-01 10:00",
                    "2024-03-01 12:00"), "end_time": pd.NaT},
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        # only j1 survives => peak 1
        assert out["peak_concurrency"].iloc[0] == 1

    def test_empty_input_returns_empty_with_schema(self):
        jobs = pd.DataFrame(columns=list(_job("s", "a", "p", "j", "2024-03-01",
                                               "2024-03-01").keys()))
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert out.empty
        assert "peak_concurrency" in out.columns

    def test_triple_overlap(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 12:00"),
            _job("s", "a", "p", "j2", "2024-03-01 09:30", "2024-03-01 12:00"),
            _job("s", "a", "p", "j3", "2024-03-01 10:00", "2024-03-01 12:00"),
        ])
        out = F.concurrency_by_pool_day(jobs, POOL)
        assert out["peak_concurrency"].iloc[0] == 3


class TestJobMix:
    def test_shares_sum_to_one(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 10:00",
                 cat="BT", secs=300.0),
            _job("s", "a", "p", "j2", "2024-03-01 09:00", "2024-03-01 10:00",
                 cat="RMT", secs=100.0),
        ])
        out = F.job_mix_by_pool_day(jobs, POOL)
        share_cols = [c for c in out.columns if c.startswith("share_")]
        assert out[share_cols].sum(axis=1).round(6).iloc[0] == 1.0

    def test_largest_job_share(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", "j1", "2024-03-01 09:00", "2024-03-01 10:00",
                 secs=300.0),
            _job("s", "a", "p", "j2", "2024-03-01 09:00", "2024-03-01 10:00",
                 secs=100.0),
        ])
        out = F.job_mix_by_pool_day(jobs, POOL)
        assert out["largest_job_share"].iloc[0] == pytest.approx(0.75)

    def test_n_jobs(self):
        jobs = pd.DataFrame([
            _job("s", "a", "p", f"j{i}", "2024-03-01 09:00",
                 "2024-03-01 10:00") for i in range(5)
        ])
        out = F.job_mix_by_pool_day(jobs, POOL)
        assert out["n_jobs"].iloc[0] == 5
