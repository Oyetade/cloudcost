"""
features.py  --  the feature constructions that are real algorithm, not a
library one-liner, isolated here so they can be tested in isolation.

The concurrency sweep is the piece that was a clean tsrange overlap in SQL
and becomes a sort-and-sweep in pandas. It proxies the "number of machines"
driver (7.4) that neither job_seconds nor task_count captures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency_by_pool_day(
    jobs: pd.DataFrame,
    pool_keys: list[str],
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> pd.DataFrame:
    """Peak and mean concurrent jobs per pool per day, via an event sweep.

    For each pool, emit +1 at every start and -1 at every end, sort by time,
    cumulative-sum to get the running count, and reduce to daily peak/mean.
    This counts how many jobs overlapped at each moment: the machine-count
    proxy of 7.4.

    Ties: when a start and an end share a timestamp, ends are applied before
    starts (a job ending frees the slot the incoming job takes), so an
    instantaneous handover does not inflate the peak. Achieved by sorting
    ends (-1) ahead of starts (+1) at equal time via a secondary key.

    Rows with a null start or end are dropped (job_usage allows nulls per the
    DDL); the caller should report how many were dropped.

    Returns one row per (pool_keys..., day) with peak_concurrency and
    mean_concurrency.
    """
    df = jobs.dropna(subset=[start_col, end_col]).copy()
    if df.empty:
        cols = pool_keys + ["day", "peak_concurrency", "mean_concurrency"]
        return pd.DataFrame(columns=cols)

    starts = df[pool_keys + [start_col]].rename(columns={start_col: "t"})
    starts["delta"] = 1
    ends = df[pool_keys + [end_col]].rename(columns={end_col: "t"})
    ends["delta"] = -1

    events = pd.concat([starts, ends], ignore_index=True)
    events["day"] = pd.to_datetime(events["t"]).dt.date
    # ends (delta -1) sort before starts (delta +1) at equal timestamp
    events = events.sort_values(pool_keys + ["t", "delta"]).reset_index(drop=True)
    events["running"] = events.groupby(pool_keys)["delta"].cumsum()

    out = (
        events.groupby(pool_keys + ["day"])["running"]
        .agg(peak_concurrency="max", mean_concurrency="mean")
        .reset_index()
    )
    out["mean_concurrency"] = out["mean_concurrency"].astype(float)
    return out


def job_mix_by_pool_day(
    jobs: pd.DataFrame,
    pool_keys: list[str],
    day_col: str = "run_date",
    category_col: str = "job_category",
    seconds_col: str = "job_seconds",
) -> pd.DataFrame:
    """Job-mix features (7.4): share of daily job_seconds by category, count
    of distinct jobs, and the largest single job's share. Mix shifts precede
    cost shifts when a new workload ramps.

    Returns one row per (pool_keys..., day) with:
      n_jobs, largest_job_share, and one share column per category.
    """
    df = jobs.copy()
    grp = pool_keys + [day_col]

    totals = df.groupby(grp)[seconds_col].sum().rename("total_seconds")
    n_jobs = df.groupby(grp).size().rename("n_jobs")
    largest = df.groupby(grp)[seconds_col].max().rename("largest_seconds")

    base = pd.concat([totals, n_jobs, largest], axis=1).reset_index()
    base["largest_job_share"] = np.where(
        base["total_seconds"] > 0,
        base["largest_seconds"] / base["total_seconds"],
        0.0,
    )

    cat = (
        df.groupby(grp + [category_col])[seconds_col]
        .sum()
        .reset_index()
    )
    cat = cat.merge(base[grp + ["total_seconds"]], on=grp, how="left")
    cat["share"] = np.where(
        cat["total_seconds"] > 0, cat[seconds_col] / cat["total_seconds"], 0.0
    )
    wide = cat.pivot_table(
        index=grp, columns=category_col, values="share", fill_value=0.0
    )
    wide.columns = [f"share_{c}" for c in wide.columns]
    wide = wide.reset_index()

    result = base[grp + ["n_jobs", "largest_job_share"]].merge(
        wide, on=grp, how="left"
    )
    return result
