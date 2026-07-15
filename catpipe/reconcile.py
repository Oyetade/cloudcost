"""
reconcile.py  --  does attributed cost (job_cost) reconcile to billed cost
(raw_cost) at pool-day grain?

This is the diagnostic that decides the physical forecast target for product
1a. The tension (see design note 3.3, 7.5 and the target-choice discussion):

  raw_cost   is the BILLED ledger. Ground truth for money spent, reaches back
             to 2021, but has no job_id, so it cannot attribute cost to a job,
             team or category.
  job_cost   has full lineage (job_id, team, category) and joins natively to
             job_usage, but is ATTRIBUTED (very likely DERIVED, open question
             1) and only starts Aug 2023.

At daily-pool grain the job_id is aggregated away on both sides, so the
missing job_id costs the pool model little. What matters instead is whether
job_cost is a faithful re-expression of billed cost or a reweighted /
partial slice of it.

CONFIRMED (real data, 2026): on the pool-days both tables cover, job_cost
equals raw_cost EXACTLY (aggregate ratio 1.0000, every rel_diff 0). But
job_cost omits ~58% of raw_cost's pool-days -- pools that cost money on days
with no attributed job. So job_cost is an EXACT SUBSET of raw_cost restricted
to job-bearing pool-days, not a distortion of it. Consequences:
  - physical pool forecast (1a) targets raw_cost: it is the superset, and
    forecasting the bill needs the pool-days job_cost omits;
  - team/attribution forecast (2) targets job_cost: its subsetting to
    job-bearing pool-days is exactly right, and it carries team/category;
  - raw_cost minus job_cost on the omitted pool-days is idle/unattributed
    provisioned-capacity cost -- a first-class quantity that grows as more
    Prod moves to cloud.

This module answers that empirically. Pure pandas; reads a snapshot dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import transform as T

# Pool-day grain. batch_account_name + pool_name identify the pool; the join
# to activity elsewhere uses the same key.
POOL_DAY_KEYS = ["run_date", "subscription_id", "batch_account_name",
                 "pool_name"]


def _batch_raw_cost_by_pool_day(raw_cost: pd.DataFrame) -> pd.DataFrame:
    """Billed cost aggregated to pool-day, restricted to the batch-attributable
    portion (pool_name not null) so the comparison is like-for-like: job_cost
    only ever covers batch jobs, so raw_cost must be filtered to the same
    scope before comparing, else raw_cost is trivially larger.
    """
    batch = T.batch_slice(raw_cost)
    return (
        batch.groupby(POOL_DAY_KEYS, observed=True, dropna=False)["pre_tax_cost"]
        .sum()
        .rename("raw_cost")
        .reset_index()
    )


def _job_cost_by_pool_day(job_cost: pd.DataFrame) -> pd.DataFrame:
    """Attributed cost aggregated to pool-day."""
    return (
        job_cost.groupby(POOL_DAY_KEYS, observed=True)["cost"]
        .sum()
        .rename("job_cost")
        .reset_index()
    )


def reconcile_pool_day(
    raw_cost: pd.DataFrame,
    job_cost: pd.DataFrame,
    rel_tol: float = 0.05,
) -> pd.DataFrame:
    """Outer-join billed and attributed cost at pool-day and quantify the gap.

    Outer join, not inner: a pool-day present on only one side is itself a
    finding (billed cost with no attribution, or attribution with no matching
    ledger line). Those become raw_cost or job_cost = 0 after filling.

    Returns one row per pool-day with:
      raw_cost, job_cost      : the two aggregates (0 where absent)
      abs_diff                : job_cost - raw_cost
      rel_diff                : abs_diff / raw_cost (NaN where raw_cost == 0)
      coverage                : 'both', 'raw_only' (unattributed billed cost),
                                or 'job_only' (attribution with no ledger)
      within_tol              : |rel_diff| <= rel_tol and coverage == 'both'
    """
    raw = _batch_raw_cost_by_pool_day(raw_cost)
    job = _job_cost_by_pool_day(job_cost)

    merged = raw.merge(job, on=POOL_DAY_KEYS, how="outer", indicator=True)
    merged["coverage"] = merged["_merge"].map({
        "both": "both", "left_only": "raw_only", "right_only": "job_only",
    })
    merged = merged.drop(columns="_merge")
    merged["raw_cost"] = merged["raw_cost"].fillna(0.0)
    merged["job_cost"] = merged["job_cost"].fillna(0.0)

    merged["abs_diff"] = merged["job_cost"] - merged["raw_cost"]
    merged["rel_diff"] = np.where(
        merged["raw_cost"] != 0,
        merged["abs_diff"] / merged["raw_cost"],
        np.nan,
    )
    merged["within_tol"] = (
        (merged["coverage"] == "both")
        & (merged["rel_diff"].abs() <= rel_tol)
    )
    return merged


def reconciliation_summary(recon: pd.DataFrame) -> dict:
    """Roll the pool-day reconciliation up into a verdict. Reports the shape
    of the divergence, not just its size, because HOW it diverges tells you
    what job_cost actually is.
    """
    n = len(recon)
    both = recon[recon["coverage"] == "both"]

    total_raw = recon["raw_cost"].sum()
    total_job = recon["job_cost"].sum()

    # aggregate ratio: does attributed cost recover billed cost in total?
    agg_ratio = (total_job / total_raw) if total_raw else np.nan

    summary = {
        "pool_days": n,
        "coverage_counts": recon["coverage"].value_counts().to_dict(),
        "total_raw_cost": float(total_raw),
        "total_job_cost": float(total_job),
        "aggregate_job_over_raw": float(agg_ratio) if not np.isnan(agg_ratio)
        else None,
        "pct_raw_cost_unattributed": (
            float(recon.loc[recon["coverage"] == "raw_only", "raw_cost"].sum()
                  / total_raw) if total_raw else None
        ),
        "median_abs_rel_diff_both": (
            float(both["rel_diff"].abs().median()) if len(both) else None
        ),
        "share_within_tol_of_both": (
            float(both["within_tol"].mean()) if len(both) else None
        ),
    }
    summary["verdict"] = _verdict(summary)
    return summary


def _verdict(s: dict) -> str:
    """A plain-language read of what the numbers imply for the target choice.

    Two INDEPENDENT dimensions, which the earlier version wrongly conflated:
      (a) do costs AGREE on the pool-days both tables cover? (median rel_diff)
      (b) does job_cost COVER all of raw_cost's pool-days? (unattributed share)

    Confirmed against real data (2026): on shared pool-days job_cost equals
    raw_cost EXACTLY (agg_ratio 1.0000, all rel_diff 0), but job_cost omits
    ~58% of raw_cost's pool-days (pools that cost money on days with no
    attributed job). So job_cost is not distorted -- it is an exact SUBSET of
    raw_cost restricted to job-bearing pool-days. raw_cost is the physical
    target because it is the superset; job_cost is the attribution target;
    raw_cost minus job_cost on the raw_only pool-days is idle/unattributed
    provisioned-capacity cost, a first-class quantity.
    """
    ratio = s["aggregate_job_over_raw"]
    unattr = s["pct_raw_cost_unattributed"]
    median_abs = s["median_abs_rel_diff_both"]
    share_ok = s["share_within_tol_of_both"]

    if ratio is None or share_ok is None or median_abs is None:
        return "insufficient overlap to judge; check coverage_counts"

    agree_on_overlap = median_abs <= 0.02 and share_ok >= 0.9
    job_is_subset = (unattr or 0) >= 0.05  # raw_cost has pool-days job lacks

    if agree_on_overlap and job_is_subset:
        return (
            "job_cost equals raw_cost on shared pool-days but omits "
            f"{(unattr or 0)*100:.0f}% of raw_cost's pool-days: it is an EXACT "
            "SUBSET of raw_cost restricted to job-bearing pool-days, not a "
            "distorted target. raw_cost wins for the physical forecast (1a) "
            "because it is the superset; job_cost is the attribution target "
            "(2); raw_cost minus job_cost on the omitted pool-days is "
            "idle/unattributed capacity cost, worth modelling in its own right."
        )
    if agree_on_overlap and not job_is_subset:
        return (
            "job_cost equals raw_cost on shared pool-days AND covers "
            "essentially all of them: the two are interchangeable at pool "
            "grain. Either serves as the physical target; prefer job_cost only "
            "if its team/category columns are wanted on the same frame."
        )
    if not agree_on_overlap and job_is_subset:
        return (
            "job_cost both disagrees with raw_cost where they overlap AND "
            "omits pool-days. raw_cost wins for the physical forecast; the "
            "disagreement on shared days needs explaining before job_cost is "
            "trusted even for attribution."
        )
    return (
        "job_cost and raw_cost cover the same pool-days but disagree on cost "
        "there. Prefer raw_cost for the physical forecast; inspect the largest "
        "rel_diff pool-days to understand the disagreement."
    )
