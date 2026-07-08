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
partial slice of it. If they agree at pool-day, job_cost is safe to use as
the 1a target from Aug 2023 (cleaner join to activity) and raw_cost need
only supply the pre-2023 tail. If they diverge, the divergence IS the
attribution distortion you would otherwise forecast, and raw_cost wins for
1a decisively.

This module answers that empirically. Pure pandas; reads a snapshot dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    batch = raw_cost[raw_cost["pool_name"].notna()]
    return (
        batch.groupby(POOL_DAY_KEYS)["pre_tax_cost"]
        .sum()
        .rename("raw_cost")
        .reset_index()
    )


def _job_cost_by_pool_day(job_cost: pd.DataFrame) -> pd.DataFrame:
    """Attributed cost aggregated to pool-day."""
    return (
        job_cost.groupby(POOL_DAY_KEYS)["cost"]
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
    """A plain-language read of what the numbers imply for the target choice."""
    ratio = s["aggregate_job_over_raw"]
    unattr = s["pct_raw_cost_unattributed"]
    share_ok = s["share_within_tol_of_both"]

    if ratio is None or share_ok is None:
        return "insufficient overlap to judge; check coverage_counts"

    if 0.97 <= ratio <= 1.03 and share_ok >= 0.9 and (unattr or 0) < 0.05:
        return ("job_cost faithfully re-expresses billed cost at pool-day. "
                "Safe to use job_cost as the 1a target from Aug 2023 for the "
                "cleaner activity join; raw_cost supplies the pre-2023 tail.")
    if ratio < 0.9 or (unattr or 0) >= 0.1:
        return ("job_cost recovers only part of billed cost (unattributed "
                "spend or a reweighted slice). It is a distorted target for a "
                "physical forecast: raw_cost wins for 1a; job_cost stays the "
                "team-model target only.")
    return ("job_cost and raw_cost are close but not tight. Usable with "
            "caution; prefer raw_cost for 1a unless the residual is explained. "
            "Inspect the largest rel_diff pool-days before deciding.")
