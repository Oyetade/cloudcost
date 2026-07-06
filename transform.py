"""
transform.py  --  section 7 programme, from the Parquet snapshot to the
Appendix A training frames.

Order matches the note: gate first (7.3), then the five-key join (7.5), then
the priceable mask and daily aggregation. Every join is followed by an
assertion from assertions.py. Nothing here reaches the modelling code; it
stops at the frames.

Reads the Parquet snapshot produced by extract.py. Pure pandas + numpy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import assertions as A
from . import features as F

GATE_TYPES = ("Cost", "Usage", "Attribution")
JOB_KEYS = ["run_date", "subscription_id", "batch_account_name",
            "pool_name", "job_id"]
POOL_KEYS = ["run_date", "subscription_id", "batch_account_name", "pool_name"]


def load_snapshot(snapshot_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read every table Parquet in a snapshot directory into a dict."""
    d = Path(snapshot_dir)
    tables = {}
    for p in d.glob("*.parquet"):
        tables[p.stem] = pd.read_parquet(p)
    return tables


def build_gate(run_status: pd.DataFrame) -> pd.DataFrame:
    """The run_status completeness gate (7.3).

    run_status is keyed by (run_date, subscription_id, run_type, run_time),
    so the same run type can appear more than once per subscription-day.
    Take the LATEST run per (date, subscription, type) before pivoting, then
    a slice is complete iff Cost, Usage AND Attribution all read Complete.

    Written as 'status == Complete for all three' rather than 'status !=
    some failure', so any unanticipated status fails safe.

    Returns one row per (run_date, subscription_id) with a gate_complete bool.
    """
    rs = run_status.copy()
    # latest run per (date, subscription, type): update_time is the tiebreaker
    sort_col = "update_time" if "update_time" in rs.columns else "run_time"
    rs = (
        rs.sort_values(sort_col)
        .drop_duplicates(subset=["run_date", "subscription_id", "run_type"],
                         keep="last")
    )
    rs["is_complete"] = rs["status"].astype(str).eq("Complete")
    piv = rs.pivot_table(
        index=["run_date", "subscription_id"],
        columns="run_type",
        values="is_complete",
        aggfunc="first",
    )
    # a type absent for a slice => NaN => treat as not complete (fail-safe)
    for t in GATE_TYPES:
        if t not in piv.columns:
            piv[t] = False
    piv[list(GATE_TYPES)] = piv[list(GATE_TYPES)].fillna(False).astype(bool)
    piv["gate_complete"] = piv[list(GATE_TYPES)].all(axis=1)
    return piv.reset_index()[["run_date", "subscription_id", "gate_complete"]]


def apply_gate(
    frame: pd.DataFrame, gate: pd.DataFrame, context: str
) -> pd.DataFrame:
    """Left-join the gate onto the full grid so absence counts as failure
    (an inner join would silently pass missing days through), then keep only
    complete slices. Absent gate rows => gate_complete NaN => filtered out.
    """
    merged = frame.merge(gate, on=["run_date", "subscription_id"], how="left")
    merged["gate_complete"] = merged["gate_complete"].fillna(False)
    kept = merged[merged["gate_complete"]].copy()
    A.assert_gate_complete(kept, context)
    return kept


def join_job_attributes(
    job_usage: pd.DataFrame, job_cost: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Five-key left join bringing job_cost's static attributes onto
    job_usage (7.5). Left join from job_usage so usage orphans are kept, not
    silently discarded. Row count must equal job_usage's (validation two).

    Brings across job_name, job_category, job_ownership, job_team ONLY.
    Never same-day cost. Returns (joined, orphan_report).
    """
    attr_cols = ["job_name", "job_category", "job_ownership", "job_team"]
    present = [c for c in attr_cols if c in job_cost.columns]

    orphans = A.report_anti_join(
        job_usage, job_cost, JOB_KEYS, "job_usage", "job_cost"
    )

    # The five-key must be unique on job_usage itself, else a retried job
    # (same id within a day) inflates every downstream activity aggregate.
    # The row-count identity below compares against len(job_usage) and so
    # cannot catch a left-side duplicate; check it explicitly here.
    A.assert_no_duplicates(job_usage, JOB_KEYS, "job_usage")

    right = job_cost[JOB_KEYS + present].drop_duplicates(subset=JOB_KEYS)
    joined = job_usage.merge(right, on=JOB_KEYS, how="left")
    A.assert_row_count_identity(joined, len(job_usage), "job_usage x job_cost")

    # usage orphans get explicit Unknown rather than NaN
    for c in present:
        if joined[c].dtype.name == "category":
            if "Unknown" not in joined[c].cat.categories:
                joined[c] = joined[c].cat.add_categories(["Unknown"])
        joined[c] = joined[c].fillna("Unknown")
    return joined, orphans


def priceable_mask(raw_cost: pd.DataFrame) -> pd.Series:
    """The priceable mask (section 9 / 5.6): rows where an effective price is
    defined. Many ledger lines carry zero cost with tiny usage (free-tier
    storage ops) which make effective prices undefined. Effective price =
    pre_tax_cost / usage_quantity is only meaningful where usage_quantity > 0
    and cost > 0. Variance decomposition uses effective prices under THIS
    mask, never retail prices.
    """
    return (raw_cost["usage_quantity"] > 0) & (raw_cost["pre_tax_cost"] > 0)


def daily_cost_by_pool(raw_cost: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw_cost to daily pool grain: the physical target (product
    1a). Batch-associated rows only (pool_name not null); the non-pool
    residual is handled separately (product 1b).
    """
    batch = raw_cost[raw_cost["pool_name"].notna()].copy()
    agg = (
        batch.groupby(POOL_KEYS)["pre_tax_cost"]
        .sum()
        .rename("cost")
        .reset_index()
    )
    return agg


def build_pool_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Product 1a: pool-level daily frame. Target = daily pool cost; features
    = activity aggregates + concurrency + job-mix, all lagged or
    calendar-known. This skeleton assembles target + activity features and
    applies the gate; lagging/calendar leads are added by the feature factory
    downstream (kept out of the skeleton to stay focused).
    """
    raw_cost = tables["raw_cost"]
    job_usage = tables["job_usage"]
    job_cost = tables["job_cost"]
    gate = build_gate(tables["run_status"])

    A.assert_no_duplicates(raw_cost, ["run_date", "subscription_id",
                                      "resource_group_name", "resource_type",
                                      "meter"], "raw_cost")

    target = daily_cost_by_pool(raw_cost)

    joined, orphans = join_job_attributes(job_usage, job_cost)
    activity = (
        joined.groupby(POOL_KEYS)
        .agg(job_seconds=("job_seconds", "sum"),
             task_count=("task_count", "sum"))
        .reset_index()
    )
    concurrency = F.concurrency_by_pool_day(
        joined, POOL_KEYS[:0] + ["subscription_id", "batch_account_name",
                                 "pool_name"]
    )

    frame = target.merge(activity, on=POOL_KEYS, how="left")
    frame = apply_gate(frame, gate, "pool_frame")
    frame.attrs["orphan_report"] = orphans
    return frame


def build_team_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Product 2: team-level daily frame. Target = daily job_cost aggregated
    by team. job_team is nullable while job_category/job_ownership are NOT
    NULL, so NULL team is kept DISTINCT from the Unknown category (3.3).
    """
    job_cost = tables["job_cost"]
    gate = build_gate(tables["run_status"])

    jc = job_cost.copy()
    # NULL team distinct from Unknown category: label it explicitly
    if jc["job_team"].dtype.name == "category":
        if "__NULL_TEAM__" not in jc["job_team"].cat.categories:
            jc["job_team"] = jc["job_team"].cat.add_categories(["__NULL_TEAM__"])
    jc["job_team"] = jc["job_team"].fillna("__NULL_TEAM__")

    target = (
        jc.groupby(["run_date", "subscription_id", "job_team"])["cost"]
        .sum()
        .rename("cost")
        .reset_index()
    )
    frame = apply_gate(target, gate, "team_frame")
    return frame
