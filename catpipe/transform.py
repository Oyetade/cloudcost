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

from datetime import date
from pathlib import Path

import pandas as pd

# Regime boundaries (section 5.7). Feature availability, not just gate
# availability, differs across these. Model layer filters on data_regime;
# the transform only labels.
#
# COST_HISTORY_START: raw_cost is continuous from Jan 2021, but the monthly
# row count AND monthly total pre_tax_cost both ramp together through 2021,
# flattening around Aug 2022. That lockstep ramp is genuine onboarding
# (workloads progressively brought under management), not a logging-
# granularity change, so pre-Aug-2022 totals UNDERSTATE the true estate and
# must not train a level-based target. Rows before this floor are labelled
# 'pre_coverage' and excluded from training frames (they stay in the raw
# extract). Single tunable: move it if a later cost-vs-rowcount review shifts
# the observed plateau.
COST_HISTORY_START = date(2022, 8, 1)  # onboarding ramp complete; estate stable
ACTIVITY_START = date(2023, 8, 1)   # job_usage / job_cost begin
RUN_STATUS_START = date(2024, 1, 2)  # run_status begins; gate evaluable

from . import assertions as A
from . import features as F

GATE_TYPES = ("Cost", "Usage", "Attribution")
JOB_KEYS = ["run_date", "subscription_id", "batch_account_name",
            "pool_name", "job_id"]
POOL_KEYS = ["run_date", "subscription_id", "batch_account_name", "pool_name"]

# The batch-account sentinel (Q23, 14 July 2026). pool_name not-null is the
# batch-branch filter, but batch_account_name can still be null on those rows:
# pool-named lines exist on non-Batch resource types (vmss, disks,
# operationalinsights). pandas groupby defaults to dropna=True, so those rows
# are filtered IN and then silently grouped OUT, and the pool branch under-sums
# by exactly their cost. That is what broke assert_partition_identity by
# 17,585.93 with no candidate mechanism in the data: the leak was a groupby
# default, not a dropped row upstream.
#
# Filling before grouping, rather than passing dropna=False, keeps the null
# explicit and distinct in the key — the same discipline __NULL_TEAM__ enforces
# end to end (section 5). dropna=False would leave NaN group keys that compare
# unequal to themselves and poison every downstream merge on POOL_KEYS.
NULL_BATCH = "__NULL_BATCH__"


def batch_slice(raw_cost: pd.DataFrame) -> pd.DataFrame:
    """The batch branch of raw_cost: pool_name not null, with a null-safe
    batch_account_name. Every batch-branch aggregation goes through here, so
    the partition identity holds by construction rather than by each caller
    remembering the sentinel.
    """
    batch = raw_cost[raw_cost["pool_name"].notna()].copy()
    if "batch_account_name" in batch.columns:
        col = batch["batch_account_name"]
        if isinstance(col.dtype, pd.CategoricalDtype):
            if NULL_BATCH not in col.cat.categories:
                col = col.cat.add_categories([NULL_BATCH])
        batch["batch_account_name"] = col.fillna(NULL_BATCH)
    return batch


def stamp_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Stamp each row with its data_regime (section 5.7), so the model layer
    can select its own window rather than the transform hardcoding a date.

      pre_coverage     : before COST_HISTORY_START. Onboarding ramp; totals
                         understate the true estate. Excluded from training
                         frames (see drop_pre_coverage), kept in raw extract.
      cost_only        : COST_HISTORY_START .. ACTIVITY_START. Representative
                         cost target, but activity features are null BY
                         CONSTRUCTION, never imputed.
      featured_ungated : ACTIVITY_START .. RUN_STATUS_START. Featured, but
                         no gate could be evaluated.
      featured_gated   : on/after RUN_STATUS_START. Featured and gated.
    """
    rd = pd.to_datetime(frame["run_date"]).dt.date
    regime = pd.Series("featured_gated", index=frame.index, dtype=object)
    regime[rd < RUN_STATUS_START] = "featured_ungated"
    regime[rd < ACTIVITY_START] = "cost_only"
    regime[rd < COST_HISTORY_START] = "pre_coverage"
    frame = frame.copy()
    frame["data_regime"] = pd.Categorical(
        regime,
        categories=["pre_coverage", "cost_only",
                    "featured_ungated", "featured_gated"],
    )
    return frame


def drop_pre_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove onboarding-ramp rows from a training frame. The transform keeps
    pre_coverage rows labelled rather than silently dropping them, so this
    exclusion is an explicit, testable step the caller can see and audit.
    """
    return frame[frame["data_regime"] != "pre_coverage"].copy()


def load_snapshot(
    snapshot_dir: str | Path, filter_tiers: bool = True
) -> dict[str, pd.DataFrame]:
    """Read every table Parquet in a snapshot directory into a dict.

    DEV and TEST tiers are dropped at the gate by default (see
    filter_environment_tiers). Pass filter_tiers=False to load the raw estate,
    which the reconciliation to invoice needs: the invoice bills every tier, so
    a like-for-like comparison must not pre-filter.
    """
    d = Path(snapshot_dir)
    tables = {}
    for p in d.glob("*.parquet"):
        tables[p.stem] = pd.read_parquet(p)
    if filter_tiers:
        tables = filter_environment_tiers(tables)
    return tables


EXCLUDED_TIERS = ("DEV", "TEST")


def filter_environment_tiers(
    tables: dict[str, pd.DataFrame],
    excluded_tiers: tuple[str, ...] = EXCLUDED_TIERS,
) -> dict[str, pd.DataFrame]:
    """Drop excluded environment tiers from EVERY table, at the gate, before
    any aggregation or assertion sees the data (register Q26, SME guidance 14
    July 2026).

    Why here and not at enrichment: enrich_environment runs at step 5 of the
    frame builders, after the partition identity is asserted against
    raw_cost's grand total and after lags and rolling windows are computed.
    Filtering there would (a) reopen Q23 by shrinking the branches while the
    grand total kept the excluded rows, (b) leak DEV history into PROD
    features through the rolling windows, and (c) leave the excluded rows in
    every other table. The universe has to be narrowed once, at the top, so
    that grand total, branches, activity and gate all agree on what exists.

    Why DEV and TEST are excluded on principle, not just on instruction: from
    December 2024 the estate runs hybrid pools, spot-first with a dedicated
    fallback. DEV and TEST have no fallback: when spot capacity is
    unavailable nothing runs and the cost is zero, because their consumers
    will simply wait for capacity. Their daily cost is therefore a
    spot-AVAILABILITY series driven by Azure's auction, which nothing in this
    snapshot observes. No feature in raw_cost or job_usage could predict it.
    Excluding an exogenously-driven target is a modelling decision the charter
    can defend, not a convenience.

    PROD and PREPROD are what remain. PROD's dedicated fallback is confirmed,
    so its work always happens and its cost is forecastable from the features
    here. PREPROD is NOT confirmed either way (register Q27): if preprod
    consumers also wait for spot rather than falling back, preprod carries the
    same exogenous availability signal as DEV and belongs in EXCLUDED_TIERS.
    It is kept for now because excluding a tier on a guess is the worse error:
    a tier wrongly kept shows up as poor accuracy that can be diagnosed, while
    a tier wrongly dropped is spend that silently leaves the estate.

    environment_tier lives on environment_config keyed by subscription_id, so
    this is a subscription-level exclusion: name-matching on resource groups
    is not needed and would be brittle.

    Tables without a subscription_id are passed through untouched. A tier that
    is null (a subscription absent from environment_config) is KEPT: silence
    in the config is not evidence of DEV, and dropping unknowns would quietly
    shrink the estate.
    """
    cfg = tables.get("environment_config")
    if cfg is None or "environment_tier" not in cfg.columns:
        return tables

    tier = cfg["environment_tier"].astype(str).str.strip().str.upper()
    excluded = {t.strip().upper() for t in excluded_tiers}
    drop_subs = set(cfg.loc[tier.isin(excluded), "subscription_id"])
    if not drop_subs:
        return tables

    out = {}
    for name, df in tables.items():
        if "subscription_id" in df.columns:
            out[name] = df[~df["subscription_id"].isin(drop_subs)].copy()
        else:
            out[name] = df
    return out


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
        observed=True,
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
    frame: pd.DataFrame, gate: pd.DataFrame, context: str,
    run_status_start: "date | None" = None,
) -> pd.DataFrame:
    """Three-state gate. A missing gate row means two different things
    depending on when it falls, and conflating them silently discards the
    pre-run_status history (section 5.7):

      - on/after run_status_start: missing => the load is unverified =>
        FAIL, exclude the row (an inner join would pass it through silently);
      - before run_status_start: missing => run_status did not yet exist =>
        UNGATED, keep the row but mark it so downstream knows it passed no
        gate.

    Adds a gate_state column with values 'gated_complete', 'gated_failed',
    'ungated'. Rows with 'gated_failed' are dropped; the other two are kept.
    If run_status_start is None it is inferred as the gate's own minimum
    run_date, which is the correct default when the gate is built from the
    same snapshot.
    """
    merged = frame.merge(gate, on=["run_date", "subscription_id"], how="left")

    if run_status_start is None and len(gate):
        run_status_start = gate["run_date"].min()

    before_era = (
        merged["run_date"] < run_status_start
        if run_status_start is not None
        else pd.Series(False, index=merged.index)
    )
    complete = merged["gate_complete"].fillna(False)

    state = pd.Series("gated_failed", index=merged.index, dtype=object)
    state[complete] = "gated_complete"
    state[before_era & merged["gate_complete"].isna()] = "ungated"
    merged["gate_state"] = state

    kept = merged[merged["gate_state"] != "gated_failed"].copy()
    A.assert_no_failed_gate(kept, context)
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
    batch = batch_slice(raw_cost)
    agg = (
        batch.groupby(POOL_KEYS, observed=True, dropna=False)["pre_tax_cost"]
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

    # Duplicate check applies to the BATCH slice only. The pool-key
    # (…, batch_account_name, pool_name) is a batch-shaped identity; on
    # non-batch rows (storage, network, etc.) pool_name and batch_account_name
    # are null, so that key legitimately repeats — a storage account can bill
    # many "Read Operations" lines a day. Running the check on the whole table
    # flags those benign non-batch rows as duplicates (observed: 310k of them,
    # all null-pool). Since daily_cost_by_pool uses only pool-not-null rows,
    # check exactly that slice. Non-batch raw_cost has its own grain and its
    # own duplicate check belongs with product 1b (the non-pool residual).
    # Q24 (14 July 2026): this key was hand-written and omitted
    # meter_sub_category, so it flagged 142 legitimate Azure rows as
    # duplicates: Storage/Read Operations billed separately for Files and
    # Tables is two rows, not one row twice. Use the shared grain constant so
    # the two checks cannot drift apart again.
    batch_rows = raw_cost[raw_cost["pool_name"].notna()]
    A.assert_no_duplicates(batch_rows, A.grain_present(batch_rows),
                           "raw_cost[batch]")

    target = daily_cost_by_pool(raw_cost)

    joined, orphans = join_job_attributes(job_usage, job_cost)
    activity = (
        joined.groupby(POOL_KEYS, observed=True)
        .agg(job_seconds=("job_seconds", "sum"),
             task_count=("task_count", "sum"))
        .reset_index()
    )

    # Concurrency (7.4): the machine-count proxy. The sweep dates events by
    # their timestamps, so its day column can differ from run_date for jobs
    # spanning midnight; rename and join on the pool-day key regardless —
    # run_date is the frame's grain and the sweep's day is its best estimate
    # of when the concurrency occurred.
    pool_id_keys = ["subscription_id", "batch_account_name", "pool_name"]
    concurrency = F.concurrency_by_pool_day(joined, pool_id_keys)
    if len(concurrency):
        concurrency = concurrency.rename(columns={"day": "run_date"})

    # Job mix (7.4): category shares, distinct-job count, largest job's
    # share. Mix shifts precede cost shifts when a new workload ramps.
    if len(joined):
        job_mix = F.job_mix_by_pool_day(joined, pool_id_keys)
    else:
        job_mix = pd.DataFrame(
            columns=pool_id_keys + ["run_date", "n_jobs", "largest_job_share"]
        )

    # Left join from the cost target: pre-ACTIVITY_START pool-days get NaN
    # activity. That NaN is null-BY-CONSTRUCTION (we did not measure), NOT
    # zero activity (a pool that ran nothing). Never fillna(0) here; the
    # regime label explains the null and the model layer decides what to do.
    frame = target.merge(activity, on=POOL_KEYS, how="left")
    if len(concurrency):
        frame = frame.merge(concurrency, on=POOL_KEYS, how="left")
    else:
        frame["peak_concurrency"] = pd.NA
        frame["mean_concurrency"] = pd.NA
    if len(job_mix):
        frame = frame.merge(job_mix, on=POOL_KEYS, how="left")
    else:
        frame["n_jobs"] = pd.NA
        frame["largest_job_share"] = pd.NA
    A.assert_row_count_identity(frame, len(target),
                                "pool_frame x concurrency/job_mix")
    frame = stamp_regime(frame)
    frame = drop_pre_coverage(frame)
    frame = apply_gate(frame, gate, "pool_frame",
                       run_status_start=RUN_STATUS_START)
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
        jc.groupby(["run_date", "subscription_id", "job_team"], observed=True)["cost"]
        .sum()
        .rename("cost")
        .reset_index()
    )
    # job_cost starts at ACTIVITY_START, so the team frame has no cost_only
    # or pre_coverage regime in practice; drop_pre_coverage is a no-op here
    # but kept for symmetry and safety against backfilled early rows.
    target = stamp_regime(target)
    target = drop_pre_coverage(target)
    frame = apply_gate(target, gate, "team_frame",
                       run_status_start=RUN_STATUS_START)
    return frame
