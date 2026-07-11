"""
frames.py  --  the Appendix A training frames, model-ready.

transform.py stops at gated, regime-stamped daily aggregates. This module
takes those to frames a model can consume: daily-continuous (padded under
the gate), lagged and rolled through the feature factory, calendar-led,
enriched from environment_config, with the feature list declared in
frame.attrs and passed through assert_no_same_day_cost so an unlagged cost
column can never be a feature.

Three builders, one per product:

  build_frame_1a  pool-level physical model. Target: daily pre_tax_cost per
                  pool from raw_cost's pool branch. Features: lagged cost,
                  lagged/rolled activity from job_usage (via the five-key
                  join), concurrency and job-mix (7.4), calendar, price
                  drift. raw_cost x job_usage is the join at the heart of
                  this frame.

  build_frame_1b  non-pool model, rebuilt per A.2 (July 2026): 38% of spend,
                  not a correction term. Target: daily pre_tax_cost per
                  subscription per SEGMENT (vm_compute / platform),
                  raw_cost's null-pool branch on a date spine. No activity
                  features exist for this estate (register Q2); features are
                  lagged cost, scope (line count), calendar, and the
                  effective-price drift that replaces repr_30d. Training
                  origin 2025-02-01 (after the December glide), stamped as
                  post_glide, selected by training_slice_1b.

  build_frame_2   team attribution model. Target: daily job_cost.cost per
                  job_team, directly from job_cost (no raw_cost join needed
                  for the target). Features: team-aggregated activity via
                  the five-key job_usage x job_cost join, lagged team cost,
                  mix shares, calendar. Unknown is a team; NULL team stays
                  distinct (__NULL_TEAM__). unknown_pct is computed per day
                  (register Q4) as a frame column: it feeds the A.3 filter
                  (filter_unknown) and the A.4 rule from one computation.

Shared disciplines, enforced not aspirational:
  - the 1a + 1b partition sums to raw_cost's grand total
    (assert_partition_identity: the 62.43 lesson);
  - the append-only invariant is re-checked on the snapshot
    (assert_one_write_per_slice: Q1's tripwire);
  - sum over teams equals total attributed cost per day (A.3 additivity);
  - every feature is lagged or calendar-known (assert_no_same_day_cost).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import assertions as A
from . import feature_factory as FF
from . import transform as T

# A.2: train from 2025-02-01, after the December 2024 glide settles. The
# pre-break history is the scenario engine's worked repricing example, not
# training data. Revisit if register Q3 reveals further planned waves.
TRAIN_ORIGIN_1B = date(2025, 2, 1)

# Identity keys / labels that are never features. cost / pre_tax_cost are
# additionally caught by assert_no_same_day_cost, belt-and-braces.
NON_FEATURE_COLS = [
    "run_date", "cost", "pre_tax_cost", "data_regime", "gate_state",
    "gate_complete", "padded", "post_glide", "unknown_pct", "fully_gated",
    "subscription_name",
]


# ---------------------------------------------------------------------------
# enrichment: environment_config
# ---------------------------------------------------------------------------

def derive_region(subscription_name: pd.Series) -> pd.Series:
    """Region from the naming convention in subscription_name (section 4):
    'neu' / 'weu' tokens embed the Azure region. This is the small region
    lookup the doc says will be needed to join internal usage to market
    prices; kept here so the rule exists in exactly one place. Unknown
    patterns map to 'unknown' rather than guessing.
    """
    s = subscription_name.astype(str).str.lower()
    region = pd.Series("unknown", index=s.index, dtype=object)
    region[s.str.contains("neu", na=False)] = "northeurope"
    region[s.str.contains("weu", na=False)] = "westeurope"
    return region


def enrich_environment(
    frame: pd.DataFrame, environment_config: pd.DataFrame
) -> pd.DataFrame:
    """Join environment_config on subscription_id (its primary key), bringing
    environment_tier, environment_sub_tier, subscription_name and a derived
    region. 1:1 by construction; the row-count identity is asserted so a
    duplicated config row cannot silently multiply the frame.
    """
    cols = ["subscription_id", "subscription_name",
            "environment_tier", "environment_sub_tier"]
    present = [c for c in cols if c in environment_config.columns]
    cfg = environment_config[present].drop_duplicates(
        subset=["subscription_id"])
    out = frame.merge(cfg, on="subscription_id", how="left")
    A.assert_row_count_identity(out, len(frame), "frame x environment_config")
    if "subscription_name" in out.columns:
        out["region"] = derive_region(out["subscription_name"])
    return out


# ---------------------------------------------------------------------------
# product 1a: pool-level frame
# ---------------------------------------------------------------------------

POOL_ID_KEYS = ["subscription_id", "batch_account_name", "pool_name"]


def build_frame_1a(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The A.1 training frame. Assembly order:

      1. transform.build_pool_frame: raw_cost pool target x job_usage
         activity (five-key join through job_cost attributes), concurrency,
         job mix, regime stamp, gate.
      2. pad to daily continuity per pool, gate-aware: a gated day with no
         rows is a true zero (the pool cost and ran nothing); an unverifiable
         missing day is excluded, so every padded row is featured_gated by
         construction (asserted).
      3. price drift per pool from the batch slice of raw_cost.
      4. lags, rolls, calendar from the feature factory.
      5. environment_config enrichment.

    Feature list in frame.attrs['feature_cols'], target 'cost'. Log/asinh
    transform of the target is the model layer's job (7.3), as is window
    selection by data_regime.
    """
    pool = T.build_pool_frame(tables)
    gate = T.build_gate(tables["run_status"])

    activity_cols = [c for c in [
        "job_seconds", "task_count", "peak_concurrency", "mean_concurrency",
        "n_jobs", "largest_job_share",
    ] if c in pool.columns]
    share_cols = [c for c in pool.columns if c.startswith("share_")]

    frame = FF.pad_daily(
        pool, POOL_ID_KEYS,
        zero_cols=["cost"] + activity_cols + share_cols,
        gate=gate,
    )
    # padding is gate-aware, so no padded row can predate the run_status
    # era; a padded cost_only row would carry invented zero activity.
    frame = T.stamp_regime(frame)
    if (frame["padded"] & (frame["data_regime"] != "featured_gated")).any():
        raise A.DataQualityError(
            "frame_1a: padded rows outside featured_gated; gate-aware "
            "padding failed")
    frame.loc[frame["padded"], "gate_state"] = "gated_complete"

    # price drift per pool, from the same batch slice that feeds the target
    batch = tables["raw_cost"][tables["raw_cost"]["pool_name"].notna()]
    drift = FF.effective_price_drift(batch, POOL_ID_KEYS)
    n_before = len(frame)
    frame = frame.merge(drift, on=POOL_ID_KEYS + ["run_date"], how="left")
    A.assert_row_count_identity(frame, n_before, "frame_1a x drift")

    frame = FF.add_lags(frame, POOL_ID_KEYS,
                        cols=["cost"], lags=[1, 7])
    frame = FF.add_rolling(frame, POOL_ID_KEYS,
                           cols=["cost"], windows=[28])
    frame = FF.add_lags(
        frame, POOL_ID_KEYS,
        cols=activity_cols + share_cols + ["price_drift"], lags=[1],
    )
    frame = FF.add_rolling(frame, POOL_ID_KEYS,
                           cols=["job_seconds"], windows=[7])
    frame = FF.add_calendar(frame)
    frame = enrich_environment(frame, tables.get(
        "environment_config", pd.DataFrame(columns=["subscription_id"])))

    # same-day columns are raw material for the lags above, never features
    same_day = activity_cols + share_cols + ["price_drift"]
    exclude = NON_FEATURE_COLS + same_day + POOL_ID_KEYS
    feature_cols = FF.feature_columns(frame, exclude=exclude)
    # pool identity is a native categorical feature (A.1); tier/region are
    # enrichment categoricals. They re-enter the feature list deliberately.
    categoricals = [c for c in
                    ["pool_name", "environment_tier",
                     "environment_sub_tier", "region"]
                    if c in frame.columns]
    feature_cols += [c for c in categoricals if c not in feature_cols]

    frame.attrs["target"] = "cost"
    frame.attrs["feature_cols"] = feature_cols
    frame.attrs["categorical_cols"] = categoricals
    frame.attrs["orphan_report"] = pool.attrs.get("orphan_report", {})
    return frame


# ---------------------------------------------------------------------------
# product 1b: non-pool frame
# ---------------------------------------------------------------------------

VM_RESOURCE_PREFIX = "microsoft.compute/virtualmachines"
VM_METER_CATEGORIES = {"Virtual Machines", "Virtual Machines Licences"}
VM_SERVICE_NAMES = {"Virtual Machines"}


def classify_segment(raw_slice: pd.DataFrame) -> pd.Series:
    """A.2's decomposition rule: split the null-pool slice into the volatile,
    break-prone VM compute segment and the platform segment (storage,
    networking, security — the part v18 thought was the whole).

    A row is vm_compute if any of: resource_type under
    microsoft.compute/virtualmachines (which also catches scale sets);
    meter_category Virtual Machines or Virtual Machines Licences (the licence
    line split out at the December 2024 break belongs with the estate it was
    split from); service_name Virtual Machines. Everything else is platform.
    The rule lives in one named, tested function so Q2's answer can amend it
    in one place.
    """
    rt = raw_slice.get("resource_type", pd.Series(index=raw_slice.index,
                                                  dtype=object))
    mc = raw_slice.get("meter_category", pd.Series(index=raw_slice.index,
                                                   dtype=object))
    sn = raw_slice.get("service_name", pd.Series(index=raw_slice.index,
                                                 dtype=object))
    is_vm = (
        rt.astype(str).str.lower().str.startswith(VM_RESOURCE_PREFIX)
        | mc.astype(str).isin(VM_METER_CATEGORIES)
        | sn.astype(str).isin(VM_SERVICE_NAMES)
    )
    return pd.Series(np.where(is_vm, "vm_compute", "platform"),
                     index=raw_slice.index)


def build_frame_1b(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The A.2 training frame, rebuilt. Assembly order:

      1. slice raw_cost to pool_name null; run the two duplicate checks —
         full-row duplicates raise (a double load corrupts every sum), the
         candidate business key is REPORTED not asserted, because the stable
         resource identifier is register Q7 and the daily sum is robust to
         that ambiguity. The report lands in frame.attrs['grain_report'].
      2. classify segments (vm_compute / platform) at line level.
      3. aggregate to (run_date, subscription_id, segment): cost and
         line_count — the scope feature that moved with every genuine scope
         change in the July sessions.
      4. assert the partition: pool branch + non-pool branch = raw_cost's
         grand total, before anything is dropped (the 62.43 lesson).
      5. pad to a date spine per (subscription, segment), gate-aware, so
         days with no pool rows still carry their residual; stamp regimes;
         apply the three-state gate.
      6. effective-price drift per (subscription, segment); lags, rolls,
         calendar; environment enrichment.

    post_glide marks run_date >= 2025-02-01; training_slice_1b selects the
    honest training window. The frame keeps the pre-glide history labelled,
    because the scenario engine wants it as a worked repricing.
    """
    raw_cost = tables["raw_cost"]
    gate = T.build_gate(tables["run_status"])

    nonpool = raw_cost[raw_cost["pool_name"].isna()].copy()

    # (1) duplicate discipline
    A.assert_no_duplicates(
        nonpool, [c for c in nonpool.columns if c != "update_time"],
        "raw_cost[non-pool] (full-row; guards double loads)")
    grain_report = A.report_duplicate_rate(
        nonpool,
        [k for k in ["run_date", "subscription_id", "resource_group_name",
                     "resource_type", "meter"] if k in nonpool.columns],
        "raw_cost[non-pool] candidate key (register Q7)")

    # (2) segments
    nonpool["segment"] = classify_segment(nonpool)

    # (3) daily aggregate at the frame grain
    keys = ["run_date", "subscription_id", "segment"]
    agg = (
        nonpool.groupby(keys, observed=True)
        .agg(cost=("pre_tax_cost", "sum"),
             line_count=("pre_tax_cost", "size"))
        .reset_index()
    )

    # (4) the partition identity, before any row is dropped
    pool_total = float(
        T.daily_cost_by_pool(raw_cost)["cost"].sum())
    A.assert_partition_identity(
        {"pool_branch": pool_total,
         "non_pool_branch": float(agg["cost"].sum())},
        float(raw_cost["pre_tax_cost"].sum()),
        "frame_1b: pool + non-pool vs raw_cost grand total",
    )

    # (5) spine, regimes, gate. The spine is SHARED across segments and
    # subscriptions (the frame's full date range): a gated day with no lines
    # in a segment is a true zero the identity needs, not a gap.
    rd_all = pd.to_datetime(agg["run_date"]).dt.date
    frame = FF.pad_daily(agg, ["subscription_id", "segment"],
                         zero_cols=["cost", "line_count"], gate=gate,
                         spine=(rd_all.min(), rd_all.max()))
    frame = T.stamp_regime(frame)
    frame = T.drop_pre_coverage(frame)
    frame = T.apply_gate(frame, gate, "frame_1b",
                         run_status_start=T.RUN_STATUS_START)

    # (6) price drift, features, enrichment
    drift = FF.effective_price_drift(nonpool, ["subscription_id", "segment"])
    frame = frame.merge(drift, on=keys, how="left")

    frame = FF.add_lags(frame, ["subscription_id", "segment"],
                        cols=["cost"], lags=[1, 7])
    frame = FF.add_rolling(frame, ["subscription_id", "segment"],
                           cols=["cost"], windows=[28])
    frame = FF.add_lags(frame, ["subscription_id", "segment"],
                        cols=["line_count", "price_drift"], lags=[1])
    frame = FF.add_calendar(frame)
    frame = enrich_environment(frame, tables.get(
        "environment_config", pd.DataFrame(columns=["subscription_id"])))

    rd = pd.to_datetime(frame["run_date"]).dt.date
    frame["post_glide"] = rd >= TRAIN_ORIGIN_1B

    exclude = (NON_FEATURE_COLS
               + ["line_count", "price_drift", "subscription_id", "segment"])
    feature_cols = FF.feature_columns(frame, exclude=exclude)
    # A.2: one global model with subscription and segment as native
    # categoricals — pooled strength, preserved additivity.
    categoricals = [c for c in
                    ["subscription_id", "segment", "environment_tier",
                     "environment_sub_tier", "region"] if c in frame.columns]
    feature_cols += [c for c in categoricals if c not in feature_cols]

    frame.attrs["target"] = "cost"
    frame.attrs["feature_cols"] = feature_cols
    frame.attrs["categorical_cols"] = categoricals
    frame.attrs["grain_report"] = grain_report
    frame.attrs["train_origin"] = TRAIN_ORIGIN_1B.isoformat()
    return frame


def training_slice_1b(frame: pd.DataFrame) -> pd.DataFrame:
    """The honest 1b training window: post-glide and gate-verified. The rest
    of the frame stays available, labelled, for the scenario engine and the
    baselines.
    """
    return frame[
        frame["post_glide"] & (frame["data_regime"] == "featured_gated")
    ].copy()


# ---------------------------------------------------------------------------
# product 2: team frame
# ---------------------------------------------------------------------------

UNKNOWN_TEAMS = ("Unknown", "__NULL_TEAM__")


def _label_null_team(job_cost: pd.DataFrame) -> pd.DataFrame:
    """NULL team stays DISTINCT from the Unknown category (3.3): a row the
    classifier never touched is a different fact from a row the classifier
    labelled Unknown, and merging them destroys the drift monitor's signal.
    """
    jc = job_cost.copy()
    if jc["job_team"].dtype.name == "category":
        if "__NULL_TEAM__" not in jc["job_team"].cat.categories:
            jc["job_team"] = jc["job_team"].cat.add_categories(
                ["__NULL_TEAM__"])
    jc["job_team"] = jc["job_team"].fillna("__NULL_TEAM__")
    return jc


def build_frame_2(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The A.3 training frame. Assembly order:

      1. label NULL team; gate job_cost rows at (run_date, subscription_id)
         with the three-state gate.
      2. unknown_pct per run_date — the share of that day's attributed cost
         sitting in Unknown or NULL team. Computed once, here; it feeds both
         the A.3 frame filter (filter_unknown) and the A.4 rule (register
         Q4: the completeness gate is blind to attribution failure).
      3. target grid: (run_date x job_team) complete over the frame's days,
         zero-filled — a team absent on a day attributably ran nothing, and
         the complete grid preserves sum-over-teams = total attributed cost,
         which is asserted per day.
      4. activity per team-day through the five-key job_usage x job_cost
         join (7.5): job_seconds, task_count, n_jobs, and job_seconds share
         per category (the mix features; yesterday's mix predicts today's
         level). Usage orphans land in Unknown; matched rows with NULL team
         stay __NULL_TEAM__, preserving the distinction end to end.
      5. lags, rolls, calendar.

    Target transform (log, or share of daily total — XVA is ~62% of
    attributed spend, a twenty-fold spread) is the model layer's decision;
    the frame carries the raw target. Calendar features must earn their
    place per team (Pillar2 is uniform across the month, register Q20);
    they are provided, not presumed useful.
    """
    gate = T.build_gate(tables["run_status"])
    jc = _label_null_team(tables["job_cost"])
    jc = T.stamp_regime(jc)
    jc = T.drop_pre_coverage(jc)
    jc = T.apply_gate(jc, gate, "frame_2 rows",
                      run_status_start=T.RUN_STATUS_START)

    # (2) unknown_pct per day, on the rows that survive the gate
    day_total = jc.groupby("run_date", observed=True)["cost"].sum()
    day_unknown = (
        jc[jc["job_team"].isin(UNKNOWN_TEAMS)]
        .groupby("run_date", observed=True)["cost"].sum()
        .reindex(day_total.index, fill_value=0.0)
    )
    unknown_pct = (day_unknown / day_total.replace(0, np.nan)).fillna(0.0)
    unknown_pct.name = "unknown_pct"

    # (3) complete (day x team) grid, zero-filled
    per_team = (
        jc.groupby(["run_date", "job_team"], observed=True)["cost"]
        .sum().rename("cost").reset_index()
    )
    days = sorted(per_team["run_date"].unique())
    teams = sorted(per_team["job_team"].astype(str).unique())
    grid = pd.MultiIndex.from_product(
        [days, teams], names=["run_date", "job_team"]).to_frame(index=False)
    frame = grid.merge(per_team, on=["run_date", "job_team"], how="left")
    frame["padded"] = frame["cost"].isna()
    frame["cost"] = frame["cost"].fillna(0.0)

    # additivity: sum over teams per day equals total attributed cost
    grid_daily = frame.groupby("run_date", observed=True)["cost"].sum()
    gap = (grid_daily - day_total.reindex(grid_daily.index)).abs()
    if (gap > 0.01).any():
        bad = gap[gap > 0.01].head(3)
        raise A.DataQualityError(
            f"frame_2: sum over teams != total attributed cost on "
            f"{int((gap > 0.01).sum())} days. Examples: {bad.to_dict()}")

    # per-day gate character: True only if every surviving row that day was
    # gated_complete (ungated Aug-23..Jan-24 rows make a day False)
    fully = (
        jc.assign(_g=jc["gate_state"].eq("gated_complete"))
        .groupby("run_date", observed=True)["_g"].all()
        .rename("fully_gated")
    )
    frame = frame.merge(fully, on="run_date", how="left")
    frame = frame.merge(unknown_pct, on="run_date", how="left")
    frame = T.stamp_regime(frame)

    # (4) activity per team-day via the five-key join. Pre-labelling NULL
    # team means: matched-but-null -> __NULL_TEAM__; genuine usage orphans
    # (no job_cost row at all) -> Unknown from the join's own fillna.
    joined, orphans = T.join_job_attributes(tables["job_usage"], jc)
    activity = (
        joined.groupby(["run_date", "job_team"], observed=True)
        .agg(job_seconds=("job_seconds", "sum"),
             task_count=("task_count", "sum"),
             n_jobs=("job_id", "size"))
        .reset_index()
    )
    frame = frame.merge(activity, on=["run_date", "job_team"], how="left")

    if "job_category" in joined.columns and len(joined):
        cat = (
            joined.groupby(["run_date", "job_team", "job_category"],
                           observed=True)["job_seconds"].sum().reset_index()
        )
        tot = (
            cat.groupby(["run_date", "job_team"],
                        observed=True)["job_seconds"]
            .sum().rename("team_seconds").reset_index()
        )
        cat = cat.merge(tot, on=["run_date", "job_team"], how="left")
        cat["share"] = np.where(cat["team_seconds"] > 0,
                                cat["job_seconds"] / cat["team_seconds"], 0.0)
        wide = cat.pivot_table(observed=True,
                               index=["run_date", "job_team"],
                               columns="job_category", values="share",
                               fill_value=0.0)
        wide.columns = [f"share_{str(c).lower()}" for c in wide.columns]
        frame = frame.merge(wide.reset_index(),
                            on=["run_date", "job_team"], how="left")

    # job_cost's window is inside the activity era, so a team-day with no
    # usage rows genuinely ran nothing: zero is a value here, not a lie
    # (contrast the pool frame's cost_only nulls, which are never filled).
    activity_cols = (["job_seconds", "task_count", "n_jobs"]
                     + [c for c in frame.columns if c.startswith("share_")])
    frame[activity_cols] = frame[activity_cols].fillna(0.0)

    # (5) features
    frame = FF.add_lags(frame, ["job_team"], cols=["cost"], lags=[1, 7])
    frame = FF.add_rolling(frame, ["job_team"], cols=["cost"], windows=[28])
    frame = FF.add_lags(frame, ["job_team"],
                        cols=activity_cols, lags=[1])
    frame = FF.add_calendar(frame)

    exclude = NON_FEATURE_COLS + activity_cols + ["job_team"]
    feature_cols = FF.feature_columns(frame, exclude=exclude)
    feature_cols += ["job_team"]  # native categorical, one global model

    frame.attrs["target"] = "cost"
    frame.attrs["feature_cols"] = feature_cols
    frame.attrs["categorical_cols"] = ["job_team"]
    frame.attrs["orphan_report"] = orphans
    return frame


def filter_unknown(frame: pd.DataFrame,
                   max_unknown_pct: float = 0.20) -> pd.DataFrame:
    """The A.3 frame filter (register Q4): drop days whose attributed cost
    is largely Unknown, separately from the run_status gate. Training on an
    84.7%-Unknown day teaches the model that Unknown is a large volatile
    team. The threshold is a declared tunable: normal days run under 2-4%,
    the pathological days run 54-85%, so 0.20 separates the populations with
    room on both sides.
    """
    return frame[frame["unknown_pct"] <= max_unknown_pct].copy()
