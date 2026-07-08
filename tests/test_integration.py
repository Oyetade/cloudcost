"""End-to-end integration: full pipeline against a synthetic snapshot.

Guards that the pieces compose, not just pass in isolation: the gate excludes
incomplete slices, the priceable mask drops free-tier zero rows, activity
aggregates join on the pool key, and NULL team stays distinct from Unknown.
"""

from datetime import date

import pandas as pd
import pytest

from catpipe import transform as T


def _ts(x):
    return pd.Timestamp(x, tz="UTC")


@pytest.fixture
def snapshot():
    raw_cost = pd.DataFrame([
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             resource_group_name="rg", resource_type="vmss", meter="D64 Spot",
             batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=100.0, usage_quantity=10.0),
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             resource_group_name="rg", resource_type="disk", meter="S10",
             batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=0.0, usage_quantity=5.0),  # free-tier, unpriceable
        dict(run_date=date(2024, 3, 2), subscription_id="s",
             resource_group_name="rg", resource_type="vmss", meter="D64 Spot",
             batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=200.0, usage_quantity=20.0),  # incomplete slice
        dict(run_date=date(2022, 6, 1), subscription_id="s",
             resource_group_name="rg", resource_type="vmss", meter="D64 Spot",
             batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=50.0, usage_quantity=5.0),  # pre_coverage (< Aug 22)
        dict(run_date=date(2022, 10, 1), subscription_id="s",
             resource_group_name="rg", resource_type="vmss", meter="D64 Spot",
             batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=70.0, usage_quantity=7.0),  # cost_only (>= Aug 22)
    ])
    job_usage = pd.DataFrame([
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             batch_account_name="a", pool_name="eodpool", job_id="BT_j1",
             start_time=_ts("2024-03-01 09:00"),
             end_time=_ts("2024-03-01 11:00"),
             job_seconds=7200.0, task_count=3),
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             batch_account_name="a", pool_name="eodpool", job_id="BT_j2",
             start_time=_ts("2024-03-01 10:00"),
             end_time=_ts("2024-03-01 12:00"),
             job_seconds=7200.0, task_count=2),
    ])
    job_cost = pd.DataFrame([
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             batch_account_name="a", pool_name="eodpool", job_id="BT_j1",
             job_name="BT", job_category="BT", job_ownership="Risk",
             job_team="Pillar1", cost=60.0),
        dict(run_date=date(2024, 3, 1), subscription_id="s",
             batch_account_name="a", pool_name="eodpool", job_id="BT_j2",
             job_name="BT", job_category="BT", job_ownership="Risk",
             job_team=None, cost=40.0),  # NULL team
    ])

    def rs(rd, rt, st):
        return dict(run_date=rd, subscription_id="s", run_type=rt,
                    status=st, update_time=_ts("2024-03-03"))

    run_status = pd.DataFrame([
        rs(date(2024, 3, 1), "Cost", "Complete"),
        rs(date(2024, 3, 1), "Usage", "Complete"),
        rs(date(2024, 3, 1), "Attribution", "Complete"),
        rs(date(2024, 3, 2), "Cost", "Complete"),
        rs(date(2024, 3, 2), "Usage", "Complete"),
        # 3/2 Attribution missing => gated out
    ])
    return dict(raw_cost=raw_cost, job_usage=job_usage,
                job_cost=job_cost, run_status=run_status)


def test_pool_frame_gates_masks_and_drops_pre_coverage(snapshot):
    pool = T.build_pool_frame(snapshot)
    # 2022-06 (pre_coverage) is dropped; 3/2 (incomplete, in-era) is gated
    # out; 3/1 (gated_complete) and 2022-10 (cost_only) survive.
    assert len(pool) == 2
    by_date = pool.set_index("run_date")

    # pre_coverage onboarding row must not reach the frame
    assert date(2022, 6, 1) not in by_date.index
    assert "pre_coverage" not in set(pool["data_regime"])

    # the featured, gated day
    row_2024 = by_date.loc[date(2024, 3, 1)]
    assert row_2024["cost"] == 100.0             # free-tier zero row excluded
    assert row_2024["job_seconds"] == 14400.0
    assert row_2024["data_regime"] == "featured_gated"
    assert row_2024["gate_state"] == "gated_complete"

    # the cost-only day: kept, but activity is null BY CONSTRUCTION, not zero
    row_2022 = by_date.loc[date(2022, 10, 1)]
    assert row_2022["cost"] == 70.0
    assert pd.isna(row_2022["job_seconds"])      # never imputed to 0
    assert row_2022["data_regime"] == "cost_only"
    assert row_2022["gate_state"] == "ungated"

    assert pool.attrs["orphan_report"]["both"] == 2


def test_cost_only_activity_is_null_never_zero(snapshot):
    # The load-bearing distinction: a pre-activity pool-day must not be
    # imputed to zero activity, which would be indistinguishable from a
    # featured-era pool that genuinely ran nothing.
    pool = T.build_pool_frame(snapshot)
    cost_only = pool[pool["data_regime"] == "cost_only"]
    assert len(cost_only) == 1
    assert cost_only["job_seconds"].isna().all()


def test_team_frame_keeps_null_distinct(snapshot):
    team = T.build_team_frame(snapshot)
    assert set(team["job_team"]) == {"Pillar1", "__NULL_TEAM__"}
    assert len(team) == 2
