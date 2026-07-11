"""Frames: the three model-ready training frames and the new invariants.

Synthetic snapshot design: two gated featured days (2024-03-01/02) with one
pool, a non-pool slice containing both VM-compute and platform lines, an
Unknown-heavy attribution day, and a NULL-team job — the smallest data that
exercises every join, the partition identity, the segmentation rule, the
unknown_pct computation, and the NULL-vs-Unknown distinction end to end.
"""

from datetime import date

import pandas as pd
import pytest

from catpipe import assertions as A
from catpipe import frames as FR
from catpipe import transform as T


def _ts(x):
    return pd.Timestamp(x, tz="UTC")


def _rs(rows):
    out = []
    for rd, sub in rows:
        for rt in ("Cost", "Usage", "Attribution"):
            out.append(dict(run_date=rd, subscription_id=sub, run_type=rt,
                            status="Complete", update_time=_ts("2024-03-05")))
    return pd.DataFrame(out)


@pytest.fixture
def snapshot():
    d1, d2 = date(2024, 3, 1), date(2024, 3, 2)
    raw_cost = pd.DataFrame([
        # pool branch
        dict(run_date=d1, subscription_id="s1", resource_group_name="rg",
             resource_type="microsoft.compute/virtualmachinescalesets",
             service_name="Virtual Machines", meter_category="Virtual Machines",
             meter="D64 Spot", batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=100.0, usage_quantity=10.0,
             update_time=_ts("2024-03-03")),
        dict(run_date=d2, subscription_id="s1", resource_group_name="rg",
             resource_type="microsoft.compute/virtualmachinescalesets",
             service_name="Virtual Machines", meter_category="Virtual Machines",
             meter="D64 Spot", batch_account_name="a", pool_name="eodpool",
             pre_tax_cost=110.0, usage_quantity=11.0,
             update_time=_ts("2024-03-04")),
        # non-pool branch: a VM line and two platform lines
        dict(run_date=d1, subscription_id="s1", resource_group_name="rg",
             resource_type="microsoft.compute/virtualmachines",
             service_name="Virtual Machines", meter_category="Virtual Machines",
             meter="D8 v5", batch_account_name=None, pool_name=None,
             pre_tax_cost=40.0, usage_quantity=8.0,
             update_time=_ts("2024-03-03")),
        dict(run_date=d1, subscription_id="s1", resource_group_name="rg",
             resource_type="microsoft.storage/storageaccounts",
             service_name="Storage", meter_category="Storage",
             meter="Read Ops", batch_account_name=None, pool_name=None,
             pre_tax_cost=5.0, usage_quantity=1000.0,
             update_time=_ts("2024-03-03")),
        dict(run_date=d2, subscription_id="s1", resource_group_name="rg",
             resource_type="microsoft.storage/storageaccounts",
             service_name="Storage", meter_category="Storage",
             meter="Read Ops", batch_account_name=None, pool_name=None,
             pre_tax_cost=6.0, usage_quantity=1200.0,
             update_time=_ts("2024-03-04")),
    ])
    job_usage = pd.DataFrame([
        dict(run_date=d1, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="BT_j1",
             start_time=_ts("2024-03-01 09:00"),
             end_time=_ts("2024-03-01 11:00"),
             job_seconds=7200.0, task_count=3),
        dict(run_date=d1, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="RS_j2",
             start_time=_ts("2024-03-01 10:00"),
             end_time=_ts("2024-03-01 12:00"),
             job_seconds=3600.0, task_count=2),
        dict(run_date=d2, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="BT_j3",
             start_time=_ts("2024-03-02 09:00"),
             end_time=_ts("2024-03-02 10:00"),
             job_seconds=3600.0, task_count=1),
    ])
    job_cost = pd.DataFrame([
        dict(run_date=d1, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="BT_j1", job_name="BT",
             job_category="BT", job_ownership="Risk", job_team="Pillar1",
             cost=60.0),
        dict(run_date=d1, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="RS_j2", job_name="RS",
             job_category="Reg_Stress", job_ownership="Risk",
             job_team=None, cost=40.0),                    # NULL team
        dict(run_date=d2, subscription_id="s1", batch_account_name="a",
             pool_name="eodpool", job_id="BT_j3", job_name="BT",
             job_category="BT", job_ownership="Risk", job_team="Unknown",
             cost=110.0),                                  # Unknown-heavy day
    ])
    environment_config = pd.DataFrame([
        dict(subscription_id="s1", subscription_name="ba-fr-at1565-neu-prod",
             environment_tier="PROD", environment_sub_tier="Primary"),
    ])
    return dict(raw_cost=raw_cost, job_usage=job_usage, job_cost=job_cost,
                run_status=_rs([(d1, "s1"), (d2, "s1")]),
                environment_config=environment_config)


# ---------------------------------------------------------------------------
# new assertions
# ---------------------------------------------------------------------------

class TestPartitionIdentity:
    def test_clean_partition_passes(self):
        A.assert_partition_identity({"pool": 60.0, "non_pool": 40.0}, 100.0,
                                    "test")

    def test_dropped_branch_raises(self):
        with pytest.raises(A.DataQualityError, match="partition"):
            A.assert_partition_identity({"pool": 60.0, "non_pool": 20.0},
                                        100.0, "test")


class TestOneWriteInvariant:
    def _raw(self, times):
        return pd.DataFrame({
            "run_date": date(2024, 3, 1),
            "subscription_id": "s1",
            "update_time": times,
        })

    def test_single_write_passes(self):
        A.assert_one_write_per_slice(
            self._raw([_ts("2024-03-03"), _ts("2024-03-03")]))

    def test_rewrite_trips_the_tripwire(self):
        # Q1's consequence chain rests on this: the day the upsert fires,
        # the pipeline must say so loudly.
        with pytest.raises(A.DataQualityError, match="upsert"):
            A.assert_one_write_per_slice(
                self._raw([_ts("2024-03-03"), _ts("2024-03-09")]))


class TestDuplicateRateReport:
    def test_reports_without_raising(self):
        df = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
        rep = A.report_duplicate_rate(df, ["k"], "t")
        assert rep["duplicate_rows"] == 2
        assert rep["duplicate_rate"] == pytest.approx(2 / 3)
        assert rep["examples"]


# ---------------------------------------------------------------------------
# enrichment
# ---------------------------------------------------------------------------

class TestEnrichment:
    def test_tier_and_region_join_on_subscription_id(self, snapshot):
        f = pd.DataFrame({"subscription_id": ["s1", "s1"]})
        out = FR.enrich_environment(f, snapshot["environment_config"])
        assert (out["environment_tier"] == "PROD").all()
        assert (out["region"] == "northeurope").all()

    def test_duplicate_config_cannot_multiply_the_frame(self, snapshot):
        cfg = pd.concat([snapshot["environment_config"]] * 2,
                        ignore_index=True)
        f = pd.DataFrame({"subscription_id": ["s1"]})
        out = FR.enrich_environment(f, cfg)  # deduped, then identity holds
        assert len(out) == 1

    def test_weu_maps_to_westeurope(self):
        cfg = pd.DataFrame([dict(subscription_id="s2",
                                 subscription_name="ba-fr-x-weu-dev",
                                 environment_tier="DEV",
                                 environment_sub_tier="Playground")])
        out = FR.enrich_environment(
            pd.DataFrame({"subscription_id": ["s2"]}), cfg)
        assert out["region"].iloc[0] == "westeurope"


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_vm_resource_type_is_vm_compute(self):
        df = pd.DataFrame({
            "resource_type": ["microsoft.compute/virtualmachines",
                              "microsoft.storage/storageaccounts"],
            "meter_category": ["Virtual Machines", "Storage"],
            "service_name": ["Virtual Machines", "Storage"],
        })
        assert FR.classify_segment(df).tolist() == ["vm_compute", "platform"]

    def test_licence_line_stays_with_the_vm_estate(self):
        # the December-2024 reclassification: the licence split out of the
        # VM meter must not migrate the cost into the platform segment
        df = pd.DataFrame({
            "resource_type": [None],
            "meter_category": ["Virtual Machines Licences"],
            "service_name": ["Licences"],
        })
        assert FR.classify_segment(df).iloc[0] == "vm_compute"

    def test_scale_sets_prefix_matches(self):
        df = pd.DataFrame({
            "resource_type": ["microsoft.compute/virtualmachinescalesets"],
            "meter_category": ["Other"], "service_name": ["Other"],
        })
        assert FR.classify_segment(df).iloc[0] == "vm_compute"


# ---------------------------------------------------------------------------
# frame 1a
# ---------------------------------------------------------------------------

class TestFrame1a:
    def test_shape_features_and_no_leak(self, snapshot):
        f = FR.build_frame_1a(snapshot)
        assert len(f) == 2  # two gated pool-days
        feats = f.attrs["feature_cols"]
        assert "cost" not in feats and "pre_tax_cost" not in feats
        for expected in ["cost_lag1", "cost_lag7", "cost_roll28",
                         "job_seconds_lag1", "task_count_lag1",
                         "peak_concurrency_lag1", "n_jobs_lag1",
                         "dow", "d_to_quarter_end", "pool_name",
                         "environment_tier"]:
            assert expected in feats, expected

    def test_concurrency_and_mix_wired_in(self, snapshot):
        f = FR.build_frame_1a(snapshot)
        d1 = f[f["run_date"] == date(2024, 3, 1)].iloc[0]
        # jobs 09-11 and 10-12 overlap 10-11: peak 2
        assert d1["peak_concurrency"] == 2
        assert d1["n_jobs"] == 2
        # Reg_Stress share of day-1 job_seconds: 3600/10800
        assert d1["share_Reg_Stress"] == pytest.approx(1 / 3)

    def test_day2_lag_sees_day1(self, snapshot):
        f = FR.build_frame_1a(snapshot)
        d2 = f[f["run_date"] == date(2024, 3, 2)].iloc[0]
        assert d2["cost_lag1"] == 100.0
        assert d2["job_seconds_lag1"] == 10800.0

    def test_enrichment_present(self, snapshot):
        f = FR.build_frame_1a(snapshot)
        assert (f["environment_tier"] == "PROD").all()
        assert (f["region"] == "northeurope").all()


# ---------------------------------------------------------------------------
# frame 1b
# ---------------------------------------------------------------------------

class TestFrame1b:
    def test_partition_identity_holds_by_construction(self, snapshot):
        f = FR.build_frame_1b(snapshot)  # would raise if the split leaked
        assert len(f)

    def test_segments_aggregate_correctly(self, snapshot):
        f = FR.build_frame_1b(snapshot)
        d1 = f[f["run_date"] == date(2024, 3, 1)].set_index("segment")
        assert d1.loc["vm_compute", "cost"] == 40.0
        assert d1.loc["platform", "cost"] == 5.0
        assert d1.loc["platform", "line_count"] == 1

    def test_spine_carries_residual_on_vmless_day(self, snapshot):
        # day 2 has no VM line: the spine pads vm_compute to an explicit,
        # gate-verified zero rather than dropping the day (the 62.43 lesson
        # applied to the frame itself)
        f = FR.build_frame_1b(snapshot)
        d2 = f[(f["run_date"] == date(2024, 3, 2))
               & (f["segment"] == "vm_compute")]
        assert len(d2) == 1
        assert d2["cost"].iloc[0] == 0.0
        assert bool(d2["padded"].iloc[0])

    def test_double_load_raises(self, snapshot):
        tables = dict(snapshot)
        rc = tables["raw_cost"]
        dup = rc[rc["pool_name"].isna()].head(1)
        tables["raw_cost"] = pd.concat([rc, dup], ignore_index=True)
        with pytest.raises(A.DataQualityError, match="duplicate"):
            FR.build_frame_1b(tables)

    def test_candidate_key_duplicates_reported_not_raised(self, snapshot):
        tables = dict(snapshot)
        rc = tables["raw_cost"].copy()
        extra = rc[rc["pool_name"].isna()].head(1).copy()
        extra["usage_quantity"] = 999.0  # same business key, different row:
        extra["pre_tax_cost"] = 1.0      # two resources, one RG+meter (Q7)
        tables["raw_cost"] = pd.concat([rc, extra], ignore_index=True)
        f = FR.build_frame_1b(tables)    # must NOT raise
        assert f.attrs["grain_report"]["duplicate_rows"] == 2

    def test_post_glide_and_training_slice(self, snapshot):
        tables = dict(snapshot)
        rc = tables["raw_cost"].copy()
        late = rc[rc["pool_name"].isna()].head(1).copy()
        late["run_date"] = date(2025, 3, 1)
        tables["raw_cost"] = pd.concat([rc, late], ignore_index=True)
        tables["run_status"] = pd.concat(
            [tables["run_status"], _rs([(date(2025, 3, 1), "s1")])],
            ignore_index=True)
        f = FR.build_frame_1b(tables)
        assert not f[f["run_date"] == date(2024, 3, 1)]["post_glide"].any()
        sl = FR.training_slice_1b(f)
        assert set(sl["run_date"]) == {date(2025, 3, 1)}

    def test_feature_list_clean(self, snapshot):
        f = FR.build_frame_1b(snapshot)
        feats = f.attrs["feature_cols"]
        assert "cost" not in feats and "line_count" not in feats
        for expected in ["cost_lag1", "cost_lag7", "cost_roll28",
                         "line_count_lag1", "price_drift_lag1",
                         "dow", "d_to_month_end", "segment",
                         "subscription_id"]:
            assert expected in feats, expected


# ---------------------------------------------------------------------------
# frame 2
# ---------------------------------------------------------------------------

class TestFrame2:
    def test_null_team_distinct_from_unknown(self, snapshot):
        f = FR.build_frame_2(snapshot)
        teams = set(f["job_team"])
        assert "__NULL_TEAM__" in teams and "Unknown" in teams
        d1 = f[f["run_date"] == date(2024, 3, 1)].set_index("job_team")
        assert d1.loc["__NULL_TEAM__", "cost"] == 40.0
        assert d1.loc["Unknown", "cost"] == 0.0

    def test_additivity_sum_over_teams_equals_attributed_total(self,
                                                               snapshot):
        f = FR.build_frame_2(snapshot)
        d1 = f[f["run_date"] == date(2024, 3, 1)]["cost"].sum()
        d2 = f[f["run_date"] == date(2024, 3, 2)]["cost"].sum()
        assert d1 == pytest.approx(100.0)
        assert d2 == pytest.approx(110.0)

    def test_unknown_pct_computed_per_day(self, snapshot):
        f = FR.build_frame_2(snapshot)
        d1 = f[f["run_date"] == date(2024, 3, 1)]["unknown_pct"].iloc[0]
        d2 = f[f["run_date"] == date(2024, 3, 2)]["unknown_pct"].iloc[0]
        assert d1 == pytest.approx(0.4)   # NULL team counts (40/100)
        assert d2 == pytest.approx(1.0)   # the Unknown-heavy day

    def test_filter_unknown_drops_pathological_days(self, snapshot):
        f = FR.build_frame_2(snapshot)
        kept = FR.filter_unknown(f, max_unknown_pct=0.5)
        assert set(kept["run_date"]) == {date(2024, 3, 1)}

    def test_team_activity_through_five_key_join(self, snapshot):
        f = FR.build_frame_2(snapshot)
        d1 = f[f["run_date"] == date(2024, 3, 1)].set_index("job_team")
        assert d1.loc["Pillar1", "job_seconds"] == 7200.0
        assert d1.loc["__NULL_TEAM__", "job_seconds"] == 3600.0
        assert d1.loc["Pillar1", "n_jobs"] == 1
        # mix: Pillar1's seconds are all BT
        assert d1.loc["Pillar1", "share_bt"] == pytest.approx(1.0)

    def test_absent_team_day_is_true_zero_activity(self, snapshot):
        f = FR.build_frame_2(snapshot)
        d2 = f[f["run_date"] == date(2024, 3, 2)].set_index("job_team")
        assert d2.loc["Pillar1", "cost"] == 0.0
        assert d2.loc["Pillar1", "job_seconds"] == 0.0

    def test_lags_and_feature_list(self, snapshot):
        f = FR.build_frame_2(snapshot)
        d2 = f[f["run_date"] == date(2024, 3, 2)].set_index("job_team")
        assert d2.loc["Pillar1", "cost_lag1"] == 60.0
        feats = f.attrs["feature_cols"]
        assert "cost" not in feats and "unknown_pct" not in feats
        for expected in ["cost_lag1", "cost_lag7", "cost_roll28",
                         "job_seconds_lag1", "n_jobs_lag1", "dow",
                         "d_to_quarter_end", "job_team"]:
            assert expected in feats, expected
