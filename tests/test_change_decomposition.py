"""Tests for change_decomposition.py and explain.py.

The decomposition tests are constructed so the right answer is known in
advance: a pure price change must land entirely in price_effect, a pure
usage change entirely in usage_effect, a new meter entirely in scope, and
the identity must hold to the penny. The explanation tests assert the
TreeSHAP additivity identity and the end-to-end join."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catpipe import change_decomposition as CD
from catpipe import transform as T
from catpipe.assertions import DataQualityError
from catpipe.explain import explain_event, format_explanation, local_attribution
from catpipe.persistence import load_bundle, save_bundle
from tests import helpers_ops as H


def _period(rows):
    return pd.DataFrame(rows, columns=["meter", "pre_tax_cost", "usage_quantity"])


class TestBennetDecomposition:
    def test_pure_price_change_is_all_price_effect(self):
        a = _period([("D4s v3", 100.0, 500.0)])   # price 0.20
        b = _period([("D4s v3", 125.0, 500.0)])   # price 0.25, same usage
        d = CD.decompose(a, b)
        s = d.summary()
        assert s["price_effect"] == 25.0
        assert s["usage_effect"] == 0.0
        assert s["scope_effect"] == 0.0

    def test_pure_usage_change_is_all_usage_effect(self):
        a = _period([("D4s v3", 100.0, 500.0)])
        b = _period([("D4s v3", 120.0, 600.0)])   # price 0.20 both sides
        d = CD.decompose(a, b)
        s = d.summary()
        assert s["usage_effect"] == 20.0
        assert s["price_effect"] == 0.0

    def test_new_meter_is_scope(self):
        a = _period([("D4s v3", 100.0, 500.0)])
        b = _period([("D4s v3", 100.0, 500.0), ("E8s v5", 40.0, 100.0)])
        d = CD.decompose(a, b)
        s = d.summary()
        assert s["scope_effect"] == 40.0
        assert s["price_effect"] == 0.0
        row = d.items.set_index("meter").loc["E8s v5"]
        assert row["kind"] == "entering"

    def test_exiting_meter_is_negative_scope(self):
        a = _period([("D4s v3", 100.0, 500.0), ("E8s v5", 40.0, 100.0)])
        b = _period([("D4s v3", 100.0, 500.0)])
        assert CD.decompose(a, b).summary()["scope_effect"] == -40.0

    def test_mixed_change_is_exactly_additive(self):
        rng = np.random.default_rng(3)
        meters = [f"m{i}" for i in range(12)]
        a = _period([(m, float(rng.uniform(10, 200)), float(rng.uniform(50, 500)))
                     for m in meters[:10]])
        b = _period([(m, float(rng.uniform(10, 200)), float(rng.uniform(50, 500)))
                     for m in meters[2:]])
        d = CD.decompose(a, b)
        # The identity, unrounded: exact to the penny per item and in total
        per_item = (d.items["price_effect"] + d.items["usage_effect"]
                    + d.items["scope_effect"] + d.items["unpriced_effect"])
        assert (per_item - d.items["delta"]).abs().max() <= CD.PENNY
        assert abs(per_item.sum() - d.delta) <= CD.PENNY
        # The display summary rounds components independently, so allow a
        # penny per component against the rounded delta
        s = d.summary()
        recomposed = (s["price_effect"] + s["usage_effect"]
                      + s["scope_effect"] + s["unpriced_effect"])
        assert abs(recomposed - s["delta"]) <= 0.04

    def test_unpriced_cost_is_labelled_not_hidden(self):
        a = _period([("Support fee", 30.0, 0.0)])
        b = _period([("Support fee", 45.0, 0.0)])
        d = CD.decompose(a, b)
        s = d.summary()
        assert s["unpriced_effect"] == 15.0
        assert d.items["kind"].iloc[0] == "unpriced"

    def test_day_vs_baseline_scales_to_per_day(self):
        days = pd.date_range("2026-06-01", periods=29, freq="D")
        rows = []
        for d in days[:-1]:   # 28 baseline days at cost 100
            rows.append({"run_date": d, "pool_name": "pool_0",
                         "meter": "D4s v3", "pre_tax_cost": 100.0,
                         "usage_quantity": 500.0})
        rows.append({"run_date": days[-1], "pool_name": "pool_0",
                     "meter": "D4s v3", "pre_tax_cost": 150.0,
                     "usage_quantity": 750.0})  # same price, more usage
        raw = pd.DataFrame(rows)
        d = CD.day_vs_baseline(raw, days[-1], {"pool_name": "pool_0"})
        s = d.summary()
        assert s["cost_before"] == 100.0      # per-day baseline, not the sum
        assert s["delta"] == 50.0
        assert s["usage_effect"] == 50.0
        assert s["price_effect"] == 0.0

    def test_no_baseline_days_is_loud(self):
        raw = pd.DataFrame([{"run_date": pd.Timestamp("2026-06-01"),
                             "pool_name": "pool_0", "meter": "m",
                             "pre_tax_cost": 1.0, "usage_quantity": 1.0}])
        with pytest.raises(DataQualityError, match="no baseline days"):
            CD.day_vs_baseline(raw, "2026-06-01", {"pool_name": "pool_0"})


class TestTeamContribution:
    def test_deltas_sum_to_group_change_and_flag_the_mover(self):
        days = pd.date_range("2026-06-01", periods=15, freq="D")
        rows = []
        for d in days[:-1]:
            rows += [
                {"run_date": d, "pool_name": "pool_0", "job_team": "Risk",
                 "cost": 60.0},
                {"run_date": d, "pool_name": "pool_0", "job_team": "IPV",
                 "cost": 40.0},
            ]
        rows += [
            {"run_date": days[-1], "pool_name": "pool_0", "job_team": "Risk",
             "cost": 60.0},
            {"run_date": days[-1], "pool_name": "pool_0", "job_team": "IPV",
             "cost": 90.0},   # IPV alone moves
        ]
        tc = CD.team_contribution(pd.DataFrame(rows), days[-1],
                                  {"pool_name": "pool_0"}, baseline_days=14)
        assert abs(tc["delta"].sum() - 50.0) < 0.01
        assert tc.index[0] == "IPV"
        assert abs(tc.loc["IPV", "share_of_change"] - 1.0) < 0.01
        assert abs(tc.loc["Risk", "delta"]) < 0.01


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tables = H.make_tables()
    featured = H.featurize_pool(T.build_pool_frame(tables))
    boosters = H.fit_boosters(featured)
    card = H.make_card(featured)
    root = tmp_path_factory.mktemp("models")
    model = load_bundle(save_bundle(root / "v1", boosters, card))
    return model, featured, tables


class TestExplain:

    def test_attribution_sums_to_prediction(self, setup):
        model, featured, _ = setup
        att = local_attribution(model, featured.head(50))
        contrib_cols = [c for c in att.columns if c.startswith("contrib_")]
        total = att[contrib_cols].sum(axis=1) + att["baseline"]
        np.testing.assert_allclose(total, att["pred_mean"], rtol=0, atol=1e-6)

    def test_event_explanation_end_to_end(self, setup):
        model, featured, tables = setup
        row = featured.iloc[-1]
        group = {"subscription_id": row["subscription_id"],
                 "batch_account_name": row["batch_account_name"],
                 "pool_name": str(row["pool_name"])}
        e = explain_event(model, featured, tables["raw_cost"],
                          row["run_date"], group,
                          job_cost=tables["job_cost"])
        d = e.to_dict()
        assert d["prediction"]["pred_mean"] > 0
        assert len(d["model_drivers"]) > 0
        recomposed = (d["decomposition"]["price_effect"]
                      + d["decomposition"]["usage_effect"]
                      + d["decomposition"]["scope_effect"]
                      + d["decomposition"]["unpriced_effect"])
        assert abs(recomposed - d["decomposition"]["delta"]) <= 0.02
        assert d["team_contribution"] is not None

        text = format_explanation(e)
        assert "price" in text and "usage" in text
        assert "feature contributions" in text

    def test_ambiguous_alert_key_is_refused(self, setup):
        model, featured, tables = setup
        with pytest.raises(Exception, match="exactly one"):
            explain_event(model, featured, tables["raw_cost"],
                          featured["run_date"].iloc[0],
                          {"pool_name": "no_such_pool"})
