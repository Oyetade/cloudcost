"""Tests for persistence.py: round-trip identity, frozen levels, schema
drift, quantile discipline, conformal margins.

The round-trip identity test is the one that matters: predict on a fixed
frame, save, load, predict again, assert bit-identical. It is the only
guard against serving skew that can be written in advance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from catpipe.persistence import (
    LoadedModel, PersistenceError, apply_frozen_levels, assert_schema,
    load_bundle, save_bundle, schema_hash,
)
from tests import helpers_ops as H


@pytest.fixture(scope="module")
def featured():
    tables = H.make_tables()
    from catpipe import transform as T
    return H.featurize_pool(T.build_pool_frame(tables))


@pytest.fixture(scope="module")
def boosters(featured):
    return H.fit_boosters(featured)


@pytest.fixture(scope="module")
def card(featured):
    return H.make_card(featured)


def _direct_predictions(boosters, featured):
    """What the fitted model emits, computed without persistence."""
    X = featured[H.FEATURES]
    out = pd.DataFrame(index=featured.index)
    out["q05"] = np.sinh(boosters["q05"].predict(X))
    out["q50"] = np.sinh(boosters["q50"].predict(X))
    out["q95"] = np.sinh(boosters["q95"].predict(X))
    out["pred_mean"] = boosters["mean"].predict(X)
    q = ["q05", "q50", "q95"]
    out[q] = np.sort(out[q].to_numpy(), axis=1)
    return out.clip(lower=0.0)


class TestRoundTrip:
    def test_identity_across_save_load(self, tmp_path, featured, boosters, card):
        before = _direct_predictions(boosters, featured)
        loaded = load_bundle(save_bundle(tmp_path / "v1", boosters, card))
        after = loaded.predict(featured)
        for col in ("q05", "q50", "q95", "pred_mean"):
            np.testing.assert_array_equal(
                before[col].to_numpy(), after[col].to_numpy(),
                err_msg=f"{col} drifted across save/load: serving skew",
            )
        assert after.attrs["point_col"] == "pred_mean"

    def test_bundle_is_immutable(self, tmp_path, boosters, card):
        save_bundle(tmp_path / "v1", boosters, card)
        with pytest.raises(PersistenceError, match="immutable"):
            save_bundle(tmp_path / "v1", boosters, card)

    def test_card_must_declare_what_it_receives(self, tmp_path, boosters, card):
        partial = {k: v for k, v in boosters.items() if k != "mean"}
        with pytest.raises(PersistenceError, match="declares boosters"):
            save_bundle(tmp_path / "v2", partial, card)

    def test_hand_edited_card_is_rejected(self, tmp_path, boosters, card):
        path = save_bundle(tmp_path / "v1", boosters, card)
        card_file = path / "model_card.json"
        d = json.loads(card_file.read_text())
        d["feature_names"] = d["feature_names"][::-1]
        card_file.write_text(json.dumps(d))
        with pytest.raises(PersistenceError, match="schema hash"):
            load_bundle(path)

    def test_missing_booster_file_is_loud(self, tmp_path, boosters, card):
        path = save_bundle(tmp_path / "v1", boosters, card)
        (path / "booster_mean.txt").unlink()
        with pytest.raises(PersistenceError, match="booster mean"):
            load_bundle(path)


class TestFrozenLevels:
    def test_absent_pool_does_not_renumber_the_rest(
        self, tmp_path, featured, boosters, card
    ):
        """The classic silent failure: a slice missing one pool must not
        renumber the codes of the pools that remain."""
        loaded = load_bundle(save_bundle(tmp_path / "v1", boosters, card))
        full = loaded.predict(featured)
        mask = featured["pool_name"] != "pool_0"
        subset = loaded.predict(featured.loc[mask])
        np.testing.assert_array_equal(
            full.loc[mask, "q50"].to_numpy(), subset["q50"].to_numpy(),
            err_msg="dropping a pool changed other pools' predictions: "
                    "levels were re-derived from the data",
        )

    def test_unseen_level_maps_to_missing_and_is_counted(self, featured, card):
        novel = featured.head(20).copy()
        novel["pool_name"] = novel["pool_name"].astype("string")
        novel.loc[novel.index[:2], "pool_name"] = "pool_new"
        recoded, counts = apply_frozen_levels(novel, card)
        assert counts["pool_name"] == 2
        assert recoded["pool_name"].isna().sum() == 2

    def test_unseen_flood_raises(self, featured, card):
        novel = featured.head(20).copy()
        novel["pool_name"] = "pool_new"
        with pytest.raises(PersistenceError, match="scoping event"):
            apply_frozen_levels(novel, card)


class TestSchema:
    def test_dtype_drift_raises(self, featured, card):
        drifted = featured.copy()
        drifted["dow"] = drifted["dow"].astype("float64")
        with pytest.raises(PersistenceError, match="dtype drift"):
            assert_schema(drifted, card)

    def test_missing_feature_raises(self, featured, card):
        with pytest.raises(PersistenceError, match="missing features"):
            assert_schema(featured.drop(columns=["cost_lag7"]), card)

    def test_hash_is_order_sensitive(self):
        dt = {"a": "float64", "b": "int64"}
        assert schema_hash(["a", "b"], dt) != schema_hash(["b", "a"], dt)

    def test_novel_null_in_never_null_feature_raises(
        self, tmp_path, featured, boosters, card
    ):
        """The schema guard fires first on a dtype change, so exercise the
        null guard directly: a nullable-int feature trained without nulls
        that goes null at inference must refuse, not let LightGBM consume
        the NaN and predict something."""
        import dataclasses
        loaded = load_bundle(save_bundle(tmp_path / "v1", boosters, card))
        nullable_card = dataclasses.replace(
            loaded.card,
            feature_dtypes={**loaded.card.feature_dtypes, "dow": "Int64"},
        )
        holey = LoadedModel(card=nullable_card, boosters=loaded.boosters)
        holed = featured.copy()
        holed["dow"] = holed["dow"].astype("Int64")
        holed.loc[holed.index[0], "dow"] = pd.NA
        with pytest.raises(PersistenceError, match="null at inference"):
            holey._check_novel_nulls(holed)


class TestPredictionDiscipline:
    def test_quantiles_never_cross_and_floor_at_zero(
        self, tmp_path, featured, boosters, card
    ):
        loaded = load_bundle(save_bundle(tmp_path / "v1", boosters, card))
        preds = loaded.predict(featured)
        assert (preds["q05"] <= preds["q50"]).all()
        assert (preds["q50"] <= preds["q95"]).all()
        assert (preds[["q05", "q50", "q95", "pred_mean"]] >= 0).all().all()

    def test_conformal_margins_widen_with_pooled_fallback(
        self, tmp_path, featured, boosters, card
    ):
        margins = pd.DataFrame({
            "pool_name": ["pool_1", None],
            "lower_margin": [5.0, 2.0],
            "upper_margin": [7.0, 3.0],
        })
        plain = load_bundle(save_bundle(tmp_path / "plain", boosters, card))
        conf = LoadedModel(card=plain.card, boosters=plain.boosters,
                           conformal_margins=margins)
        raw = plain.predict(featured)
        cal = conf.predict(featured)
        is_p1 = (featured["pool_name"] == "pool_1").to_numpy()
        np.testing.assert_allclose(cal.loc[is_p1, "q95"],
                                   raw.loc[is_p1, "q95"] + 7.0)
        np.testing.assert_allclose(cal.loc[~is_p1, "q95"],
                                   raw.loc[~is_p1, "q95"] + 3.0)
        assert (cal["q05"] <= raw["q05"]).all()
        assert (cal["q05"] >= 0).all()
