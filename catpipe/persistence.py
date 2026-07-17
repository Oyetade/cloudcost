"""
persistence.py  --  save a trained forecaster down; reload it for inference.

A trained forecaster is not one object: it is several LightGBM boosters
(q05/q50/q95, and the mean booster the monthly-bias fix added), a frozen
categorical level map, a transform choice, a feature list in a fixed order,
and conformal margins. Pickling a class instance couples the artefact to the
exact class definition at save time; that promise breaks within a quarter.
So the bundle is a directory of stable formats, and a model card records
everything predict() needs that the boosters do not carry:

    models/pool/v2026-08-01/
        booster_q05.txt ... booster_mean.txt   (Booster.save_model text)
        conformal_margins.parquet              (optional)
        model_card.json

Rules enforced here, in the same spirit as assertions.py: the referential
guarantees we would like (training frame and inference frame agree) are
asserted, loudly, because nothing else will prove them.

  - Categorical levels are frozen at save and reapplied at load, never
    re-derived from inference data: pandas assigns category codes from
    whatever levels happen to be present, so one absent pool renumbers
    every code and the model degrades without erroring.
  - The feature schema (names, order, dtypes) is hashed at save and
    asserted at load and at predict. Drift raises; it never warns.
  - Unseen levels and never-null features going null are counted and
    raised, not absorbed.
  - A bundle directory is immutable. A retrain writes a new version.

This module is model-agnostic on purpose: it takes a dict of boosters plus
a card, so QuantileGBM.save()/load() (in the local tree's models.py) are a
few lines of glue over save_bundle()/load_bundle(). The local frames declare
target/features/categoricals/group keys in attrs; those slot straight into
the ModelCard fields.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class PersistenceError(RuntimeError):
    """Raised when a bundle cannot be saved, loaded, or safely applied."""


# --- schema hashing --------------------------------------------------------

def schema_hash(feature_names: Sequence[str], dtypes: Mapping[str, str]) -> str:
    """Hash of feature names in order plus their dtypes. Declared once,
    asserted everywhere, the same discipline as a shared grain constant:
    two hand-maintained copies of a feature list is how frames drift.
    """
    payload = json.dumps(
        [[name, str(dtypes[name])] for name in feature_names],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_dtypes(frame: pd.DataFrame, feature_names: Sequence[str]) -> dict[str, str]:
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise PersistenceError(f"features absent from frame: {missing}")
    return {c: str(frame[c].dtype) for c in feature_names}


# --- the model card ---------------------------------------------------------

@dataclass
class BoosterSpec:
    """One booster in the bundle.

    transform names the target scale the booster was trained on. The
    quantile boosters train on the transformed target (quantiles are
    invariant under monotone transforms, so predictions invert cleanly);
    the mean booster trains on the untransformed target so summed daily
    means give an unbiased monthly total, and is never inverted.
    """

    name: str                # "q05" | "q50" | "q95" | "mean"
    transform: str           # "asinh" | "none"
    is_quantile: bool
    quantile: float | None = None


@dataclass
class ModelCard:
    frame: str                                # "pool" | "team" (local: "frame_1a"...)
    target: str                               # "cost" in this tree
    feature_names: list[str]                  # exact training order
    feature_dtypes: dict[str, str]
    categorical_features: list[str]
    categorical_levels: dict[str, list[str]]  # frozen at fit
    group_keys: list[str]                     # e.g. sub/batch_account/pool
    boosters: list[BoosterSpec]
    point_col: str                            # "pred_mean" for the GBMs
    train_origin: str
    train_end: str
    snapshot: str                             # snapshot dir stem trained from
    horizon_days: int
    filter_tiers: bool = False                # local tree's load_snapshot flag
    excluded_tiers: list[str] = field(default_factory=list)
    zero_fill_prefixes: list[str] = field(default_factory=list)
    # Prefixes (e.g. "share_") whose card-listed features may legitimately be
    # ABSENT from an inference frame: the job-mix pivot only creates a column
    # per category observed in the window, so a quiet fortnight with no Audit
    # jobs has no share_Audit_lag1 column at all. For these, and only these,
    # a missing feature is filled with 0.0 (the true value of an unobserved
    # category's share) rather than refused. Anything else missing still
    # raises. Novel pivot columns in the frame that the card does not list
    # are the mirror case (a NEW category appeared) and are counted for the
    # manifest, never silently absorbed.
    schema: str = ""                          # filled by save_bundle
    created_at: str = ""
    git_sha: str = ""
    lightgbm_version: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ModelCard":
        d = json.loads(text)
        d["boosters"] = [BoosterSpec(**b) for b in d["boosters"]]
        return cls(**d)


def freeze_levels(
    frame: pd.DataFrame, categorical_features: Sequence[str]
) -> dict[str, list[str]]:
    """Capture categorical levels at fit time. If QuantileGBM already holds
    its frozen levels (it freezes at fit so an unseen pool maps to missing),
    pass those through instead; this exists for the standalone case.
    """
    levels: dict[str, list[str]] = {}
    for col in categorical_features:
        s = frame[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            levels[col] = [str(v) for v in s.cat.categories]
        else:
            levels[col] = sorted(str(v) for v in s.dropna().unique())
    return levels


def apply_frozen_levels(
    frame: pd.DataFrame,
    card: ModelCard,
    *,
    max_unseen_frac: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recode categoricals from the card's frozen levels, never from the data.

    Unseen levels become NaN (LightGBM's missing), the fit-time behaviour,
    but they are counted: a day where a large share of rows carry unseen
    pools is a scoping event, not a forecast. Above max_unseen_frac on any
    categorical this raises. Returns (recoded frame, per-column unseen row
    counts); the run manifest records the counts even when zero.
    """
    out = frame.copy()
    unseen_counts: dict[str, int] = {}
    for col in card.categorical_features:
        allowed = card.categorical_levels[col]
        as_str = out[col].astype("string")
        unseen_mask = as_str.notna() & ~as_str.isin(allowed)
        n_unseen = int(unseen_mask.sum())
        unseen_counts[col] = n_unseen
        if len(out) and n_unseen / len(out) > max_unseen_frac:
            raise PersistenceError(
                f"{col}: {n_unseen} of {len(out)} rows "
                f"({n_unseen / len(out):.1%}) carry levels unseen at training, "
                f"above the {max_unseen_frac:.0%} limit. This is a scoping "
                "event; investigate before scoring."
            )
        out[col] = pd.Categorical(as_str.where(~unseen_mask), categories=allowed)
    return out, unseen_counts


_LAGGED_FORM = re.compile(r".*_(lag|roll(ing)?)_?\d+$")


def assert_no_unlagged_features(
    feature_names: Sequence[str],
    frame_columns: Sequence[str],
    target: str,
) -> None:
    """The charter's leakage rule, enforced against the CARD rather than the
    frame: no same-day cost or activity feature enters any model.

    The frame legitimately carries unlagged columns (job_seconds,
    peak_concurrency, the share_* family, price_drift): they are the raw
    material the lags are built from. The leak is one of them appearing in
    feature_names. assertions.assert_no_same_day_cost only catches literal
    cost columns; this catches the activity side too, by construction: a
    feature is an unlagged twin if the frame also carries a _lag{n} version
    of it, because the factory only lags columns that are same-day
    observations. Calendar and static columns have no lagged twins and pass
    untouched. Call at card-build time, before save_bundle.
    """
    cols = set(frame_columns)
    offenders = []
    for f in feature_names:
        if f == target or f in ("cost", "pre_tax_cost"):
            offenders.append(f)
            continue
        if _LAGGED_FORM.match(f):
            continue  # already a lagged/rolled form
        if any(c != f and c.startswith(f + "_lag") for c in cols):
            offenders.append(f)
    if offenders:
        raise PersistenceError(
            f"unlagged same-day features in the card: {offenders}. These are "
            "the target or its same-day correlates arriving through a side "
            "door; only their _lag/_roll forms may enter feature_names."
        )


def reindex_pivot_features(
    frame: pd.DataFrame, card: ModelCard
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Reconcile pivot-created feature columns between training and inference.

    Two directions, both driven by card.zero_fill_prefixes:
      - a card feature absent from the frame (category not observed in the
        window): added as 0.0, its true value, and reported;
      - a frame column matching a prefix that the card does NOT list (a NEW
        category appeared since training): reported for the manifest. It is
        not fed to the model regardless (X selects card features only), but
        a new job category is a scoping signal the run must surface, the
        column-space mirror of an unseen pool level.
    """
    added: list[str] = []
    novel: list[str] = []
    if not card.zero_fill_prefixes:
        return frame, {"zero_filled": added, "novel_pivot_columns": novel}
    out = frame
    prefixes = tuple(card.zero_fill_prefixes)
    for f in card.feature_names:
        if f.startswith(prefixes) and f not in out.columns:
            if out is frame:
                out = frame.copy()
            out[f] = 0.0
            added.append(f)
    card_set = set(card.feature_names)
    for c in out.columns:
        if c.startswith(prefixes) and _LAGGED_FORM.match(c) and c not in card_set:
            novel.append(c)
    return out, {"zero_filled": added, "novel_pivot_columns": sorted(novel)}


def assert_schema(frame: pd.DataFrame, card: ModelCard) -> None:
    """Raise unless the frame carries the card's features with matching
    dtypes. Categoricals compare on kind only, because apply_frozen_levels
    legitimately changes the level set.
    """
    missing = [c for c in card.feature_names if c not in frame.columns]
    if missing:
        raise PersistenceError(f"inference frame missing features: {missing}")
    mismatched = []
    for col in card.feature_names:
        want = card.feature_dtypes[col]
        got = str(frame[col].dtype)
        if want.startswith("category") and got.startswith("category"):
            continue
        if want != got:
            mismatched.append((col, want, got))
    if mismatched:
        raise PersistenceError(
            "dtype drift between training and inference frames: "
            + "; ".join(f"{c}: trained {w}, got {g}" for c, w, g in mismatched)
        )


# --- save / load -------------------------------------------------------------

def save_bundle(
    path: str | Path,
    boosters: Mapping[str, Any],          # name -> lgb.Booster
    card: ModelCard,
    conformal_margins: pd.DataFrame | None = None,
) -> Path:
    """Write the artefact directory. Refuses to overwrite: a model version
    is immutable, and a retrain writes a new directory."""
    path = Path(path)
    if path.exists():
        raise PersistenceError(
            f"{path} already exists. Bundles are immutable; save a new version."
        )
    declared = {b.name for b in card.boosters}
    if declared != set(boosters):
        raise PersistenceError(
            f"card declares boosters {sorted(declared)} but received "
            f"{sorted(boosters)}"
        )
    card.schema = schema_hash(card.feature_names, card.feature_dtypes)
    path.mkdir(parents=True)
    for name, booster in boosters.items():
        booster.save_model(str(path / f"booster_{name}.txt"))
    if conformal_margins is not None:
        conformal_margins.to_parquet(path / "conformal_margins.parquet")
    (path / "model_card.json").write_text(card.to_json())
    return path


def load_bundle(path: str | Path) -> "LoadedModel":
    import lightgbm as lgb

    path = Path(path)
    card_file = path / "model_card.json"
    if not card_file.exists():
        raise PersistenceError(f"no model_card.json under {path}")
    card = ModelCard.from_json(card_file.read_text())
    expected = schema_hash(card.feature_names, card.feature_dtypes)
    if card.schema != expected:
        raise PersistenceError(
            "model card schema hash does not match its own feature list; "
            "the card has been edited by hand or corrupted"
        )
    boosters = {}
    for spec in card.boosters:
        f = path / f"booster_{spec.name}.txt"
        if not f.exists():
            raise PersistenceError(
                f"card declares booster {spec.name} but {f.name} is absent"
            )
        boosters[spec.name] = lgb.Booster(model_file=str(f))
    margins_file = path / "conformal_margins.parquet"
    margins = pd.read_parquet(margins_file) if margins_file.exists() else None
    return LoadedModel(card=card, boosters=boosters, conformal_margins=margins)


# --- the reloaded model -------------------------------------------------------

_INVERSES = {
    "asinh": np.sinh,
    "none": lambda x: x,
}


@dataclass
class LoadedModel:
    """A reloaded bundle whose predict() must be bit-identical to the fitted
    model's. tests/test_persistence.py::test_round_trip_identity is the
    guard; it is the only defence against serving skew that can be written
    in advance."""

    card: ModelCard
    boosters: dict[str, Any]
    conformal_margins: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a frame that came through the training builders.

        Applies frozen levels, asserts schema, checks never-null features,
        predicts every booster, inverts transforms, sorts quantiles per row
        so intervals never cross, floors at zero, and applies conformal
        margins where present. Output carries point_col and the unseen-level
        counts in attrs for the run manifest.
        """
        X, recoded, unseen, pivot_report = self.design_matrix(frame)
        out = pd.DataFrame(index=frame.index)
        for spec in self.card.boosters:
            raw = self.boosters[spec.name].predict(X)
            inverse = _INVERSES.get(spec.transform)
            if inverse is None:
                raise PersistenceError(f"unknown transform {spec.transform!r}")
            col = "pred_mean" if spec.name == "mean" else spec.name
            out[col] = inverse(raw)

        qcols = [c for c in ("q05", "q50", "q95") if c in out.columns]
        if len(qcols) == 3:
            out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
        num = out.select_dtypes("number").columns
        out[num] = out[num].clip(lower=0.0)

        if self.conformal_margins is not None:
            out = self._apply_margins(out, recoded)

        out.attrs["unseen_level_counts"] = unseen
        out.attrs["pivot_report"] = pivot_report
        out.attrs["point_col"] = self.card.point_col
        return out

    def design_matrix(self, frame: pd.DataFrame):
        """The exact matrix predict() scores: pivot columns reconciled,
        frozen levels applied, schema asserted, never-null checked, features
        selected in card order. Public so explanation tooling (per-alert
        pred_contrib) is guaranteed to attribute the same matrix the model
        scored, never a re-derived one."""
        filled, pivot_report = reindex_pivot_features(frame, self.card)
        recoded, unseen = apply_frozen_levels(filled, self.card)
        assert_schema(recoded, self.card)
        self._check_novel_nulls(recoded)
        X = recoded.loc[:, self.card.feature_names]
        return X, recoded, unseen, pivot_report

    def _check_novel_nulls(self, frame: pd.DataFrame) -> None:
        """A lag column null at inference means the history window did not
        reach, or a join broke. LightGBM would consume the NaN and predict
        something; raise instead. Never-null is inferred from dtype (int and
        bool features cannot legitimately go NaN; float lags can carry the
        by-construction nulls the regime labels explain). If the card gains
        an explicit never_null list profiled from the training frame, prefer
        it here.
        """
        offenders = []
        for col in self.card.feature_names:
            want = self.card.feature_dtypes[col].lower()
            if want.startswith(("int", "uint", "bool")) and frame[col].isna().any():
                offenders.append(col)
        if offenders:
            raise PersistenceError(
                "features null at inference that were never null in training: "
                f"{offenders}. Likely a short history window or a broken join."
            )

    def _apply_margins(self, preds: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-group conformal margins with pooled fallback, additive on the
        reported scale: q05 - lower (floored at zero), q95 + upper.

        Margin table format: the group columns, lower_margin, upper_margin,
        and one all-null group row as the pooled fallback. If calibrate.py's
        native structure is persisted instead, swap this for its
        apply_margins; the semantics here match its per-group-with-pooled-
        fallback behaviour.
        """
        m = self.conformal_margins
        group_cols = [c for c in m.columns if c not in ("lower_margin", "upper_margin")]
        pooled_mask = (
            m[group_cols].isna().all(axis=1) if group_cols
            else pd.Series(True, index=m.index)
        )
        pooled = m.loc[pooled_mask]
        specific = m.loc[~pooled_mask]

        out = preds.copy()
        if group_cols and not specific.empty:
            keyed = (
                frame[group_cols].astype("string")
                .merge(
                    specific.assign(
                        **{c: specific[c].astype("string") for c in group_cols}
                    ),
                    on=group_cols, how="left",
                )
            )
            lower = keyed["lower_margin"].to_numpy()
            upper = keyed["upper_margin"].to_numpy()
        else:
            lower = np.full(len(out), np.nan)
            upper = np.full(len(out), np.nan)
        if not pooled.empty:
            lower = np.where(np.isnan(lower), float(pooled["lower_margin"].iloc[0]), lower)
            upper = np.where(np.isnan(upper), float(pooled["upper_margin"].iloc[0]), upper)
        if np.isnan(lower).any() or np.isnan(upper).any():
            raise PersistenceError(
                "conformal margins missing for some groups and no pooled "
                "fallback row present"
            )
        out["q05"] = np.clip(out["q05"] - lower, 0.0, None)
        out["q95"] = out["q95"] + upper
        return out
