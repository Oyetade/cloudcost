"""
models.py  --  the quantile GBM of A.1 (and, unchanged, of A.2 and A.3:
one model class, three frames). Plugs into the harness through the same
fit / predict-quantiles interface as the baselines, so the comparison with
the incumbent is on identical folds by construction.

Design decisions, from the doc:

  Quantile objectives at 5/50/95 from the outset (A.1): three LightGBM
  boosters, one per quantile. The median is the point forecast; the 5-95
  band is A.4 Layer 1. Predicted quantiles are additionally sorted per row,
  because independently trained quantile models can cross, and a crossed
  interval is not an interval.

  Native categoricals, never one-hot (A.1): pool_name, job_team,
  subscription_id, segment, tier, region enter as pandas category dtype
  with levels FROZEN AT FIT — a category unseen in training is mapped to
  missing at predict, not silently given a fresh code.

  Target transform (7.3): daily cost is strictly positive and right-skewed,
  train on a transformed target and invert for reporting. Default is asinh
  rather than log, because the padded frames legitimately contain exact
  zero-cost days and log would need an epsilon fudge. Quantiles are
  invariant under monotone transforms, so per-day quantiles invert exactly;
  the 5.4 caution (monthly totals from a log-scale model need summed
  simulated paths, not exponentiated points) applies to aggregates and is
  carried in the class docstring for the reporting layer.

  Early stopping on a chronological tail of the training window (never a
  random split: shuffling time leaks the future into validation). If the
  tail is too thin, fall back to a fixed round count rather than validating
  on nothing.

Stack: lightgbm (approved July 2026) + pandas + numpy.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import FrameSpec

DEFAULT_QUANTILES = (0.05, 0.50, 0.95)

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}


_TRANSFORMS = {
    "asinh": (np.arcsinh, np.sinh),
    "log1p": (np.log1p, np.expm1),
    "none": (lambda x: x, lambda x: x),
}


class QuantileGBM:
    """Gradient-boosted quantile forecaster over a catpipe frame.

    Features and categoricals default to what the frame declares about
    itself (attrs feature_cols / categorical_cols, set by frames.py), passed
    explicitly at construction because pandas does not reliably propagate
    attrs through the harness's train/test slicing. Every feature in the
    frame is lagged or calendar-known by construction (feature_factory), so
    this model cannot leak same-day cost even by accident.

    Reporting caveat (5.4): per-day quantiles invert exactly under the
    monotone target transform, but a MONTHLY total should be produced by
    summing simulated daily paths, not by summing inverted daily medians;
    the harness's monthly_pct_err uses summed medians for every model alike,
    which is fair for comparison but is a point estimate, not a monthly
    median.
    """

    name = "quantile_gbm"

    def __init__(
        self,
        features: list[str],
        categoricals: list[str] | None = None,
        transform: str = "asinh",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        params: dict | None = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
        valid_tail_days: int = 60,
        min_valid_rows: int = 30,
    ):
        if 0.50 not in quantiles:
            raise ValueError("quantiles must include the median (0.50): it "
                             "is the point forecast")
        if transform not in _TRANSFORMS:
            raise ValueError(f"transform must be one of {list(_TRANSFORMS)}")
        self.features = list(features)
        self.categoricals = [c for c in (categoricals or [])
                             if c in self.features]
        self.transform = transform
        self.quantiles = tuple(sorted(quantiles))
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.valid_tail_days = valid_tail_days
        self.min_valid_rows = min_valid_rows

    @classmethod
    def for_frame(cls, frame: pd.DataFrame, **kwargs) -> "QuantileGBM":
        """Construct from what the frame declares about itself. The frame is
        the single source of truth for what the model is allowed to know.
        """
        feats = frame.attrs.get("feature_cols")
        if not feats:
            raise ValueError(
                "frame declares no feature_cols in .attrs; build it with "
                "frames.build_frame_* or pass features explicitly")
        return cls(features=feats,
                   categoricals=frame.attrs.get("categorical_cols") or [],
                   **kwargs)

    # -- internals ----------------------------------------------------------

    def _matrix(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing declared features: {missing}")
        X = frame[self.features].copy()
        for c in self.categoricals:
            if fit:
                X[c] = X[c].astype("category")
                self._cat_levels[c] = X[c].cat.categories
            else:
                # freeze levels at fit: unseen categories become missing
                # EXPLICITLY (pandas 4 will refuse the implicit coercion),
                # never a silently re-coded new level
                known = frame[c].where(frame[c].isin(self._cat_levels[c]))
                X[c] = pd.Categorical(known,
                                      categories=self._cat_levels[c])
        num = [c for c in self.features if c not in self.categoricals]
        X[num] = X[num].astype(float)
        return X

    def fit(self, train: pd.DataFrame, spec: FrameSpec) -> None:
        fwd, _ = _TRANSFORMS[self.transform]

        self._cat_levels = {}
        X = self._matrix(train, fit=True)
        y_raw = train[spec.target].astype(float).to_numpy()
        y = fwd(y_raw)

        # chronological tail for early stopping; never a random split
        dates = pd.to_datetime(train[spec.date_col])
        cutoff = dates.max() - pd.Timedelta(days=self.valid_tail_days - 1)
        valid_mask = (dates >= cutoff).to_numpy()
        use_valid = int(valid_mask.sum()) >= self.min_valid_rows \
            and int((~valid_mask).sum()) >= self.min_valid_rows

        self._boosters = {}
        for q in self.quantiles:
            params = dict(self.params, objective="quantile", alpha=q)
            self._boosters[q] = self._train_one(
                params, X, y, valid_mask, use_valid)

        # The mean booster (7.3 / 5.4, added 15 July 2026). Quantiles invert
        # exactly under a monotone transform, so q50 IS the daily median: that
        # part was never wrong. The bias was downstream, in the SUM. For a
        # right-skewed daily cost the median sits below the mean, so summing
        # daily medians under-forecasts a monthly total, one-signed, with no
        # cancellation: exactly the estate ~ wape ~ |bias| ~ -0.28 signature
        # on frames 1a and 2. This booster is trained on the UNTRANSFORMED
        # target with an L2 objective, so it estimates E[y|x] directly and
        # sums to an unbiased monthly total. It does not touch q05/q50/q95:
        # the intervals and their conformal calibration are unchanged.
        self._mean_booster = self._train_one(
            dict(self.params, objective="regression"),
            X, y_raw, valid_mask, use_valid)

    def _train_one(self, params, X, y, valid_mask, use_valid):
        """Early stopping on a chronological tail, then refit on the full
        window at the stopped round, so the tail's information is not thrown
        away at predict time. Shared by the quantile and mean objectives so
        they cannot drift apart.
        """
        if use_valid:
            dtrain = lgb.Dataset(X[~valid_mask], label=y[~valid_mask],
                                 categorical_feature=self.categoricals)
            dvalid = lgb.Dataset(X[valid_mask], label=y[valid_mask],
                                 reference=dtrain,
                                 categorical_feature=self.categoricals)
            booster = lgb.train(
                params, dtrain,
                num_boost_round=self.num_boost_round,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds,
                                              verbose=False),
                           lgb.log_evaluation(0)],
            )
            return lgb.train(
                params,
                lgb.Dataset(X, label=y, categorical_feature=self.categoricals),
                num_boost_round=max(booster.best_iteration, 1),
            )
        return lgb.train(
            params,
            lgb.Dataset(X, label=y, categorical_feature=self.categoricals),
            num_boost_round=300,
        )

    def predict(self, test: pd.DataFrame, spec: FrameSpec) -> pd.DataFrame:
        _, inv = _TRANSFORMS[self.transform]
        X = self._matrix(test, fit=False)

        raw = np.column_stack([
            self._boosters[q].predict(X) for q in self.quantiles
        ])
        # non-crossing: independently trained quantile boosters can cross;
        # sorting per row restores a valid interval (standard rearrangement)
        raw.sort(axis=1)
        vals = inv(raw)

        out = test[spec.group_keys + [spec.date_col]].copy()
        for i, q in enumerate(self.quantiles):
            out[f"q{int(round(q * 100)):02d}"] = vals[:, i]
        cols = [f"q{int(round(q * 100)):02d}" for q in self.quantiles]
        out[cols] = out[cols].clip(lower=0)  # cost is non-negative

        # pred_mean is trained on the RAW target, so it is NOT inverted: it is
        # already on the cost scale. This is the column to sum for a monthly
        # total; q50 remains the daily median and the honest thing to report
        # for a single day.
        if getattr(self, "_mean_booster", None) is not None:
            out["pred_mean"] = np.clip(
                self._mean_booster.predict(X), 0, None)
        return out

    def feature_importance(self) -> pd.DataFrame:
        """Gain importance of the median booster: the 'explain' verb's first
        instalment (charter 2.1), and the sanity check that the model leans
        on activity and calendar rather than memorising pool identity.
        """
        b = self._boosters[0.50]
        return pd.DataFrame({
            "feature": b.feature_name(),
            "gain": b.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False).reset_index(drop=True)
