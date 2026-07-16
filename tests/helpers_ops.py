"""Shared helpers for the persistence/ledger/scoring tests: a synthetic
ten-table snapshot in the shapes extract.py lands, a mini feature factory
(stand-in for the local tree's feature_factory), and a fitted bundle.

Deterministic throughout (seeded, LightGBM deterministic=True) so the
round-trip identity test means something."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from catpipe import transform as T
from catpipe.persistence import (
    BoosterSpec, ModelCard, frame_dtypes, freeze_levels,
)

FEATURES = ["cost_lag1", "cost_lag7", "cost_roll28", "dow", "pool_name"]
CATEGORICALS = ["pool_name"]
GROUP_KEYS = ["subscription_id", "batch_account_name", "pool_name"]
TARGET = "cost"

SUB = "sub-001"
BATCH = "batchacct1"


def make_tables(n_days: int = 120, pools: int = 3, seed: int = 7,
                start: date = date(2026, 1, 1)) -> dict[str, pd.DataFrame]:
    """Synthetic snapshot tables in extract.py's landed shapes, entirely
    inside the featured_gated regime, every day's three runs Complete."""
    rng = np.random.default_rng(seed)
    days = [start + timedelta(days=i) for i in range(n_days)]

    raw_rows, usage_rows, cost_rows, rs_rows = [], [], [], []
    for d in days:
        for rtype in T.GATE_TYPES:
            rs_rows.append({
                "run_date": d, "subscription_id": SUB, "run_type": rtype,
                "status": "Complete",
                "update_time": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=6),
            })
        for p in range(pools):
            pool = f"pool_{p}"
            weekday = d.weekday() < 5
            base = 150.0 * (p + 1)
            cost = base * (1.0 if weekday else 0.35) * rng.lognormal(0, 0.2)
            raw_rows.append({
                "run_date": d, "subscription_id": SUB,
                "resource_group_name": f"rg-{p}",
                "resource_type": "microsoft.batch/batchaccounts",
                "meter": "D4s v3", "batch_account_name": BATCH,
                "pool_name": pool, "pre_tax_cost": cost,
                "usage_quantity": cost / 0.2,
            })
            job_id = f"job-{p}-{d.isoformat()}"
            seconds = float(max(0.0, rng.normal(3e4 if weekday else 1e3, 2e3)))
            usage_rows.append({
                "run_date": d, "subscription_id": SUB,
                "batch_account_name": BATCH, "pool_name": pool,
                "job_id": job_id, "job_seconds": seconds,
                "task_count": int(seconds // 30),
                "start_time": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=1),
                "end_time": pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=5),
            })
            cost_rows.append({
                "run_date": d, "subscription_id": SUB,
                "batch_account_name": BATCH, "pool_name": pool,
                "job_id": job_id, "job_name": f"nightly_{p}",
                "job_category": "reg-stress", "job_ownership": "owned",
                "job_team": f"team_{p}", "cost": cost * 0.8,
            })

    return {
        "raw_cost": pd.DataFrame(raw_rows),
        "job_usage": pd.DataFrame(usage_rows),
        "job_cost": pd.DataFrame(cost_rows),
        "run_status": pd.DataFrame(rs_rows),
    }


def write_snapshot(tables: dict[str, pd.DataFrame], out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def featurize_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Mini feature factory: lags/rolls per pool, shifted-then-rolled so
    day t is outside its own window, calendar lead. Stand-in for the local
    tree's feature_factory; the score pipeline receives the real one via
    --featurize."""
    df = frame.sort_values(GROUP_KEYS + ["run_date"]).copy()
    g = df.groupby(GROUP_KEYS, observed=True)
    df["cost_lag1"] = g[TARGET].shift(1)
    df["cost_lag7"] = g[TARGET].shift(7)
    df["cost_roll28"] = g[TARGET].transform(
        lambda s: s.shift(1).rolling(28, min_periods=28).mean()
    )
    df["dow"] = pd.to_datetime(df["run_date"]).dt.dayofweek.astype("int64")
    df["pool_name"] = df["pool_name"].astype("category")
    return df.dropna(subset=["cost_lag1", "cost_lag7", "cost_roll28"]).reset_index(drop=True)


def fit_boosters(frame: pd.DataFrame, seed: int = 7):
    import lightgbm as lgb

    X = frame[FEATURES]
    y = frame[TARGET]
    base = dict(
        objective="quantile", n_estimators=50, learning_rate=0.1,
        min_child_samples=10, deterministic=True, force_row_wise=True,
        seed=seed, verbose=-1,
    )
    boosters = {}
    for name, alpha in (("q05", 0.05), ("q50", 0.50), ("q95", 0.95)):
        m = lgb.LGBMRegressor(alpha=alpha, **base)
        m.fit(X, np.arcsinh(y), categorical_feature=CATEGORICALS)
        boosters[name] = m.booster_
    mean_params = {**base, "objective": "regression"}
    m = lgb.LGBMRegressor(**mean_params)
    m.fit(X, y, categorical_feature=CATEGORICALS)  # untransformed target
    boosters["mean"] = m.booster_
    return boosters


def make_card(featured: pd.DataFrame, frame_name: str = "pool") -> ModelCard:
    return ModelCard(
        frame=frame_name,
        target=TARGET,
        feature_names=list(FEATURES),
        feature_dtypes=frame_dtypes(featured, FEATURES),
        categorical_features=list(CATEGORICALS),
        categorical_levels=freeze_levels(featured, CATEGORICALS),
        group_keys=list(GROUP_KEYS),
        boosters=[
            BoosterSpec("q05", "asinh", True, 0.05),
            BoosterSpec("q50", "asinh", True, 0.50),
            BoosterSpec("q95", "asinh", True, 0.95),
            BoosterSpec("mean", "none", False),
        ],
        point_col="pred_mean",
        train_origin="2026-01-01",
        train_end="2026-04-30",
        snapshot="test-fixture",
        horizon_days=1,
        seed=7,
    )
