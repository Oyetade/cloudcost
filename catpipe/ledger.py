"""
ledger.py  --  the prediction ledger: the serving pipeline's product.

Metrics, reports, alerts and drift monitoring are all views over this
table; none of them can reconstruct it. So it is append-only, and its grain
carries every discriminator:

    (run_date, *group_keys, model_version, scored_at)

Days are legitimately re-scored: a load completes late, or a new model
deploys. Two predictions for one day with no discriminator is the Q24
lesson again, duplication living in the key rather than the data.
current_view() resolves latest-by-scored_at per (day, group, model
version); nothing on disk is ever rewritten.

current_view() also refuses to blend model versions silently. A residual
series spanning a retrain boundary would feed CUSUM residuals from a model
that no longer exists, and the trailing MAD-based sigma with it; the first
retrain would produce an alert storm (R6 self-inflicted) or a blind spot.

Storage: parquet, one file per scoring run, under
    <root>/frame=<frame>/scored_at=<ts>.parquet
with the per-version watermark alongside as watermark.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


class LedgerError(RuntimeError):
    pass


REQUIRED_COLS = ("run_date", "model_version", "scored_at")


class PredictionLedger:
    def __init__(self, root: str | Path, frame: str, group_keys: Sequence[str]):
        self.root = Path(root)
        self.frame = frame
        self.group_keys = list(group_keys)
        self.dir = self.root / f"frame={frame}"
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- write ----------------------------------------------------------------

    def append(self, predictions: pd.DataFrame, model_version: str) -> Path:
        """Append one scoring run. Refuses duplicates on the full grain
        within the append: a re-score is a new scored_at, never a rewrite."""
        df = predictions.copy()
        scored_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        df["model_version"] = model_version
        df["scored_at"] = scored_at

        missing = [c for c in (*REQUIRED_COLS, *self.group_keys)
                   if c not in df.columns]
        if missing:
            raise LedgerError(f"ledger append missing columns: {missing}")

        grain = ["run_date", *self.group_keys, "model_version", "scored_at"]
        if df.duplicated(subset=grain).any():
            raise LedgerError(
                "duplicate rows on the ledger grain within one append; "
                "the grain is missing a discriminator"
            )

        path = self.dir / f"scored_at={scored_at}.parquet"
        if path.exists():
            raise LedgerError(f"{path} already exists; appends are immutable")
        df.to_parquet(path, index=False)
        return path

    # -- read -----------------------------------------------------------------

    def read_all(self) -> pd.DataFrame:
        files = sorted(self.dir.glob("scored_at=*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def current_view(self, model_version: str | None = None) -> pd.DataFrame:
        """Latest prediction per (run_date, group), for one model version.

        With more than one version present and none named, this raises
        rather than blending: see the module docstring.
        """
        df = self.read_all()
        if df.empty:
            return df
        if model_version is not None:
            df = df[df["model_version"] == model_version]
        elif df["model_version"].nunique() > 1:
            raise LedgerError(
                "ledger holds multiple model versions "
                f"({sorted(df['model_version'].unique())}); name one, or "
                "handle versions explicitly. Mixing them silently corrupts "
                "residual-based detection."
            )
        keys = ["run_date", *self.group_keys, "model_version"]
        df = df.sort_values("scored_at")
        return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

    # -- watermark ------------------------------------------------------------

    @property
    def _watermark_file(self) -> Path:
        return self.dir / "watermark.json"

    def watermark(self, model_version: str) -> pd.Timestamp | None:
        """Max run_date already scored by this model version. Watermarks are
        per version: a new deployment starts fresh and back-scores its own
        history if asked."""
        if not self._watermark_file.exists():
            return None
        data = json.loads(self._watermark_file.read_text())
        v = data.get(model_version)
        return pd.Timestamp(v) if v else None

    def advance_watermark(self, model_version: str, run_date) -> None:
        run_date = pd.Timestamp(run_date)
        data = {}
        if self._watermark_file.exists():
            data = json.loads(self._watermark_file.read_text())
        prev = data.get(model_version)
        if prev and pd.Timestamp(prev) > run_date:
            raise LedgerError(
                f"watermark for {model_version} would move backwards "
                f"({prev} -> {run_date.date()})"
            )
        data[model_version] = str(run_date.date())
        self._watermark_file.write_text(json.dumps(data, indent=2))
