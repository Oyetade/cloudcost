"""
extract.py  --  Postgres (read-only) -> Parquet snapshot

Lands the ten Cost Attribution Tool tables to local Parquet. The two large
tables (raw_cost, job_usage) are paged by run_date month-by-month to stay
under the connection's statement/volume limit; the eight small tables are
pulled whole. Dtypes are fixed here, once, so every downstream read is
already typed and compact.

Stack: sqlalchemy + psycopg + pandas + pyarrow only.

Until the append-only-vs-restated question (section 6 / open questions) is
settled, this extracts FULL tables, not deltas. When it closes, only
_month_bounds / the WHERE predicate changes; nothing downstream moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

SCHEMA = "cat"

# --- dtype policy, applied at landing -------------------------------------
# Low-cardinality strings -> category (this is where the memory savings live).
# Date/timestamp strings -> parsed. Everything else left as-is.
# Categoricals save memory on low-cardinality DESCRIPTIVE columns, but they
# are dangerous on columns used as groupby/merge KEYS: pandas groupby with
# the default observed=False generates the full cartesian product of all
# category levels, most of them empty, exploding row counts. So KEY columns
# (subscription_id, batch_account_name, pool_name, job_* keys, run_type)
# are deliberately kept as plain strings here; only descriptive columns are
# categoricals. Downstream groupbys also pass observed=True as belt-and-braces.
CATEGORY_COLS = {
    "subscription_name", "resource_type",
    "service_name", "service_tier", "meter",
    "meter_category", "meter_sub_category", "currency",
    "os_type", "location", "sku",
    "product_name", "product_type", "environment_tier",
    "environment_sub_tier", "status", "category",
    "ownership", "placement_score", "eviction_rate",
}
# Explicitly NOT categorical (join/group keys): subscription_id,
# resource_group_name, batch_account_name, pool_name, run_type, team,
# job_category, job_ownership, job_team.
DATE_COLS = {"run_date", "date", "start_date", "end_date"}
TS_COLS = {"start_time", "end_time", "update_time", "date_time", "run_time"}

# Tables small enough to pull whole.
SMALL_TABLES = [
    "job_cost", "job_classification", "environment_config", "run_status",
    "spot_prices", "dedicated_prices_retail", "spot_eviction_rates",
    "spot_placement_scores",
]
# Tables paged by run_date.
PAGED_TABLES = ["raw_cost", "job_usage"]


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in DATE_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
        elif c in TS_COLS:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        elif c in CATEGORY_COLS:
            df[c] = df[c].astype("category")
    return df


def _month_starts(first: date, last: date) -> list[date]:
    """Month-start boundaries covering [first, last], inclusive."""
    out, y, m = [], first.year, first.month
    while (y, m) <= (last.year, last.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _next_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


@dataclass
class ExtractManifest:
    snapshot: str
    started: str
    rows: dict[str, int] = field(default_factory=dict)
    density: dict[str, dict] = field(default_factory=dict)
    finished: str | None = None

    def write(self, out_dir: Path) -> None:
        self.finished = datetime.now(timezone.utc).isoformat()
        (out_dir / "_manifest.json").write_text(json.dumps(self.__dict__, indent=2))


def _date_range(engine, table: str) -> tuple[date, date]:
    q = text(f"SELECT min(run_date) AS lo, max(run_date) AS hi FROM {SCHEMA}.{table}")
    with engine.connect() as conn:
        lo, hi = conn.execute(q).one()
    return pd.to_datetime(lo).date(), pd.to_datetime(hi).date()


def _extract_whole(engine, table: str) -> pd.DataFrame:
    df = pd.read_sql(text(f"SELECT * FROM {SCHEMA}.{table}"), engine)
    return _apply_dtypes(df)


def _extract_paged(engine, table: str) -> pd.DataFrame:
    lo, hi = _date_range(engine, table)
    frames = []
    for start in _month_starts(lo, hi):
        end = _next_month(start)
        q = text(
            f"SELECT * FROM {SCHEMA}.{table} "
            "WHERE run_date >= :start AND run_date < :end"
        )
        page = pd.read_sql(q, engine, params={"start": start, "end": end})
        if len(page):
            frames.append(page)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _apply_dtypes(df)


def monthly_density(df: pd.DataFrame, date_col: str = "run_date") -> dict:
    """Row count per calendar month. Confirms whether a table's round start
    year (raw_cost 'from 2021', retail prices 'from 2017') is continuous
    coverage or a thin backfilled tail (section 5.7 residual check). Written
    to the manifest so the coverage claim is verifiable, not assumed.
    """
    if date_col not in df.columns or df.empty:
        return {}
    m = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)
    return m.value_counts().sort_index().to_dict()


def run_extract(dsn: str, out_root: str | Path) -> Path:
    """Extract all ten tables to a timestamped Parquet snapshot directory."""
    engine = create_engine(dsn)  # dsn: "postgresql+psycopg://user:pw@host/db"
    snapshot = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ExtractManifest(
        snapshot=snapshot, started=datetime.now(timezone.utc).isoformat()
    )

    for table in SMALL_TABLES:
        df = _extract_whole(engine, table)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)
        manifest.rows[table] = len(df)

    for table in PAGED_TABLES:
        df = _extract_paged(engine, table)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)
        manifest.rows[table] = len(df)
        # raw_cost's 'from 2021' is the long-tail claim the pool model's
        # baseline history rests on: record its shape so it is verifiable.
        manifest.density[table] = monthly_density(df)

    manifest.write(out_dir)
    engine.dispose()
    return out_dir


if __name__ == "__main__":
    import os
    dsn = os.environ["CAT_DSN"]  # keep credentials out of source
    path = run_extract(dsn, out_root="./snapshots")
    print(f"Snapshot landed: {path}")
