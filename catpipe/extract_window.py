"""
extract_window.py  --  windowed Postgres extract for scoring runs.

extract.py pulls FULL tables because, until Q1 closed, a delta extract
could not be trusted: a restated raw_cost would silently invalidate any
locally-held history. Q1 is now resolved (append-only in practice; the
loader's upsert has never fired and the one-write invariant is re-checked
in the pipeline assertions), so a windowed pull is legitimate for serving,
exactly as extract.py's docstring anticipated: only the WHERE predicate
changes; nothing downstream moves.

The scoring window is computed from the model card's features
(min_history.recommended_extract_days), not chosen by hand. The two large
tables are paged month-by-month as before, restricted to months
intersecting the window; the eight small tables are pulled whole, because
they are small and several (environment_config, run_status) are needed in
full for the gate and the tier filter anyway.

The snapshot directory produced is byte-compatible with extract.py's:
transform.load_snapshot() and every builder read it unchanged.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from . import extract as E


def window_start(days_back: int, today: date | None = None) -> date:
    today = today or datetime.now(timezone.utc).date()
    return today - timedelta(days=days_back)


def _extract_paged_window(engine, table: str, start: date) -> pd.DataFrame:
    """Month-paged pull restricted to run_date >= start. Reuses extract.py's
    month arithmetic so page boundaries are identical to a full extract's."""
    lo, hi = E._date_range(engine, table)
    lo = max(lo, date(start.year, start.month, 1))
    if lo > hi:
        return pd.DataFrame()
    frames = []
    for month_start in E._month_starts(lo, hi):
        month_end = E._next_month(month_start)
        q = text(
            f"SELECT * FROM {E.SCHEMA}.{table} "
            "WHERE run_date >= :start AND run_date < :end "
            "AND run_date >= :floor"
        )
        page = pd.read_sql(
            q, engine,
            params={"start": month_start, "end": month_end, "floor": start},
        )
        if len(page):
            frames.append(page)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return E._apply_dtypes(df)


def run_extract_window(dsn: str, out_root: str | Path, days_back: int) -> Path:
    """Extract a trailing window to a timestamped snapshot directory in the
    same layout as run_extract's, with the window recorded in the manifest."""
    engine = create_engine(dsn)
    snapshot = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    start = window_start(days_back)
    manifest = E.ExtractManifest(
        snapshot=snapshot, started=datetime.now(timezone.utc).isoformat()
    )
    manifest.density["_window"] = {
        "days_back": days_back, "start": str(start)
    }

    for table in E.SMALL_TABLES:
        df = E._extract_whole(engine, table)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)
        manifest.rows[table] = len(df)

    for table in E.PAGED_TABLES:
        df = _extract_paged_window(engine, table, start)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)
        manifest.rows[table] = len(df)
        manifest.density[table] = E.monthly_density(df)

    manifest.write(out_dir)
    engine.dispose()
    return out_dir
