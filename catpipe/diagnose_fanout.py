"""
diagnose_fanout.py  --  find where the pool frame's row count explodes.

Symptom: raw_cost has ~4.66M rows but build_pool_frame produced a ~63M-row
frame (a ~14x blow-up). That means a join is fanning out on a key that is
not unique on one side. This script traces the row count through every stage
of the build against a REAL snapshot and prints, at each step, the row count
and whether the key is unique -- so the exact fanout point is visible rather
than guessed.

Run:
    PYTHONPATH=. python -m catpipe.diagnose_fanout --snapshot ./snapshots/<ts>

It mutates nothing and writes nothing; it only reads the snapshot and prints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import transform as T


def _uniq(df: pd.DataFrame, keys: list[str]) -> str:
    present = [k for k in keys if k in df.columns]
    if len(present) != len(keys):
        missing = set(keys) - set(present)
        return f"MISSING KEYS {missing}"
    dup = df.duplicated(present).sum()
    return "unique" if dup == 0 else f"{dup:,} DUPLICATE keys"


def _nulls(df: pd.DataFrame, keys: list[str]) -> str:
    out = []
    for k in keys:
        if k in df.columns:
            n = df[k].isna().sum()
            if n:
                out.append(f"{k}={n:,} nulls")
    return ", ".join(out) if out else "no nulls in keys"


def diagnose(snapshot_dir: str | Path) -> None:
    tables = T.load_snapshot(snapshot_dir)
    raw_cost = tables["raw_cost"]
    job_usage = tables["job_usage"]
    job_cost = tables["job_cost"]
    run_status = tables["run_status"]

    line = "=" * 70
    print(line)
    print("INPUT ROW COUNTS")
    print("-" * 70)
    for name, df in [("raw_cost", raw_cost), ("job_usage", job_usage),
                     ("job_cost", job_cost), ("run_status", run_status)]:
        print(f"  {name:12} {len(df):>12,} rows")

    print("\nKEY UNIQUENESS ON INPUTS")
    print("-" * 70)
    print(f"  raw_cost on POOL_KEYS      : {_uniq(raw_cost, T.POOL_KEYS)}")
    print(f"    (raw_cost is expected NON-unique: many meters per pool-day)")
    print(f"  raw_cost pool-key nulls    : {_nulls(raw_cost, T.POOL_KEYS)}")
    print(f"  job_usage on JOB_KEYS      : {_uniq(job_usage, T.JOB_KEYS)}")
    print(f"  job_cost on JOB_KEYS       : {_uniq(job_cost, T.JOB_KEYS)}")
    print(f"  run_status per (date,sub,type): "
          f"{_uniq(run_status, ['run_date','subscription_id','run_type'])}")

    # Stage 1: the cost target
    target = T.daily_cost_by_pool(raw_cost)
    print("\nSTAGE-BY-STAGE ROW COUNTS")
    print("-" * 70)
    print(f"  1. daily_cost_by_pool(raw_cost)  = {len(target):>12,}  "
          f"[{_uniq(target, T.POOL_KEYS)}]")
    print(f"     nulls dropped by groupby?      {_nulls(raw_cost, T.POOL_KEYS)}")

    # Stage 2: the gate
    gate = T.build_gate(run_status)
    print(f"  2. build_gate(run_status)        = {len(gate):>12,}  "
          f"[{_uniq(gate, ['run_date','subscription_id'])}]")

    # Stage 3: the activity aggregate
    joined, _ = T.join_job_attributes(job_usage, job_cost)
    activity = (
        joined.groupby(T.POOL_KEYS, observed=True)
        .agg(job_seconds=("job_seconds", "sum"),
             task_count=("task_count", "sum"))
        .reset_index()
    )
    print(f"  3. activity aggregate            = {len(activity):>12,}  "
          f"[{_uniq(activity, T.POOL_KEYS)}]")

    # Stage 4: target x activity
    frame = target.merge(activity, on=T.POOL_KEYS, how="left")
    grew4 = len(frame) - len(target)
    print(f"  4. target x activity merge       = {len(frame):>12,}  "
          f"({'+' if grew4>=0 else ''}{grew4:,} vs target)")
    if len(frame) > len(target):
        print("     ^^ FANOUT: activity not unique on POOL_KEYS, "
              "or target has dup keys")

    # Stage 5: regime + drop
    frame = T.stamp_regime(frame)
    frame = T.drop_pre_coverage(frame)
    print(f"  5. stamp_regime + drop_pre_cov   = {len(frame):>12,}")

    # Stage 6: the gate join
    before6 = len(frame)
    frame = T.apply_gate(frame, gate, "diagnose",
                         run_status_start=T.RUN_STATUS_START)
    print(f"  6. apply_gate (before drop)      = merged then filtered")
    print(f"     rows after gate              = {len(frame):>12,}  "
          f"({'+' if len(frame)-before6>=0 else ''}{len(frame)-before6:,} "
          f"vs stage 5)")
    if len(frame) > before6:
        print("     ^^ FANOUT: gate not unique on (run_date, subscription_id)")

    print(line)
    print("READING THIS: the first stage whose row count exceeds the expected")
    print("pool-day count is the fanout source. raw_cost row count is the")
    print("ceiling for a correct pool frame -- the frame must be SMALLER.")
    print(line)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diagnose pool-frame row fanout.")
    p.add_argument("--snapshot", required=True, help="Snapshot dir to read.")
    args = p.parse_args(argv)
    diagnose(args.snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
