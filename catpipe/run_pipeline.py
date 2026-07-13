"""
run_pipeline.py  --  the entry point. Connects extract -> transform ->
reconcile so the pipeline runs end to end from the command line.

Usage (set the DSN once, as an env var, so credentials never touch source):

    export CAT_DSN="postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME"

    # 1. Extract everything to a Parquet snapshot, then build frames:
    python -m catpipe.run_pipeline --extract --out ./snapshots

    # 2. Or, if you already have a snapshot, skip the DB and just build:
    python -m catpipe.run_pipeline --snapshot ./snapshots/<timestamp>

    # 3. Just the target-choice reconciliation on an existing snapshot:
    python -m catpipe.run_pipeline --snapshot ./snapshots/<timestamp> \
        --reconcile-only

Nothing here writes to Postgres; the only DB access is read-only SELECT in
extract.py. Frames are written as Parquet next to the snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import (assertions, baselines, extract, frames, harness,
               reconcile, transform)


def build_frames(snapshot_dir: str | Path) -> dict:
    """Load a snapshot and build the pool and team frames. Returns a dict of
    frames plus the reconciliation summary."""
    tables = transform.load_snapshot(snapshot_dir)

    required = {"raw_cost", "job_usage", "job_cost", "run_status"}
    missing = required - set(tables)
    if missing:
        raise SystemExit(
            f"Snapshot {snapshot_dir} is missing tables: {sorted(missing)}. "
            "Re-run with --extract, or check the snapshot path."
        )

    # Q1's tripwire on every snapshot: the append-only invariant that
    # point-in-time back-tests rest on. Cheap, and the day the loader's
    # upsert fires, this is what says so.
    assertions.assert_one_write_per_slice(tables["raw_cost"])

    pool = transform.build_pool_frame(tables)
    team = transform.build_team_frame(tables)

    recon = reconcile.reconcile_pool_day(tables["raw_cost"], tables["job_cost"])
    summary = reconcile.reconciliation_summary(recon)

    return {"pool": pool, "team": team,
            "reconciliation": recon, "recon_summary": summary}


def build_ml_frames(snapshot_dir: str | Path) -> dict:
    """The three model-ready frames of frames.py: 1a (pool), 1b (non-pool
    segments) and 2 (team). Returns {name: frame}; each frame carries its
    target, feature list and reports in .attrs.
    """
    tables = transform.load_snapshot(snapshot_dir)
    assertions.assert_one_write_per_slice(tables["raw_cost"])
    return {
        "frame_1a": frames.build_frame_1a(tables),
        "frame_1b": frames.build_frame_1b(tables),
        "frame_2": frames.build_frame_2(tables),
    }


def _write_ml_frames(ml: dict, out_dir: Path) -> None:
    """Parquet per frame plus a manifest carrying what parquet cannot:
    the declared feature lists, categoricals, and the orphan/grain reports.
    The manifest is the reviewer's single place to see exactly what each
    model is allowed to know.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, frame in ml.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
        manifest[name] = {
            "rows": len(frame),
            "target": frame.attrs.get("target"),
            "feature_cols": frame.attrs.get("feature_cols"),
            "categorical_cols": frame.attrs.get("categorical_cols"),
            "train_origin": frame.attrs.get("train_origin"),
            "orphan_report": frame.attrs.get("orphan_report"),
            "grain_report": frame.attrs.get("grain_report"),
        }
    (out_dir / "ml_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))


def backtest_baselines(ml: dict) -> dict:
    """Build-order item 4 in motion: the F3 baselines through the shared
    walk-forward harness, per frame, each on its own honest window (5.7):

      frame_1a  featured_gated only
      frame_1b  featured_gated AND post_glide (origin >= 2025-02-01)
      frame_2   featured_gated AND unknown_pct <= 0.20 (the A.3 filter)

    Returns {frame_name: {"summary": DataFrame, "ledger": DataFrame}}.
    A frame whose honest window is still too short for a single fold is
    reported as such rather than scored dishonestly.
    """
    masks = {
        "frame_1a": lambda f: None,
        "frame_1b": lambda f: f["post_glide"],
        "frame_2": lambda f: f["unknown_pct"] <= 0.20,
    }
    out = {}
    for name, frame in ml.items():
        try:
            summary, ledger = harness.run_models(
                frame, baselines.all_baselines(),
                mask=masks[name](frame))
            out[name] = {"summary": summary, "ledger": ledger}
        except harness.BacktestError as e:
            out[name] = {"error": str(e)}
    return out


def _write_backtests(bt: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, res in bt.items():
        if "error" in res:
            report[name] = {"error": res["error"]}
            continue
        res["ledger"].to_parquet(out_dir / f"backtest_{name}_ledger.parquet",
                                 index=False)
        report[name] = res["summary"].to_dict(orient="records")
    (out_dir / "backtest_summary.json").write_text(
        json.dumps(report, indent=2, default=str))


def _print_backtests(bt: dict) -> None:
    print("\nBASELINE WALK-FORWARD (F3), per frame")
    print("-" * 68)
    for name, res in bt.items():
        if "error" in res:
            print(f"  {name}: not scoreable -- {res['error']}")
            continue
        print(f"  {name}:")
        cols = ["model", "mae_daily", "monthly_pct_err", "coverage_90"]
        print(res["summary"][cols].to_string(index=False))


def _write_frames(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["pool"].to_parquet(out_dir / "frame_pool.parquet", index=False)
    result["team"].to_parquet(out_dir / "frame_team.parquet", index=False)
    result["reconciliation"].to_parquet(
        out_dir / "reconciliation_pool_day.parquet", index=False)
    (out_dir / "recon_summary.json").write_text(
        json.dumps(result["recon_summary"], indent=2))


def _print_report(result: dict) -> None:
    pool, team = result["pool"], result["team"]
    s = result["recon_summary"]
    print("\n" + "=" * 68)
    print("FRAMES BUILT")
    print("-" * 68)
    print(f"  pool frame : {len(pool):>7} rows  "
          f"regimes={dict(pool['data_regime'].value_counts())}")
    print(f"  team frame : {len(team):>7} rows")
    print("\nTARGET-CHOICE RECONCILIATION (job_cost vs raw_cost, pool-day)")
    print("-" * 68)
    print(f"  aggregate job/raw ratio : {s['aggregate_job_over_raw']}")
    print(f"  coverage                : {s['coverage_counts']}")
    print(f"  VERDICT: {s['verdict']}")
    print("=" * 68 + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the CAT cost pipeline.")
    p.add_argument("--extract", action="store_true",
                   help="Extract from Postgres first (needs CAT_DSN).")
    p.add_argument("--snapshot", type=str, default=None,
                   help="Existing snapshot dir to build from (skips extract).")
    p.add_argument("--out", type=str, default="./snapshots",
                   help="Root dir for snapshots and frames.")
    p.add_argument("--reconcile-only", action="store_true",
                   help="Only run the target-choice reconciliation.")
    p.add_argument("--ml-frames", action="store_true",
                   help="Also build the model-ready ML frames (1a, 1b, 2) "
                        "and write them with a feature manifest.")
    p.add_argument("--backtest", action="store_true",
                   help="Run the F3 baselines through the monthly "
                        "walk-forward on each ML frame (implies --ml-frames)"
                        " and write summaries + prediction ledgers.")
    args = p.parse_args(argv)

    if args.extract:
        dsn = os.environ.get("CAT_DSN")
        if not dsn:
            raise SystemExit("Set CAT_DSN to extract. See module docstring.")
        print("Extracting from Postgres (read-only) ...")
        snapshot_dir = extract.run_extract(dsn, out_root=args.out)
        print(f"Snapshot landed: {snapshot_dir}")
    elif args.snapshot:
        snapshot_dir = Path(args.snapshot)
    else:
        raise SystemExit("Pass --extract or --snapshot <dir>. "
                         "See module docstring for examples.")

    if args.reconcile_only:
        tables = transform.load_snapshot(snapshot_dir)
        recon = reconcile.reconcile_pool_day(
            tables["raw_cost"], tables["job_cost"])
        summary = reconcile.reconciliation_summary(recon)
        print(json.dumps(summary, indent=2))
        return 0

    result = build_frames(snapshot_dir)
    _write_frames(result, Path(snapshot_dir) / "frames")
    _print_report(result)
    print(f"Frames written to {Path(snapshot_dir) / 'frames'}")

    if args.ml_frames or args.backtest:
        ml = build_ml_frames(snapshot_dir)
        out = Path(snapshot_dir) / "frames"
        _write_ml_frames(ml, out)
        for name, frame in ml.items():
            n_feat = len(frame.attrs.get("feature_cols") or [])
            print(f"  {name}: {len(frame):>7} rows, {n_feat} features "
                  f"-> {out / (name + '.parquet')}")
        print(f"Feature manifest: {out / 'ml_manifest.json'}")

    if args.backtest:
        bt = backtest_baselines(ml)
        _write_backtests(bt, Path(snapshot_dir) / "frames")
        _print_backtests(bt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
