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

import pandas as pd

from . import (assertions, baselines, calibrate, detector, extract,
               frames, harness, models, reconcile, report, transform)


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
            "group_keys": frame.attrs.get("group_keys"),
            "feature_cols": frame.attrs.get("feature_cols"),
            "categorical_cols": frame.attrs.get("categorical_cols"),
            "train_origin": frame.attrs.get("train_origin"),
            "orphan_report": frame.attrs.get("orphan_report"),
            "grain_report": frame.attrs.get("grain_report"),
        }
    (out_dir / "ml_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))


def _models_for(frame, include_gbm: bool) -> list:
    """The model list for one frame: always the F3 baselines; plus the
    quantile GBM (raw and conformally calibrated) built from the frame's
    own declared features when asked.
    """
    out = list(baselines.all_baselines())
    if include_gbm:
        # raw AND conformally calibrated, side by side: the summary then
        # shows the coverage gap (~0.82 raw vs the 0.90 target) and its
        # correction on the same folds. A.4 Layer 1 alerts on the
        # CALIBRATED intervals, via the same calibrate.py machinery.
        out.append(models.QuantileGBM.for_frame(frame))
        out.append(calibrate.ConformalWrapper(
            models.QuantileGBM.for_frame(frame)))
    return out


def backtest_baselines(ml: dict, include_gbm: bool = False) -> dict:
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
                frame, _models_for(frame, include_gbm),
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
        cols = ["model", "mae_daily", "monthly_pct_err_estate",
                "monthly_bias", "monthly_wape", "coverage_90"]
        print(res["summary"][cols].to_string(index=False))


LEDGER_PREFERENCE = ("quantile_gbm", "rolling_median_28",
                     "seasonal_naive", "trend_3m")


def run_detector_on_backtests(bt: dict, ml: dict,
                              tables: dict,
                              previous_alerts=None,
                              detect_days: int | None = None):
    """A.4 assembled from what already exists. Layers 1 and 1.5 replay over
    the backtest prediction ledgers — RAW model intervals, because Layer 1
    computes its own trailing conformal margins per scored day (the
    calibrate.py reuse). The GBM's ledger is preferred where present, the
    strongest baseline's otherwise, so the detector demonstrates end to end
    even before lightgbm approval. Layer 2 profiles job_cost directly; the
    attribution rule reads frame_2's unknown_pct.
    """
    ledgers = {}
    for name, res in bt.items():
        if "ledger" not in res:
            continue
        led = res["ledger"]
        # the GBM's ledger where the backtest included it, the strongest
        # baseline's otherwise (a --detect run without --gbm)
        model = next((m for m in LEDGER_PREFERENCE
                      if (led["model"] == m).any()), None)
        if model is None:
            continue
        gk = ml[name].attrs.get("group_keys")
        ledgers[name] = (led[led["model"] == model].copy(), gk)

    # scoring window: the trailing detect_days of the data, so a scheduled
    # run produces the nightly view rather than a full-history replay
    # (74,586-alert lesson). detect_days=None replays everything, which is
    # the detector's own back-test mode.
    score_from = None
    if detect_days is not None and ledgers:
        last = max(pd.to_datetime(led["run_date"]).max()
                   for led, _ in ledgers.values())
        score_from = (last - pd.Timedelta(days=detect_days - 1)).date()

    return detector.run_detector(
        ledgers,
        job_cost=tables.get("job_cost"),
        frame_2=ml.get("frame_2"),
        previous_alerts=previous_alerts,
        score_from=score_from,
    )


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
    p.add_argument("--gbm", action="store_true",
                   help="Add the quantile GBM (A.1/A.2/A.3) to the backtest"
                        " model list. Implies --backtest.")
    p.add_argument("--report", action="store_true",
                   help="Write a self-contained HTML performance report "
                        "(SVG charts, no plotting library) from the "
                        "backtest, plus the alert summary when --detect "
                        "also runs. Implies --backtest.")
    p.add_argument("--detect-days", type=int, default=30, metavar="N",
                   help="Detector scoring window: alert on the trailing N "
                        "days only (default 30). 0 replays the full "
                        "history — the detector's own back-test mode.")
    p.add_argument("--detect", action="store_true",
                   help="Run the A.4 detector (interval exceedance, CUSUM "
                        "drift, job robust-z, attribution health) over the "
                        "backtest ledgers and snapshot; implies --backtest. "
                        "Preserves triage statuses from an existing "
                        "alerts.parquet.")
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

    if args.gbm or args.detect or args.report:
        args.backtest = True
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
        bt = backtest_baselines(ml, include_gbm=args.gbm)
        _write_backtests(bt, Path(snapshot_dir) / "frames")
        _print_backtests(bt)

    if args.detect:
        tables = transform.load_snapshot(snapshot_dir)
        alerts_path = Path(snapshot_dir) / "frames" / "alerts.parquet"
        previous = (pd.read_parquet(alerts_path)
                    if alerts_path.exists() else None)
        alerts = run_detector_on_backtests(
            bt, ml, tables, previous_alerts=previous,
            detect_days=args.detect_days or None)
        alerts_path.parent.mkdir(parents=True, exist_ok=True)
        alerts.to_parquet(alerts_path, index=False)
        n_by = alerts.groupby(["layer", "severity"], observed=True) \
            .size().to_dict() if len(alerts) else {}
        print(f"\nA.4 DETECTOR: {len(alerts)} alerts -> {alerts_path}")
        for (layer, sev), n in sorted(n_by.items()):
            print(f"  {layer:<14} {sev:<7} {n}")
        if len(alerts):
            print("top of the triage queue:")
            for _, a in alerts.head(3).iterrows():
                print(f"  [{a['severity']}] {a['run_date']} {a['message']}")

    if args.report:
        rp = Path(snapshot_dir) / "frames" / "backtest_report.html"
        report.write_report(rp, bt,
                            alerts=alerts if args.detect else None)
        print(f"\nHTML report -> {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
