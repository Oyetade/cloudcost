"""
score_pipeline.py  --  the inference entry point: snapshot -> frame ->
reloaded model -> prediction ledger.

The principle this module enforces: there is no inference transform path,
only the training one. build_pool_frame / build_team_frame (and, in the
local tree, the frames.py builders) ARE the inference path; this module
only sequences them, checks preconditions loudly, and writes the ledger
and a run manifest.

Fail closed means a DataQualityError aborts the run. The temptation to
catch, warn and score anyway is the silent-fallback pattern that made a
broken fix invisible for three consecutive runs; a serving path has the
same hazard and nobody reruns it three times to notice. An abort still
writes its manifest, so the audit trail survives the failure.

Usage (mirrors run_pipeline; credentials stay in CAT_DSN):

    # 1. Windowed extract, then score the pool model:
    python -m catpipe.score_pipeline --extract --days 120 --out ./snapshots \
        --bundle ./models/pool/v2026-08-01 --ledger ./ledger --frame pool

    # 2. Or score an existing snapshot:
    python -m catpipe.score_pipeline --snapshot ./snapshots/<timestamp> \
        --bundle ./models/pool/v2026-08-01 --ledger ./ledger --frame pool

    # 3. With the local tree's feature factory (dotted path to a function
    #    frame -> frame that adds lags/rolls/calendar/drift):
    ...  --featurize catpipe.feature_factory:featurize_pool

Nothing here writes to Postgres. The only DB access is the read-only
windowed SELECT in extract_window.py when --extract is passed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import transform
from .assertions import DataQualityError
from .ledger import PredictionLedger
from .min_history import min_history_days
from .persistence import LoadedModel, PersistenceError, load_bundle

# Frame registry: this tree's builders. The local tree adds its frames.py
# builders here (frame_1a, frame_1b, frame_2) when this module is merged.
FRAME_BUILDERS: dict[str, Callable] = {
    "pool": transform.build_pool_frame,
    "team": transform.build_team_frame,
}


class ScoringAborted(RuntimeError):
    """The run stopped. The manifest on disk says why."""


def _resolve_featurize(dotted: str | None) -> Callable | None:
    """'module.sub:function' -> callable. Kept as a dotted path so this
    module has no import-time dependency on the feature factory, which may
    not be merged into every checkout."""
    if not dotted:
        return None
    mod_name, _, fn_name = dotted.partition(":")
    if not fn_name:
        raise SystemExit(f"--featurize must be module:function, got {dotted!r}")
    return getattr(importlib.import_module(mod_name), fn_name)


def score_snapshot(
    snapshot_dir: str | Path,
    bundle_path: str | Path,
    ledger_root: str | Path,
    frame_name: str,
    featurize: Callable | None = None,
    ignore_watermark: bool = False,
) -> dict:
    """Score one snapshot with one bundle. Returns the manifest dict.

    Steps, in order, each one able to abort the run:
      1. load the bundle; its card decides features and history demand
      2. load the snapshot; refuse if its window is shorter than the card's
         features require (edge-degraded rolls must never serve)
      3. build the frame through the TRAINING builder; DataQualityError
         aborts (a partition-identity break means no scoring today)
      4. featurize with the training factory, if supplied
      5. select scoreable rows: featured_gated regime, gated_complete state
         (C8: a day is scoreable as soon as its runs are Complete), above
         this model version's watermark
      6. predict via LoadedModel (frozen levels, schema hash, never-null
         checks, quantile sorting, conformal margins)
      7. append to the ledger; advance the watermark; write the manifest
    """
    started = datetime.now(timezone.utc).isoformat()
    snapshot_dir = Path(snapshot_dir)
    model = load_bundle(bundle_path)
    card = model.card
    model_version = Path(bundle_path).name

    if card.frame != frame_name:
        raise ScoringAborted(
            f"bundle at {bundle_path} is for frame {card.frame!r}, "
            f"--frame asked for {frame_name!r}"
        )
    builder = FRAME_BUILDERS.get(frame_name)
    if builder is None:
        raise ScoringAborted(
            f"unknown frame {frame_name!r}; known: {sorted(FRAME_BUILDERS)}"
        )

    ledger = PredictionLedger(ledger_root, frame_name, card.group_keys)
    wm = None if ignore_watermark else ledger.watermark(model_version)
    manifest_payload: dict = {
        "frame": frame_name,
        "model_version": model_version,
        "snapshot": str(snapshot_dir),
        "started_at": started,
        "watermark_before": str(wm.date()) if wm is not None else None,
    }

    def abort(reason: str, *, soft: bool = False) -> dict:
        manifest_payload.update(aborted=True, abort_reason=reason)
        _write_manifest(ledger_root, frame_name, manifest_payload)
        if soft:
            return manifest_payload
        raise ScoringAborted(reason)

    # 2. snapshot, and the window check computed from the card's features
    tables = transform.load_snapshot(snapshot_dir)
    required = {"raw_cost", "job_usage", "job_cost", "run_status"}
    missing = required - set(tables)
    if missing:
        return abort(f"snapshot missing tables: {sorted(missing)}")

    needed = min_history_days(card.feature_names, card.horizon_days)
    rd = pd.to_datetime(tables["raw_cost"]["run_date"])
    span = int((rd.max() - rd.min()).days) + 1 if len(rd) else 0
    manifest_payload.update(min_history_days=needed, snapshot_span_days=span)
    if span < needed:
        return abort(
            f"snapshot spans {span} days; the declared features need "
            f"{needed}. Refusing to serve edge-degraded features; extract "
            f"a longer window (--days)."
        )

    # 3-4. frame through the training path, fail closed
    try:
        frame = builder(tables)
        if featurize is not None:
            frame = featurize(frame)
    except DataQualityError as exc:
        return abort(f"data-quality assertion failed: {exc}")
    if frame.empty:
        return abort("frame is empty after build")

    # 5. scoreable rows
    mask = pd.Series(True, index=frame.index)
    if "data_regime" in frame.columns:
        mask &= frame["data_regime"].astype(str).eq("featured_gated")
    if "gate_state" in frame.columns:
        mask &= frame["gate_state"].astype(str).eq("gated_complete")
    if wm is not None:
        mask &= pd.to_datetime(frame["run_date"]) > wm
    to_score = frame.loc[mask]
    if to_score.empty:
        return abort(f"no scoreable rows beyond watermark {wm}", soft=True)

    # 6. predict
    try:
        preds = model.predict(to_score)
    except PersistenceError as exc:
        return abort(f"predict refused: {exc}")
    unseen = preds.attrs.get("unseen_level_counts", {})

    # 7. ledger, watermark, manifest
    out = pd.concat(
        [to_score[["run_date", *card.group_keys]].reset_index(drop=True),
         preds.reset_index(drop=True)],
        axis=1,
    )
    ledger_file = ledger.append(out, model_version)
    max_rd = pd.to_datetime(to_score["run_date"]).max()
    if not ignore_watermark or wm is None or max_rd > wm:
        ledger.advance_watermark(model_version, max_rd)

    manifest_payload.update(
        aborted=False,
        rows_scored=int(len(out)),
        max_run_date=str(max_rd.date()),
        unseen_level_counts=unseen,
        point_col=card.point_col,
        ledger_file=str(ledger_file),
    )
    _write_manifest(ledger_root, frame_name, manifest_payload)
    return manifest_payload


def _write_manifest(ledger_root: str | Path, frame_name: str, payload: dict) -> Path:
    d = Path(ledger_root) / f"frame={frame_name}" / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    f = d / f"run_{stamp}.json"
    f.write_text(json.dumps(payload, indent=2))
    return f


def _print_report(m: dict) -> None:
    print("\n" + "=" * 68)
    print("SCORING RUN")
    print("-" * 68)
    if m.get("aborted"):
        print(f"  ABORTED: {m['abort_reason']}")
    else:
        print(f"  frame          : {m['frame']}")
        print(f"  model version  : {m['model_version']}")
        print(f"  rows scored    : {m['rows_scored']:>7}")
        print(f"  through        : {m['max_run_date']}")
        print(f"  point column   : {m['point_col']}")
        if any(m["unseen_level_counts"].values()):
            print(f"  UNSEEN LEVELS  : {m['unseen_level_counts']}")
        print(f"  ledger file    : {m['ledger_file']}")
    print("=" * 68 + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score a snapshot with a saved model.")
    p.add_argument("--extract", action="store_true",
                   help="Windowed extract from Postgres first (needs CAT_DSN).")
    p.add_argument("--days", type=int, default=120,
                   help="Trailing window for --extract (default 120).")
    p.add_argument("--out", type=str, default="./snapshots",
                   help="Root dir for snapshots.")
    p.add_argument("--snapshot", type=str, default=None,
                   help="Existing snapshot dir to score from (skips extract).")
    p.add_argument("--bundle", type=str, required=True,
                   help="Model bundle dir (models/<frame>/<version>).")
    p.add_argument("--ledger", type=str, default="./ledger",
                   help="Prediction ledger root.")
    p.add_argument("--frame", type=str, required=True,
                   choices=sorted(FRAME_BUILDERS),
                   help="Which frame builder to run.")
    p.add_argument("--featurize", type=str, default=None,
                   help="Dotted path module:function adding the training "
                        "factory's features to the built frame.")
    p.add_argument("--ignore-watermark", action="store_true",
                   help="Re-score everything scoreable in the snapshot.")
    args = p.parse_args(argv)

    if args.extract:
        dsn = os.environ.get("CAT_DSN")
        if not dsn:
            raise SystemExit("Set CAT_DSN to extract. See module docstring.")
        from .extract_window import run_extract_window
        print(f"Extracting trailing {args.days} days from Postgres (read-only) ...")
        snapshot_dir = run_extract_window(dsn, args.out, args.days)
        print(f"Snapshot landed: {snapshot_dir}")
    elif args.snapshot:
        snapshot_dir = Path(args.snapshot)
    else:
        raise SystemExit("Pass --extract or --snapshot <dir>. "
                         "See module docstring for examples.")

    manifest = score_snapshot(
        snapshot_dir=snapshot_dir,
        bundle_path=args.bundle,
        ledger_root=args.ledger,
        frame_name=args.frame,
        featurize=_resolve_featurize(args.featurize),
        ignore_watermark=args.ignore_watermark,
    )
    _print_report(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
