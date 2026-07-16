# Operationalisation layer: merge note

Verdict first: five new modules and four new test files, dropped into the
tree you shared, run 91 tests passing (your 60 plus 31 new) on lightgbm
4.6.0, pandas 3.0.2, numpy 2.4.4, and the CLI scores a snapshot end to end
through the real `build_pool_frame`. No existing module is modified; every
file in this delivery is new, so the standing merge caution (copy changed
modules individually, never unzip over the tree) reduces to copying these
in.

One precondition observed rather than shipped: the tree you sent still has
`transform.py` importing `assertions`/`features` as top-level modules. Your
local tree fixed that on 11-14 July (the dual-DataQualityError issue); the
new modules use `from .assertions import DataQualityError` and assume that
fix. Nothing to do if your local `transform.py` is current.

## New modules

| File | Delivers |
|---|---|
| `catpipe/persistence.py` | `ModelCard`, `save_bundle`/`load_bundle` (Booster text format, immutable versioned directories), frozen-level reapplication, schema hashing, `LoadedModel.predict()` with quantile sorting, per-booster transform inversion (asinh quantiles inverted, mean booster untouched, matching the monthly-bias fix), zero floor, conformal margins with pooled fallback |
| `catpipe/min_history.py` | History demand computed from feature names. Current binding feature: `price_drift_lag1` at 42 days for h=1; a deeper roll raises the demand automatically |
| `catpipe/ledger.py` | Append-only prediction ledger on `(run_date, *group_keys, model_version, scored_at)`, latest-wins `current_view()` that refuses to blend model versions, per-version watermark that cannot reverse |
| `catpipe/extract_window.py` | Windowed extract for serving, legitimate now Q1 is resolved. Reuses `extract.py`'s paging and dtype policy; snapshot layout unchanged, so `load_snapshot` and every builder read it as-is. Window recorded in the manifest |
| `catpipe/score_pipeline.py` | The `--score` entry point, mirroring `run_pipeline.py`'s CLI shape. Fail closed: `DataQualityError` aborts (manifest written, then raise, so the scheduler sees the failure); snapshot shorter than the card's features demand is refused; scoreable = `data_regime == featured_gated` and `gate_state == gated_complete` (C8), above the watermark |

## New tests

`tests/helpers_ops.py` builds a synthetic four-table snapshot in
`extract.py`'s landed shapes and carries a mini feature factory as a
stand-in for your local `feature_factory`; `tests/test_persistence.py`
carries the round-trip identity test (bit-identical predictions across
save/load), the absent-pool test (dropping pool_0 must not change pool_1's
predictions), and the schema/null guards; `tests/test_min_history_and_ledger.py`
pins the 42-day computation and the ledger semantics, plus the
ragged-window lag hazard as a demonstration to port against
`build_pool_frame` with a hole punched in `run_status`;
`tests/test_score_pipeline.py` runs the whole path through your real
builders, including the C8 test that breaking one day's Attribution run
drops exactly that day's pools from the scored set.

One fix the tests forced during the build, worth knowing: the never-null
guard originally checked `dtype.startswith("int")`, which misses pandas'
nullable `Int64`, the one integer dtype that can actually go null. The
test caught it; the guard now lowercases first.

## Wiring into the local tree

Three adjustments when merging into the current local state, none
structural:

1. Register the local frame builders in `score_pipeline.FRAME_BUILDERS`
   (`frame_1a`, `frame_1b`, `frame_2` alongside `pool`/`team`), and pass
   the tier filter through: `load_snapshot(dir, filter_tiers=card.filter_tiers)`
   where the local signature carries the flag.
2. `QuantileGBM.save()` becomes a few lines over `save_bundle()`: pass its
   four boosters, its own frozen levels into `categorical_levels`, and the
   frame's `attrs` (target, features, categoricals, group keys) into the
   card. Run the round-trip identity test against the real class before
   anything else.
3. `--featurize` takes the local factory's frame-level function
   (`module:function`); the mini factory in `tests/helpers_ops.py` shows
   the expected shape (frame in, featured frame out, factory does its own
   sorting and grouping).

## Usage

    # Windowed extract then score (credentials stay in CAT_DSN):
    python -m catpipe.score_pipeline --extract --days 120 --out ./snapshots \
        --bundle ./models/pool/v2026-08-01 --ledger ./ledger --frame pool \
        --featurize catpipe.feature_factory:featurize_pool

    # Score an existing snapshot:
    python -m catpipe.score_pipeline --snapshot ./snapshots/<ts> \
        --bundle ./models/pool/v2026-08-01 --ledger ./ledger --frame pool

## Deliberately not decided here

Retraining cadence and authority, the staleness rule under the migration,
feature-drift monitoring, detector state across model versions, and the
month-to-date publication product. Each lands on a seam this delivery
creates: staleness reads the manifest's unseen counts and the card's
training window; drift and the detector read `current_view(model_version)`;
a retrain writes a new bundle directory and a fresh watermark.
