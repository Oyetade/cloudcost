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

## Additions after the frame_1a feature list

Two behaviours added once the real 40-feature list was visible (19 of them
share_*_lag1 pivot columns):

**Pivot reconciliation** (`persistence.reindex_pivot_features`, wired into
`LoadedModel.predict`). The job-mix pivot only creates a column per category
observed in the window, so a quiet fortnight with no Audit jobs produces a
frame with no `share_Audit_lag1` at all. The card now declares
`zero_fill_prefixes=["share_"]`; a card-listed feature under a declared
prefix that is absent from the inference frame is filled with 0.0 (absence
of the category and a zero share are the same fact, and a test proves the
predictions are identical), while any other missing feature still refuses.
The mirror case, a NEW category's column appearing since training, is
reported in the manifest as `novel_pivot_columns`, the column-space
equivalent of an unseen pool level. Set `zero_fill_prefixes` when building
the card; without it, behaviour is unchanged.

**Card-level leakage assertion** (`persistence.assert_no_unlagged_features`).
`assertions.assert_no_same_day_cost` only catches literal cost columns; the
charter's rule covers same-day activity too. This checks the card's
feature_names against the frame's columns: any feature whose lagged twin
exists in the frame (job_seconds vs job_seconds_lag1, share_Audit vs
share_Audit_lag1, price_drift vs price_drift_lag1) is a same-day observation
arriving through a side door, and raises. Calendar and static columns have
no lagged twins and pass. Call it at card-build time, before save_bundle;
the real frame_1a list passes it as-is.

## Closing the Copilot-identified gap (D7/O9, per-alert attribution, Explain)

Two further modules close the review's "not found" items:

**`catpipe/change_decomposition.py`** answers "why cost changed:
price/usage/scope". Bennet (midpoint) decomposition per meter between two
periods: exactly additive, price_effect + usage_effect + scope_effect +
unpriced_effect = delta to the penny, asserted per item and in total in the
assertions.py spirit. Entering/exiting meters are scope; cost with zero
recorded usage in both periods is labelled unpriced, never hidden.
`day_vs_baseline` gives the per-alert view (one group's day against its
trailing per-day baseline) and `team_contribution` gives the team half of
the M4 wording (each team's delta versus its own baseline, shares summing
to the group change). No model, no ML: accounting arithmetic on raw_cost
and job_cost as extracted.

**`catpipe/explain.py`** joins the two halves per alert. MODEL half:
`local_attribution` uses LightGBM's pred_contrib (TreeSHAP) on the mean
booster only, so contributions read in pounds and sum exactly to pred_mean
(asserted per row); the quantile boosters' contributions live on the asinh
scale and are deliberately not reported. Attribution runs on
`LoadedModel.design_matrix`, the same matrix predict() scores, so the
explanation can never attribute a different matrix than the model saw.
BUSINESS half: the decomposition above. `explain_event` returns the
EVENT EXPLAIN structure (which meter drove it, how much was price/usage/
scope, which features drove this prediction, team contribution) and
`format_explanation` renders the plain-text block for the report; the HTML
report wiring is the remaining step, in the local tree's report module.

## Deliberately not decided here

Retraining cadence and authority, the staleness rule under the migration,
feature-drift monitoring, detector state across model versions, and the
month-to-date publication product. Each lands on a seam this delivery
creates: staleness reads the manifest's unseen counts and the card's
training window; drift and the detector read `current_view(model_version)`;
a retrain writes a new bundle directory and a fresh watermark.
