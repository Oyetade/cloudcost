# catpipe

Cost Attribution Tool (Postgres) → training-frame pipeline. Turns the ten
`cat` schema tables into the Appendix A daily training frames, implementing
the section 7 validation programme as enforced code.

## Why this shape

The firm allows read-only SELECT on Postgres, no views, and no new software
beyond `sqlalchemy`, `psycopg`, `pandas`, `numpy`, `pyarrow`. So the
transformation happens outside the database, in pandas, and the referential
integrity a SQL engine would have guaranteed is instead **asserted after
every join** (`assertions.py`). At 3 GB this is comfortable.

## Pipeline

```
Postgres (read-only)
  │  extract.py — pages raw_cost & job_usage by run_date; eight small
  │              tables pulled whole; dtypes fixed once; snapshot-stamped
  ▼
Parquet snapshot  (snapshots/<UTC timestamp>/)
  │  transform.py — gate (7.3) → five-key join (7.5) → priceable mask →
  │                 daily aggregation, every join followed by an assertion
  ▼
Training frames   (pool 1a, team 2)  ← mlfin.data reads these
```

Until the append-only-vs-restated question closes (section 6), the extract
pulls **full tables, not deltas**, and frames are **full rebuilds**. When it
closes, only the extract predicate changes; nothing downstream moves.

## Module map

| Module | Responsibility | Section |
|---|---|---|
| `extract.py` | Postgres → Parquet, paged by run_date | build order |
| `assertions.py` | Reusable invariants; raise `DataQualityError` | 7.1, 7.5, 7.3 |
| `features.py` | Concurrency sweep, job-mix | 7.4 |
| `transform.py` | Gate, join, mask, frame assembly | 7.3, 7.5, 5.4 |
| `feature_factory.py` | Lags, rolls ending at t-1, calendar leads, gate-aware padding, price drift | 5.5, 7.3, 7.4, A.2 |
| `frames.py` | Model-ready frames: 1a (pool), 1b (non-pool segments), 2 (team) | Appendix A |
| `baselines.py` | F3 baselines: seasonal naive, 3-month trend (the incumbent), rolling 28-day median | F3, 5.6 |
| `harness.py` | Monthly walk-forward: honest per-product origins, quantile metrics, prediction ledger | 5.4, 5.7 |
| `models.py` | Quantile GBM (LightGBM): one class for A.1, A.2, A.3 | A.1-A.3, 7.3 |
| `calibrate.py` | Conformal (CQR) calibration: wrapper for any forecaster + ledger functions A.4 reuses | A.4 Layer 1 |
| `detector.py` | A.4: calibrated interval exceedance, CUSUM drift, job robust-z, attribution health, alert table | A.4 |
| `report.py` | Self-contained HTML performance report (matplotlib) | F3, charter 2.1 |
| `reconcile.py` | Pool-day job_cost vs raw_cost: the target-choice diagnostic | 3.3, target choice |

## Which cost is the physical target? (reconcile.py)

`raw_cost` is billed truth without lineage (no job_id, reaches to 2021);
`job_cost` is lineage without proven billed truth (native join to activity,
but attributed/derived and only from Aug 2023). At daily-pool grain the
missing job_id costs the pool model little, since job_id is aggregated away
either way. What decides the target is whether `job_cost` faithfully
re-expresses billed cost or is a reweighted/partial slice of it.

`reconcile.reconcile_pool_day` + `reconciliation_summary` answer this
empirically: aggregate both to pool-day, outer-join, and report the ratio,
the unattributed share, per-day dispersion, and a plain verdict. Three
outcomes: **faithful** (use job_cost as the 1a target from Aug 2023 for the
cleaner join, raw_cost supplies the pre-2023 tail); **partial** (raw_cost
wins for 1a, job_cost stays the team target only); **close-but-loose** (prefer
raw_cost, inspect the largest-divergence pool-days first). This is a query you
can run today, without waiting on the open question of how job_cost is derived.

## Data regimes and the three-state gate (section 5.7)

The two products do not share a training window, and coverage differs
*within* the pool frame. Every frame is stamped with `data_regime`:

- `pre_coverage` — before Aug 2022. The raw_cost onboarding ramp: monthly
  row count and monthly total cost climb together through 2021 and flatten
  around Aug 2022, so early totals **understate** the true estate. Labelled
  and then dropped from training frames by `drop_pre_coverage`; kept in the
  raw extract.
- `cost_only` — Aug 2022 to Aug 2023. Representative cost target only;
  activity features are null **by construction** and must never be imputed
  to zero (zero activity is a real value in the featured era and a lie here).
- `featured_ungated` — Aug 2023 to Jan 2024. Featured, but run_status did
  not yet exist, so no gate could be evaluated.
- `featured_gated` — Jan 2024 on. Featured and gated.

Boundaries are single tunables (`COST_HISTORY_START`, `ACTIVITY_START`,
`RUN_STATUS_START`). The pool cost target is also **non-stationary in
level** — it peaks in 2023 and settles to a lower plateau after — so
level-based models should prefer differenced targets or explicit trend
features; the incumbent 3-month-trend baseline tolerates this by design.

The gate is correspondingly three-state (`gate_state`): a missing run_status
row means **fail** on/after the run_status era (unverified load, excluded)
but **ungated** before it (logging did not exist, kept and labelled). The
transform only labels; the **model layer** selects its window by filtering
`data_regime` — baselines use all three, the first boosted model uses
`featured_gated` alone. Back-test origin follows from the regime, per
product, so it cannot dishonestly start in 2023 for a gated model.

## What the tests cover

`pytest tests/` — 174 tests. The ones that matter:

- **Concurrency sweep** — overlap, sequential, instantaneous handover (no
  false peak), independent pools, null-time drop, triple overlap.
- **Gate (three-state)** — all-three-complete passes; missing type, errored
  type, any unknown status fail safe; latest run per type wins; a missing
  row inside the era fails, before the era is kept as ungated; an explicit
  incomplete before the era still fails.
- **Regime** — all four regimes assigned by boundary, half-open on the
  left; pre_coverage dropped from frames; cost_only activity stays null,
  never zero, end-to-end.
- **Five-key join** — clean join preserves row count; retried job on the
  usage side is caught; duplicate on the cost side does not multiply usage;
  usage orphans get Unknown, not dropped; same-day cost never crosses.
- **Priceable mask** — zero-cost and zero-usage rows excluded.
- **Team frame** — NULL team kept distinct from the Unknown category.
- **Density** — monthly row-count histogram exposes a sparse backfill tail.
- **Integration** — the full pipeline gates, masks, joins and labels
  regimes end-to-end, across cost_only and featured_gated rows.

## Open items that touch this code

- `spot_placement_scores.date_time` parsed as UTC pending confirmation.
- `run_status.run_time` is time-of-day, not elapsed; confirm hourly-run
  handling in the gate against real values.
- Whether one month of raw_cost is itself under the connection limit; if
  not, drop `_extract_paged` granularity to fortnightly or per-subscription.
- Whether raw_cost's 2021-2022 is continuous or a thin backfill: read
  `density` in the extract manifest. If sparse, move `ACTIVITY_START`'s
  cost-only floor to the observed density cliff rather than nominal 2021.
- Concurrency is a machine-count *proxy*; if a truer node count exists it
  should replace the sweep.

## The ML frames (frames.py, July 2026)

`--ml-frames` builds the three Appendix A training frames on top of the
transform layer, writing one parquet per frame plus `ml_manifest.json`, which
declares each frame's target, feature list and categoricals — the single
place a reviewer looks to see exactly what each model is allowed to know.

- **frame_1a** (pool): raw_cost pool target x job_usage activity through the
  five-key join, with concurrency and job-mix now wired in (they were
  computed but never merged), padded gate-aware per pool, lagged and rolled,
  price-drift per pool, enriched from environment_config (tier, sub-tier,
  derived neu/weu region).
- **frame_1b** (non-pool, rebuilt per A.2): the null-pool slice segmented
  into vm_compute / platform, aggregated per subscription-segment-day on a
  SHARED date spine, guarded by the partition identity (pool branch +
  non-pool branch = raw_cost grand total, asserted before anything is
  dropped), full-row duplicate assertion plus a candidate-key duplicate
  REPORT pending Q7's stable resource identifier. Effective-price drift
  (14d vs prior 28d, catches steps and glides) replaces the repr_30d flag.
  `post_glide` marks the 2025-02-01 origin; `training_slice_1b` selects it.
- **frame_2** (team): job_cost target per team-day on a complete day x team
  grid (sum over teams = total attributed cost, asserted), NULL team kept
  distinct as __NULL_TEAM__, activity per team via the five-key join,
  category mix shares, and `unknown_pct` per day — one computation feeding
  both the A.3 frame filter (`filter_unknown`) and the A.4 rule.

Every feature is lagged or calendar-known, enforced by
`assert_no_same_day_cost` on the declared feature list. The append-only
invariant (Q1) is re-checked on every snapshot via
`assert_one_write_per_slice` — the tripwire for the day the loader's upsert
fires.

## Baselines and the harness (baselines.py / harness.py, July 2026)

`--backtest` runs the F3 trio (3-month trend reproducing the incumbent,
seasonal naive, rolling 28-day median) through a single monthly walk-forward
harness, per frame, each on its own honest window: frame 1a on
featured_gated, frame 1b additionally masked to post_glide, frame 2
additionally masked to unknown_pct <= 0.20. Baselines FREEZE AT THE ORIGIN
(no peeking inside the test month — direct horizons per 5.4) and carry
empirical intervals from their own in-training residuals, so the harness is
quantile-shaped before any GBM exists. An origin whose training window would
predate the frame's honest regime is refused with an error, never silently
accepted. Metrics per model: daily MAE; THREE monthly errors answering three
questions — monthly_pct_err_estate (|sum of all predictions - sum of all
actuals| per month: what finance sees, the incumbent-comparison number),
monthly_wape (spend-weighted absolute error: attribution accuracy, where
offsetting group errors do not cancel), and monthly_pct_err (unweighted
mean over group-month ratios: a small-group diagnostic that tiny pools
inflate, never the headline); pinball loss at 5/50/95; and 5-95 interval
coverage against the 0.90 target. Outputs: `backtest_summary.json` and one
prediction ledger parquet per frame — keep the ledger, since metrics can be
recomputed from it but not the reverse. Any future model that implements
fit / predict-quantiles runs through the identical folds.

## The quantile GBM (models.py, July 2026)

`--gbm` (implies `--backtest`) adds `QuantileGBM` to the model list: three
LightGBM boosters with quantile objectives at 5/50/95, built per frame from
the frame's OWN declared feature list (`QuantileGBM.for_frame`), so the
manifest remains the single statement of what the model knows. Native
categoricals with levels frozen at fit (unseen categories map to missing,
never a fresh code); asinh target transform by default because padded
frames legitimately contain zero-cost days (quantiles invert exactly under
monotone transforms); chronological-tail early stopping, then a refit on
the full window at the stopped round; per-row quantile sorting so intervals
never cross; predictions floored at zero. `feature_importance()` exposes
gain on the median booster — the first instalment of the charter's
'explain' verb.

The GBM's known calibration gap (quantile GBMs under-cover; observed ~0.82 vs the 0.90
target on synthetics) is the detector's business: conformal widening sits
on top of the harness's coverage numbers when A.4 Layer 1 is built.

## Conformal calibration (calibrate.py, July 2026)

Quantile GBMs systematically under-cover (~0.82 observed against the 0.90
target); uncorrected, A.4 Layer 1 would fire nearly twice its design rate
and the alert table would decay into ignored noise. calibrate.py implements
CQR (conformalised quantile regression): score how far each held-out actual
escaped its interval — max(q05 - y, y - q95), negative when comfortably
inside — take the finite-sample (n+1)-corrected quantile of the scores, and
widen (or, for an over-wide model, SHRINK) every future interval by that
margin. Scaled by default, so a volatile wide-band day receives more
absolute widening than a quiet one; per-group margins where the calibration
window is dense, pooled fallback where thin; the median is never touched.
Split-conformal gives at least the target coverage in finite samples
regardless of the model's miscalibration — the same assert-don't-assume
move as the gate and the partition identity, applied to probability claims.

Two entry points, one implementation: `ConformalWrapper` wraps any harness
forecaster (chronological calibration tail at fit, widened intervals at
predict, `calibration_report` kept for the audit trail); `conformal_margins`
/ `apply_margins` are pure ledger functions, which is what A.4 Layer 1
reuses nightly against the trailing production ledger before the exceedance
rule runs. `--gbm` now backtests the raw and calibrated GBM side by side so
the coverage correction is visible on identical folds; observed on the
quarter-end synthetic: identical point accuracy, coverage 0.816 -> 0.868,
pinball_95 slightly improved (the widening is information, not padding).

## The detector (detector.py, July 2026)

`--detect` (implies `--backtest`) runs A.4 over the backtest ledgers and the
snapshot, writing `alerts.parquet`. Four layers, one table:

Layer 1, calibrated interval exceedance: for each scored day, conformal
margins are recomputed from the trailing 90-day ledger window (the
calibrate.py reuse, margins STRICTLY PAST so a spike never softens its own
alarm) and the actual is tested against the calibrated band; severity is
the scaled exceedance. Layer 1.5, CUSUM drift: residuals standardised by a
trailing robust (MAD) sigma feed a two-sided CUSUM (k=0.5, h=5), catching
the December-2024 failure mode — a glide that never breaches a daily
interval; the statistic resets after each alarm so a continuing glide says
'still drifting' repeatedly. In-control false-alarm budget ~0.8/year per
series by design. Layer 2, job robust-z: per (job_name, pool), today's cost
against the trailing median/MAD (never mean/std — the history contains
exactly the spikes the profile must not learn to expect), with a new-job
informational alert on first appearance and an absolute floor suppressing
penny alerts. Attribution health: days whose unknown_pct (frame 2's column)
exceeds 20% alert regardless of totals, because the completeness gate is
blind to attribution failure.

Every alert has a stable alert_id, severity, human-readable message and a
STATUS column from day one; merge_alert_status preserves triage states
('expected', 'investigating', ...) across re-scores, because an alert table
nobody can triage decays into noise within a month. Injected-anomaly
validation: spike caught same-day by both L1 and CUSUM; glide first alarmed
7 days after onset; 1 false CUSUM alarm in 300 stationary days; L1
normal-period escape rate 8.7% against the 10% design.

## Not yet built (deliberately)

Production serving (nightly scoring against live predictions rather than
backtest replay), the scenario engine (M5, deferred), and the triage UI —
the alert table's status column is its contract.

## Gotchas learned from real data

**Categorical keys + groupby = cartesian fanout.** pandas `groupby` on
categorical columns defaults to `observed=False`, which emits one group per
combination of *all category levels* whether or not they occur — exploding
4.66M raw_cost rows into 122M phantom pool-days. Fixed two ways: join/group
KEY columns are kept as plain strings in the extract (only descriptive
low-cardinality columns are categoricals), and every groupby/pivot passes
`observed=True` as belt-and-braces. Regression-tested.

**~76% of raw_cost has null pool.** batch_account_name and pool_name are null
on ~3.55M of 4.66M rows: these are non-batch estate spend, not pool
workloads. `daily_cost_by_pool` correctly filters to `pool_name.notna()`, so
the pool frame covers only the batch minority (~24% of rows). The other ~76%
is the non-pool residual (product 1b), still to be built. The reconciliation
ratio of ~0.34 was job_cost (batch-only) compared against all-estate
raw_cost — a scope mismatch, not attribution loss. The fair comparison is
job_cost against the batch (pool-not-null) slice only.

**The raw_cost duplicate check runs on the batch slice only.** The pool-key
is a batch-shaped identity; non-batch rows (storage, network) carry null
pool and legitimately repeat on it — a storage account bills many "Read
Operations" lines a day. A whole-table check flagged 310k such benign rows.
Since only pool-not-null rows feed the pool target, the check now runs on
exactly that slice; genuine duplicates within the batch slice still raise.
Non-batch raw_cost has a finer, resource-based grain and its own duplicate
check belongs with product 1b.

## Running it

Install the approved stack, then either extract-and-build or build from an
existing snapshot:

```bash
pip install pandas numpy pyarrow sqlalchemy psycopg pytest

# tests only (no database):
PYTHONPATH=. python -m pytest tests/ -q

# full pipeline from Postgres (read-only SELECT):
export CAT_DSN="postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME"
PYTHONPATH=. python -m catpipe.run_pipeline --extract --out ./snapshots

# build frames from an existing snapshot (skips the database):
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<timestamp>

# frames plus the three model-ready ML frames and their feature manifest:
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --ml-frames

# ...and the F3 baseline walk-forward on every frame (implies --ml-frames):
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --backtest

# ...adding the quantile GBM to the same folds:
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --gbm

# ...and the A.4 detector over the ledgers + snapshot:
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --detect

# ...plus a self-contained HTML report with charts (combinable with the above):
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --detect --report

# just the target-choice reconciliation:
PYTHONPATH=. python -m catpipe.run_pipeline --snapshot ./snapshots/<ts> --reconcile-only
```

Outputs land in `<snapshot>/frames/`: `frame_pool.parquet`,
`frame_team.parquet`, `reconciliation_pool_day.parquet`, and
`recon_summary.json` (which carries the target-choice verdict). The pipeline
prints a short report including the reconciliation verdict on completion.
