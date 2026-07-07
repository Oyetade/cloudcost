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

`pytest tests/` — 51 tests. The ones that matter:

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

## Not yet built (deliberately)

Lagging and calendar-lead feature construction (kept out of the skeleton so
the join/gate logic stays legible), the non-pool residual (product 1b), and
the anomaly detector — all of which reuse these frames.
