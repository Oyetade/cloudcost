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

## What the tests cover

`pytest tests/` — 37 tests. The ones that matter:

- **Concurrency sweep** — overlap, sequential, instantaneous handover (no
  false peak), independent pools, null-time drop, triple overlap.
- **Gate** — all-three-complete passes; missing type, errored type, and any
  unknown status all fail safe; latest run per type wins.
- **Five-key join** — clean join preserves row count; retried job on the
  usage side is caught; duplicate on the cost side does not multiply usage;
  usage orphans get Unknown, not dropped; same-day cost never crosses.
- **Priceable mask** — zero-cost and zero-usage rows excluded.
- **Team frame** — NULL team kept distinct from the Unknown category.
- **Integration** — the full pipeline gates, masks and joins end-to-end.

## Open items that touch this code

- `spot_placement_scores.date_time` parsed as UTC pending confirmation.
- `run_status.run_time` is time-of-day, not elapsed; confirm hourly-run
  handling in the gate against real values.
- Whether one month of raw_cost is itself under the connection limit; if
  not, drop `_extract_paged` granularity to fortnightly or per-subscription.
- Concurrency is a machine-count *proxy*; if a truer node count exists it
  should replace the sweep.

## Not yet built (deliberately)

Lagging and calendar-lead feature construction (kept out of the skeleton so
the join/gate logic stays legible), the non-pool residual (product 1b), and
the anomaly detector — all of which reuse these frames.
