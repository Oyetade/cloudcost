"""
assertions.py  --  the checks that make section 7 enforced, not aspirational.

Every join in the transform is followed by one of these. They raise
DataQualityError on violation so the pipeline fails loudly rather than
producing a silently corrupted frame. This is the code that replaces the
referential-integrity guarantees a SQL engine would have given us; since we
do the relational work in pandas, we assert what the database used to prove.
"""

from __future__ import annotations

import pandas as pd


class DataQualityError(AssertionError):
    """Raised when a section-7 invariant is violated."""


def assert_no_duplicates(df: pd.DataFrame, keys: list[str], table: str) -> None:
    """raw_cost has no PK (7.1): duplicates are structurally possible.

    Also guards the paged extract against a month boundary landing the same
    row twice.
    """
    dup = df.duplicated(subset=keys, keep=False)
    if dup.any():
        n = int(dup.sum())
        sample = df.loc[dup, keys].drop_duplicates().head(3).to_dict("records")
        raise DataQualityError(
            f"{table}: {n} duplicate rows on {keys}. Examples: {sample}"
        )


def assert_row_count_identity(
    result: pd.DataFrame, expected: int, context: str
) -> None:
    """Validation two (7.5): after the five-key left join, row count must
    equal job_usage's. Growth means the key is not unique (retried jobs
    sharing an identifier within a day).
    """
    if len(result) != expected:
        raise DataQualityError(
            f"{context}: row count {len(result)} != expected {expected}. "
            "Key is not unique on one side; a tiebreaker (start_time / "
            "update_time) is needed before this join is safe."
        )


def report_anti_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    left_name: str,
    right_name: str,
) -> dict[str, int]:
    """Validation one (7.5): orphans both ways. Returns counts rather than
    raising, because orphans in job_usage are expected (they get Unknown);
    the caller decides what is tolerable. Uses indicator to find them.
    """
    merged = left[keys].merge(
        right[keys].drop_duplicates(), on=keys, how="outer", indicator=True
    )
    return {
        f"{left_name}_only": int((merged["_merge"] == "left_only").sum()),
        f"{right_name}_only": int((merged["_merge"] == "right_only").sum()),
        "both": int((merged["_merge"] == "both").sum()),
    }


def assert_no_failed_gate(gated: pd.DataFrame, context: str) -> None:
    """The run_status gate is three-state (section 5.7): gated_complete,
    gated_failed, ungated. Rows that FAILED verification (a load on/after the
    run_status era with no Complete record) must never survive into a frame.
    ungated rows (before the era, no run_status to check) are allowed through
    deliberately and carry their own label. This asserts no gated_failed row
    slipped past the filter.
    """
    if "gate_state" not in gated.columns:
        raise DataQualityError(f"{context}: gate_state column missing")
    failed = gated["gate_state"].eq("gated_failed")
    if failed.any():
        raise DataQualityError(
            f"{context}: {int(failed.sum())} gated_failed rows survived the "
            "gate; verification was applied wrongly"
        )


def assert_no_same_day_cost(feature_cols: list[str], context: str) -> None:
    """Never bring across same-day cost (7.5): it is the target through a
    side door. Any feature column literally named 'cost' (unlagged) is a
    leak. Lagged cost (cost_lag_1 etc.) is fine and must NOT trip this.
    """
    leaks = [c for c in feature_cols if c == "cost" or c == "pre_tax_cost"]
    if leaks:
        raise DataQualityError(
            f"{context}: unlagged cost columns in feature set: {leaks}. "
            "These are the target arriving through a side door."
        )


def assert_partition_identity(
    part_totals: dict[str, float], grand_total: float, context: str,
    abs_tol: float = 0.05, rel_tol: float = 1e-6,
) -> None:
    """The partition check (session note section 6, the 62.43 lesson): any
    decomposition of raw_cost must sum back to the table's grand total. A
    partition that silently drops rows (an orphan day, a fanout, a filter
    applied to one side only) shows up here and nowhere else. Applies to the
    1a + 1b split in particular: pool branch + non-pool branch = everything.
    """
    total = float(sum(part_totals.values()))
    gap = abs(total - float(grand_total))
    tol = max(abs_tol, rel_tol * abs(float(grand_total)))
    if gap > tol:
        raise DataQualityError(
            f"{context}: partition does not sum to the grand total. "
            f"parts={ {k: round(v, 2) for k, v in part_totals.items()} } "
            f"sum={total:.2f} vs total={float(grand_total):.2f} "
            f"(gap {gap:.2f} > tol {tol:.2f}). A branch is dropping or "
            "double-counting rows."
        )


# The true row grain of raw_cost (registers Q24 and Q22, 14 July 2026).
# Declared once and shared, because the previous drift between two hand-written
# copies of this list IS Q24: assert_no_duplicates omitted meter_sub_category
# and so flagged 142 legitimate Azure rows (Files vs Tables under Storage /
# Read Operations) as duplicates of each other.
RAW_COST_GRAIN = [
    "run_date", "subscription_id", "resource_group_name", "resource_type",
    "meter", "meter_sub_category", "batch_account_name", "pool_name",
]


def grain_present(df: pd.DataFrame, keys: list[str] | None = None) -> list[str]:
    """RAW_COST_GRAIN restricted to the columns a frame actually carries.

    The live extract has all eight. Narrower frames (and test fixtures) may
    not, and a check that KeyErrors on a missing column is a check that gets
    deleted rather than fixed. Narrowing the key can only make a duplicate
    check STRICTER, never blinder, so this is safe in the direction that
    matters: a frame without meter_sub_category simply reproduces the old
    Q24 behaviour on that frame, and the live path — which has the column —
    gets the corrected grain.
    """
    keys = keys if keys is not None else RAW_COST_GRAIN
    return [k for k in keys if k in df.columns]


def assert_one_write_per_slice(
    raw_cost: pd.DataFrame, context: str = "raw_cost",
    keys: list[str] | None = None,
    ts_col: str = "update_time",
) -> None:
    """The append-only invariant (question register Q1, falsified 14 July
    2026): every ROW of raw_cost has exactly one update_time — one write,
    never rewritten. Back-tests being point-in-time correct, and the detector
    needing no maturity rule, both REST on this.

    The grain matters, and getting it wrong was Q22's original sin. This check
    keyed on (run_date, subscription_id) until 14 July 2026, which is not a
    row: it is a coarse bucket holding thousands of rows across resource
    groups, types, meters and sub-categories. A loader that writes a
    subscription's cost lines at different moments — which every loader does —
    produces many update_times per bucket with nothing rewritten at all. So
    the old check conflated two unrelated things:

      rewrite:      the same row written twice, different update_time. Q1.
      many rows:    different rows, written at different times. Normal.

    It could not tell them apart, so its count (829 unfiltered, 484 after the
    DEV/TEST filter) was an upper bound of unknown tightness. Keying on the
    full row grain removes the conflation: more than one update_time on one
    ROW is a rewrite and nothing else.

    dropna=False is not optional here. pandas' groupby drops null keys by
    default, and batch_account_name, pool_name and meter_sub_category are all
    nullable. Without it this check would go quiet on most of the estate and
    look like it had passed: that silent-drop default is exactly what cost
    17,585.93 in Q23.
    """
    if ts_col not in raw_cost.columns or raw_cost.empty:
        return
    keys = list(keys) if keys is not None else RAW_COST_GRAIN
    present = [k for k in keys if k in raw_cost.columns]
    if not present:
        return
    n_ts = raw_cost.groupby(present, observed=True,
                            dropna=False)[ts_col].nunique()
    rewritten = n_ts[n_ts > 1]
    if len(rewritten):
        sample = rewritten.head(3).index.tolist()
        raise DataQualityError(
            f"{context}: {len(rewritten)} rows of raw_cost carry more than "
            f"one {ts_col} at the full grain ({', '.join(present)}) — the "
            f"loader's upsert has fired and the append-only assumption (Q1) "
            f"no longer holds. Point-in-time back-test claims are void for "
            f"any origin whose feature window overlaps these rows. "
            f"Examples: {sample}"
        )


def report_duplicate_rate(
    df: pd.DataFrame, keys: list[str], table: str
) -> dict:
    """Duplicate REPORT, not assertion, for grains that are still an open
    question. The 1b candidate key (run_date, subscription, resource_group,
    resource_type, meter) may legitimately repeat until the SMEs name the
    stable resource identifier (register Q7): two storage accounts in one
    resource group bill the same meter on the same day. The daily SUM is
    robust to that ambiguity; what it is not robust to is literal
    double-loading, which assert_no_duplicates on the full row guards
    separately. This reports the rate so the frame carries the number and
    Q7's answer can be checked against it.
    """
    dup = df.duplicated(subset=keys, keep=False)
    n = int(dup.sum())
    return {
        "table": table,
        "keys": keys,
        "rows": int(len(df)),
        "duplicate_rows": n,
        "duplicate_rate": (n / len(df)) if len(df) else 0.0,
        "examples": (
            df.loc[dup, keys].drop_duplicates().head(3).to_dict("records")
            if n else []
        ),
    }
