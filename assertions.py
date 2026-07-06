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


def assert_gate_complete(gated: pd.DataFrame, context: str) -> None:
    """The run_status gate (7.3): every row entering training must sit on a
    (run_date, subscription) slice where Cost, Usage AND Attribution all read
    Complete. This asserts the gate column is all-True after filtering; a
    False slipping through means the filter was applied wrongly.
    """
    if "gate_complete" not in gated.columns:
        raise DataQualityError(f"{context}: gate_complete column missing")
    if not gated["gate_complete"].all():
        n = int((~gated["gate_complete"]).sum())
        raise DataQualityError(
            f"{context}: {n} rows passed the gate with gate_complete=False"
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
