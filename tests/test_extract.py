"""Tests for extract.py pure logic (no database dependency)."""

from datetime import date

import pandas as pd

from catpipe import extract as E


class TestMonthStarts:
    def test_spans_partial_months_inclusive(self):
        assert E._month_starts(date(2021, 1, 15), date(2021, 4, 3)) == [
            date(2021, 1, 1), date(2021, 2, 1),
            date(2021, 3, 1), date(2021, 4, 1),
        ]

    def test_single_month(self):
        assert E._month_starts(date(2024, 6, 10), date(2024, 6, 20)) == [
            date(2024, 6, 1)
        ]

    def test_year_boundary(self):
        assert E._month_starts(date(2023, 12, 1), date(2024, 1, 1)) == [
            date(2023, 12, 1), date(2024, 1, 1)
        ]


class TestNextMonth:
    def test_december_rolls_to_january(self):
        assert E._next_month(date(2025, 12, 1)) == date(2026, 1, 1)

    def test_mid_year(self):
        assert E._next_month(date(2024, 3, 1)) == date(2024, 4, 1)


class TestMonthlyDensity:
    def test_counts_rows_per_month(self):
        df = pd.DataFrame({
            "run_date": [date(2021, 1, 5), date(2021, 1, 20),
                         date(2021, 3, 1)],
        })
        d = E.monthly_density(df)
        assert d == {"2021-01": 2, "2021-03": 1}

    def test_reveals_sparse_backfill(self):
        # one stray 2021 row then dense 2023 => the histogram exposes it
        df = pd.DataFrame({
            "run_date": [date(2021, 6, 1)] + [date(2023, 5, d)
                                              for d in range(1, 21)],
        })
        d = E.monthly_density(df)
        assert d["2021-06"] == 1
        assert d["2023-05"] == 20

    def test_empty_frame(self):
        assert E.monthly_density(pd.DataFrame()) == {}
