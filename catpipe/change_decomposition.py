"""
change_decomposition.py  --  why cost changed: price, usage, scope (D7/O9).

No model here. Cost at any grain is a sum over items (meters) of
price x usage, with price derived as cost / usage. The change in cost
between two periods therefore decomposes exactly, per item, with the
Bennet (midpoint) form:

    price_effect = (p_b - p_a) * (q_a + q_b) / 2
    usage_effect = (q_b - q_a) * (p_a + p_b) / 2

which is exactly additive: price_effect + usage_effect = delta for every
continuing item, no residual, no approximation. Items present in only one
period are scope: an entering item's whole cost is scope_effect, an
exiting item's is negative scope_effect. Items carrying cost with zero
recorded usage in both periods have no derivable price; their delta goes
to unpriced_effect and is labelled, never hidden, the same honesty rule
as the Unknown team.

The module-level identity is asserted, in the assertions.py spirit:

    price + usage + scope + unpriced = cost_b - cost_a   (to the penny)

Team contribution answers the M4 wording's other half (teams with unusual
consumption): the same two-period comparison on job_cost by team, each
team's share of the total change, against its own baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .assertions import DataQualityError

PENNY = 0.01


@dataclass
class Decomposition:
    """Per-item effects plus the asserted totals."""

    items: pd.DataFrame          # per-item effects, kind-labelled
    total_a: float
    total_b: float

    @property
    def delta(self) -> float:
        return self.total_b - self.total_a

    def summary(self) -> dict:
        s = self.items[["price_effect", "usage_effect",
                        "scope_effect", "unpriced_effect"]].sum()
        return {
            "cost_before": round(self.total_a, 2),
            "cost_after": round(self.total_b, 2),
            "delta": round(self.delta, 2),
            "price_effect": round(float(s["price_effect"]), 2),
            "usage_effect": round(float(s["usage_effect"]), 2),
            "scope_effect": round(float(s["scope_effect"]), 2),
            "unpriced_effect": round(float(s["unpriced_effect"]), 2),
        }

    def top_items(self, n: int = 5) -> pd.DataFrame:
        by_size = self.items.reindex(
            self.items["delta"].abs().sort_values(ascending=False).index
        )
        return by_size.head(n)


def decompose(
    period_a: pd.DataFrame,
    period_b: pd.DataFrame,
    item_keys: list[str] | None = None,
    cost_col: str = "pre_tax_cost",
    usage_col: str = "usage_quantity",
) -> Decomposition:
    """Bennet decomposition of the cost change between two periods.

    period_a / period_b are raw_cost-shaped slices (any number of days
    each; multi-day periods are aggregated per item first, so a day
    against a 28-day baseline compares totals, and the caller scales the
    baseline if a per-day view is wanted, see day_vs_baseline).
    """
    item_keys = item_keys or ["meter"]

    def agg(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[*item_keys, cost_col, usage_col])
        return (df.groupby(item_keys, dropna=False, observed=True)
                  [[cost_col, usage_col]].sum().reset_index())

    a = agg(period_a).rename(columns={cost_col: "cost_a", usage_col: "usage_a"})
    b = agg(period_b).rename(columns={cost_col: "cost_b", usage_col: "usage_b"})
    m = a.merge(b, on=item_keys, how="outer")
    for c in ("cost_a", "usage_a", "cost_b", "usage_b"):
        m[c] = m[c].fillna(0.0)

    in_a = (m["cost_a"] != 0) | (m["usage_a"] != 0)
    in_b = (m["cost_b"] != 0) | (m["usage_b"] != 0)
    priced = (m["usage_a"] > 0) & (m["usage_b"] > 0)

    continuing = in_a & in_b & priced
    unpriced = in_a & in_b & ~priced
    entering = ~in_a & in_b
    exiting = in_a & ~in_b

    m["delta"] = m["cost_b"] - m["cost_a"]
    m["price_effect"] = 0.0
    m["usage_effect"] = 0.0
    m["scope_effect"] = 0.0
    m["unpriced_effect"] = 0.0
    m["kind"] = np.select(
        [continuing, unpriced, entering, exiting],
        ["continuing", "unpriced", "entering", "exiting"],
        default="empty",
    )

    if continuing.any():
        pa = m.loc[continuing, "cost_a"] / m.loc[continuing, "usage_a"]
        pb = m.loc[continuing, "cost_b"] / m.loc[continuing, "usage_b"]
        qa = m.loc[continuing, "usage_a"]
        qb = m.loc[continuing, "usage_b"]
        m.loc[continuing, "price_effect"] = (pb - pa) * (qa + qb) / 2.0
        m.loc[continuing, "usage_effect"] = (qb - qa) * (pa + pb) / 2.0
    m.loc[entering, "scope_effect"] = m.loc[entering, "cost_b"]
    m.loc[exiting, "scope_effect"] = -m.loc[exiting, "cost_a"]
    m.loc[unpriced, "unpriced_effect"] = m.loc[unpriced, "delta"]

    # The identity, per item and in total. Bennet is exact; a break here is
    # a data problem (e.g. negative usage) and must be loud.
    effects = (m["price_effect"] + m["usage_effect"]
               + m["scope_effect"] + m["unpriced_effect"])
    bad = (effects - m["delta"]).abs() > PENNY
    if bad.any():
        rows = m.loc[bad, item_keys + ["cost_a", "cost_b", "usage_a", "usage_b"]]
        raise DataQualityError(
            "decomposition identity broke on "
            f"{int(bad.sum())} item(s); first offenders:\n{rows.head()}"
        )

    total_a = float(m["cost_a"].sum())
    total_b = float(m["cost_b"].sum())
    if abs(effects.sum() - (total_b - total_a)) > PENNY:
        raise DataQualityError("decomposition total does not sum to delta")

    return Decomposition(items=m, total_a=total_a, total_b=total_b)


def day_vs_baseline(
    raw_cost: pd.DataFrame,
    run_date,
    group: dict[str, str],
    baseline_days: int = 28,
    item_keys: list[str] | None = None,
    cost_col: str = "pre_tax_cost",
    usage_col: str = "usage_quantity",
) -> Decomposition:
    """The per-alert convenience: one group's day against its trailing
    per-day baseline. The baseline period's totals are scaled to one day
    (sum / days observed) so both sides are a day and the effects read as
    'versus a typical recent day', which is what an alert asks.
    """
    run_date = pd.Timestamp(run_date)
    df = raw_cost
    for k, v in group.items():
        df = df[df[k] == v]
    dates = pd.to_datetime(df["run_date"])
    day = df[dates == run_date]
    base_mask = (dates < run_date) & (dates >= run_date - pd.Timedelta(days=baseline_days))
    base = df[base_mask].copy()
    n_base_days = pd.to_datetime(base["run_date"]).nunique()
    if n_base_days == 0:
        raise DataQualityError(
            f"no baseline days in the {baseline_days} days before "
            f"{run_date.date()} for group {group}"
        )
    base = base.copy()
    base[cost_col] = base[cost_col] / n_base_days
    base[usage_col] = base[usage_col] / n_base_days
    return decompose(base, day, item_keys=item_keys,
                     cost_col=cost_col, usage_col=usage_col)


def team_contribution(
    job_cost: pd.DataFrame,
    run_date,
    group: dict[str, str],
    baseline_days: int = 28,
    team_col: str = "job_team",
    cost_col: str = "cost",
) -> pd.DataFrame:
    """Each team's contribution to the day's change versus its own trailing
    per-day baseline: delta per team, share of the total change, and the
    ratio to the team's baseline. Teams entering or exiting appear with
    their full delta, the scope analogue. The deltas sum to the group's
    total change by construction.
    """
    run_date = pd.Timestamp(run_date)
    df = job_cost
    for k, v in group.items():
        df = df[df[k] == v]
    dates = pd.to_datetime(df["run_date"])
    day = df[dates == run_date]
    base_mask = (dates < run_date) & (dates >= run_date - pd.Timedelta(days=baseline_days))
    base = df[base_mask]
    n_base_days = pd.to_datetime(base["run_date"]).nunique()
    if n_base_days == 0:
        raise DataQualityError(
            f"no baseline days in the {baseline_days} days before "
            f"{run_date.date()} for group {group}"
        )
    day_by = day.groupby(team_col, dropna=False, observed=True)[cost_col].sum()
    base_by = (base.groupby(team_col, dropna=False, observed=True)[cost_col]
               .sum() / n_base_days)
    out = pd.DataFrame({"cost_day": day_by, "cost_baseline_per_day": base_by})
    out = out.fillna(0.0)
    out["delta"] = out["cost_day"] - out["cost_baseline_per_day"]
    total_delta = out["delta"].sum()
    out["share_of_change"] = np.where(
        abs(total_delta) > PENNY, out["delta"] / total_delta, np.nan)
    out["vs_own_baseline"] = np.where(
        out["cost_baseline_per_day"] > PENNY,
        out["cost_day"] / out["cost_baseline_per_day"], np.inf)
    return out.sort_values("delta", key=lambda s: s.abs(), ascending=False)
