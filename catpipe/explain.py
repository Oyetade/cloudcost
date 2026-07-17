"""
explain.py  --  the joined per-alert explanation: model attribution plus
business-driver decomposition (the EVENT EXPLAIN checklist).

Two halves, deliberately different in nature and labelled as such:

  MODEL: which features drove THIS prediction. LightGBM's pred_contrib
  (TreeSHAP) on the MEAN booster, whose target is untransformed, so every
  contribution reads in pounds and they sum exactly to the prediction:
  baseline + sum(contributions) = pred_mean, asserted per row. The quantile
  boosters' contributions live on the asinh scale and are not reported;
  a pound number computed there would be false precision.

  BUSINESS: which meter drove the EVENT, and how much was price, usage,
  scope. change_decomposition against the group's own trailing baseline.
  Exact accounting, no model.

The two halves answer different questions (why the model expected what it
expected; why the cost did what it did) and an alert needs both. They are
joined per (run_date, group), never blended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import change_decomposition as CD
from .persistence import LoadedModel, PersistenceError

CONTRIB_TOL = 1e-6


def local_attribution(
    model: LoadedModel, frame: pd.DataFrame, top_k: int = 8
) -> pd.DataFrame:
    """Per-row feature contributions to pred_mean, in pounds.

    Returns one row per input row: pred_mean, baseline (the booster's
    expected value), then contrib_<feature> for every feature. The
    additivity identity is asserted: a break means the matrix scored here
    is not the matrix the model scored, exactly the skew design_matrix
    exists to prevent.
    """
    if "mean" not in model.boosters:
        raise PersistenceError(
            "local attribution needs the mean booster; the bundle only has "
            f"{sorted(model.boosters)}"
        )
    X, _, _, _ = model.design_matrix(frame)
    contrib = model.boosters["mean"].predict(X, pred_contrib=True)
    cols = [f"contrib_{c}" for c in model.card.feature_names] + ["baseline"]
    out = pd.DataFrame(contrib, columns=cols, index=frame.index)
    out["pred_mean"] = model.boosters["mean"].predict(X)
    total = out[cols].sum(axis=1)
    if (total - out["pred_mean"]).abs().max() > CONTRIB_TOL:
        raise PersistenceError(
            "pred_contrib does not sum to the prediction; the design matrix "
            "diverged from the scored matrix"
        )
    return out


def top_drivers(attribution_row: pd.Series, k: int = 8) -> list[dict]:
    """The k largest signed feature contributions for one row."""
    contribs = attribution_row.filter(like="contrib_")
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    return [
        {"feature": name.removeprefix("contrib_"), "contribution": round(float(v), 2)}
        for name, v in top.head(k).items()
        if abs(v) > 0.005
    ]


@dataclass
class EventExplanation:
    run_date: str
    group: dict[str, str]
    prediction: dict            # pred_mean, baseline
    model_drivers: list[dict]   # top signed contributions, pounds
    decomposition: dict         # price/usage/scope/unpriced summary
    top_meters: list[dict]      # largest per-meter deltas with effects
    team_contribution: list[dict] | None

    def to_dict(self) -> dict:
        return {
            "run_date": self.run_date,
            "group": self.group,
            "prediction": self.prediction,
            "model_drivers": self.model_drivers,
            "decomposition": self.decomposition,
            "top_meters": self.top_meters,
            "team_contribution": self.team_contribution,
        }


def explain_event(
    model: LoadedModel,
    frame: pd.DataFrame,
    raw_cost: pd.DataFrame,
    run_date,
    group: dict[str, str],
    job_cost: pd.DataFrame | None = None,
    baseline_days: int = 28,
    top_k: int = 8,
) -> EventExplanation:
    """One alert's complete explanation.

    frame is the featured frame the alert row came from (the training
    builders' output); raw_cost and job_cost are snapshot tables. group is
    the alert's group keys, e.g. subscription/batch account/pool for the
    pool frame, and must match the raw_cost columns.
    """
    run_date = pd.Timestamp(run_date)
    row_mask = pd.to_datetime(frame["run_date"]) == run_date
    for k, v in group.items():
        row_mask &= frame[k] == v
    rows = frame.loc[row_mask]
    if len(rows) != 1:
        raise PersistenceError(
            f"expected exactly one frame row for {group} on "
            f"{run_date.date()}, found {len(rows)}"
        )

    attribution = local_attribution(model, rows, top_k=top_k)
    arow = attribution.iloc[0]

    decomp = CD.day_vs_baseline(
        raw_cost, run_date, group, baseline_days=baseline_days)
    meters = [
        {
            "meter": r[decomp.items.columns[0]],
            "kind": r["kind"],
            "delta": round(float(r["delta"]), 2),
            "price_effect": round(float(r["price_effect"]), 2),
            "usage_effect": round(float(r["usage_effect"]), 2),
            "scope_effect": round(float(r["scope_effect"]), 2),
        }
        for _, r in decomp.top_items(5).iterrows()
    ]

    teams = None
    if job_cost is not None:
        tc = CD.team_contribution(
            job_cost, run_date, group, baseline_days=baseline_days)
        teams = [
            {
                "team": str(idx),
                "delta": round(float(r["delta"]), 2),
                "share_of_change": (round(float(r["share_of_change"]), 3)
                                    if np.isfinite(r["share_of_change"]) else None),
                "vs_own_baseline": (round(float(r["vs_own_baseline"]), 2)
                                    if np.isfinite(r["vs_own_baseline"]) else None),
            }
            for idx, r in tc.head(5).iterrows()
        ]

    return EventExplanation(
        run_date=str(run_date.date()),
        group=dict(group),
        prediction={
            "pred_mean": round(float(arow["pred_mean"]), 2),
            "baseline": round(float(arow["baseline"]), 2),
        },
        model_drivers=top_drivers(arow, k=top_k),
        decomposition=decomp.summary(),
        top_meters=meters,
        team_contribution=teams,
    )


def format_explanation(e: EventExplanation) -> str:
    """A plain-text block for the report or an alert message. British
    English, effects first, model second."""
    d = e.decomposition
    lines = [
        f"Cost event: {', '.join(f'{k}={v}' for k, v in e.group.items())} "
        f"on {e.run_date}",
        f"Change vs trailing baseline: {d['delta']:+,.2f} "
        f"({d['cost_before']:,.2f} -> {d['cost_after']:,.2f})",
        f"  price {d['price_effect']:+,.2f} | usage {d['usage_effect']:+,.2f}"
        f" | scope {d['scope_effect']:+,.2f}"
        + (f" | unpriced {d['unpriced_effect']:+,.2f}"
           if abs(d["unpriced_effect"]) >= 0.01 else ""),
        "Largest meters:",
    ]
    for m in e.top_meters:
        lines.append(
            f"  {m['meter']} ({m['kind']}): {m['delta']:+,.2f} "
            f"[price {m['price_effect']:+,.2f}, usage {m['usage_effect']:+,.2f}"
            f", scope {m['scope_effect']:+,.2f}]"
        )
    if e.team_contribution:
        lines.append("Team contribution to the change:")
        for t in e.team_contribution:
            share = (f"{t['share_of_change']:+.0%}"
                     if t["share_of_change"] is not None else "n/a")
            lines.append(f"  {t['team']}: {t['delta']:+,.2f} ({share} of change)")
    lines.append(
        f"Model expectation: {e.prediction['pred_mean']:,.2f} "
        f"(booster baseline {e.prediction['baseline']:,.2f}); "
        "largest feature contributions:"
    )
    for f in e.model_drivers:
        lines.append(f"  {f['feature']}: {f['contribution']:+,.2f}")
    return "\n".join(lines)
