"""
report.py  --  a self-contained HTML performance report from the backtest
artefacts: model comparison charts, actual-vs-forecast series with the 5-95
band, coverage against target, and the alert summary. Charts are matplotlib
(approved July 2026), embedded as base64 PNGs so the report remains a
single file that opens anywhere and can sit beside backtest_summary.json in
the snapshot's frames/ directory.

The report never recomputes a metric; it renders what the harness scored,
so a number in the report is a number in backtest_summary.json.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")  # headless: the pipeline has no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RED, GREY = "#2563eb", "#dc2626", "#6b7280"


def _fig_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}"/>'


def series_chart(led: pd.DataFrame, title: str) -> str:
    """Actual vs median forecast with the 5-95 band; fold origins as faint
    verticals (the model never saw data at or after an origin when
    forecasting that fold)."""
    d = led.sort_values("run_date")
    x = pd.to_datetime(d["run_date"])
    fig, ax = plt.subplots(figsize=(8.2, 2.9))
    ax.fill_between(x, d["q05"], d["q95"], color=BLUE, alpha=0.15,
                    linewidth=0, label="5-95 band")
    ax.plot(x, d["q50"], color=BLUE, lw=1.2, ls="--", label="median")
    ax.plot(x, d["y_true"], color="black", lw=1.0, label="actual")
    for o in pd.to_datetime(d["origin"]).unique():
        ax.axvline(o, color=GREY, lw=0.6, ls=":", alpha=0.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.margins(x=0.01)
    return _fig_html(fig)


def bar_chart(labels, values, title: str, reference: float | None = None,
              reference_label: str = "", pct: bool = False) -> str:
    """Horizontal bars, one per model, optional reference line (e.g. the
    0.90 coverage target)."""
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 0.9))
    y = np.arange(len(labels))
    ax.barh(y, values, color=BLUE, height=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for yi, v in zip(y, values):
        if np.isfinite(v):
            ax.text(v, yi, f" {v:.1%}" if pct else f" {v:,.2f}",
                    va="center", fontsize=8)
    if reference is not None:
        ax.axvline(reference, color=RED, ls="--", lw=1)
        ax.text(reference, -0.55, f" {reference_label}", color=RED,
                fontsize=8)
    if pct:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title(title, fontsize=10, loc="left")
    ax.tick_params(labelsize=8)
    return _fig_html(fig)


def _table_html(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(
            f"<td>{html.escape(f'{v:,.4g}' if isinstance(v, float) else str(v))}"
            "</td>" for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


CSS = """
body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;
     color:#111827;padding:0 12px}
h1{font-size:20px} h2{font-size:16px;margin-top:32px;
   border-bottom:1px solid #e5e7eb;padding-bottom:4px}
table{border-collapse:collapse;font-size:12px;margin:8px 0}
th,td{border:1px solid #e5e7eb;padding:4px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
.note{color:#6b7280;font-size:12px}
img{max-width:100%;margin:6px 0}
"""

LEDGER_META = ("run_date", "y_true", "q05", "q50", "q95", "origin", "model")


def build_report(
    bt: dict,
    alerts: pd.DataFrame | None = None,
    title: str = "CAT model back-test report",
    generated: date | None = None,
    max_series_groups: int = 3,
) -> str:
    """The whole report as one HTML string. Per frame: the harness summary
    table, MAE / monthly-error / coverage charts, and actual-vs-forecast
    series for the best model on the largest groups by actual spend.
    Alerts, if given, summarised by layer and severity with the triage top.
    """
    gen = generated or date.today()
    out = [f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body><h1>{html.escape(title)}</h1>"
           f"<p class='note'>generated {gen.isoformat()}; every number is "
           f"the harness's, recomputable from the prediction ledgers.</p>"]

    for frame_name, res in bt.items():
        out.append(f"<h2>{html.escape(frame_name)}</h2>")
        if "error" in res:
            out.append(f"<p class='note'>not scoreable: "
                       f"{html.escape(res['error'])}</p>")
            continue
        summary, ledger = res["summary"], res["ledger"]
        out.append(_table_html(summary.round(4)))

        s = summary.dropna(subset=["mae_daily"])
        out.append(bar_chart(s["model"].tolist(), s["mae_daily"].tolist(),
                             "Daily MAE by model (lower is better)"))
        if "monthly_pct_err_estate" in s:
            out.append(bar_chart(
                s["model"].tolist(), s["monthly_pct_err_estate"].tolist(),
                "Estate monthly error (the incumbent-comparison number)",
                pct=True))
        if "monthly_wape" in s:
            out.append(bar_chart(
                s["model"].tolist(), s["monthly_wape"].tolist(),
                "Monthly WAPE, spend-weighted (attribution accuracy: "
                "offsetting errors do not cancel)", pct=True))
        if "coverage_90" in s:
            out.append(bar_chart(
                s["model"].tolist(), s["coverage_90"].tolist(),
                "Empirical coverage of the 5-95 interval",
                reference=0.90, reference_label="target 0.90", pct=True))

        best = s.sort_values("mae_daily").iloc[0]["model"]
        led = ledger[ledger["model"] == best]
        gk = [c for c in led.columns if c not in LEDGER_META]
        if gk:
            top = (led.groupby(gk, observed=True)["y_true"].sum()
                   .sort_values(ascending=False).head(max_series_groups))
            for keys in top.index:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                m = led
                for k, v in zip(gk, keys):
                    m = m[m[k] == v]
                label = ", ".join(f"{k}={v}" for k, v in zip(gk, keys))
                out.append(series_chart(
                    m, f"{best} on {label}: actual vs forecast"))
        out.append("<p class='note'>dotted verticals are fold origins; the "
                   "model never saw data at or after an origin when "
                   "forecasting that fold.</p>")

    if alerts is not None and len(alerts):
        out.append("<h2>A.4 alerts</h2>")
        by = (alerts.groupby(["layer", "severity"], observed=True).size()
              .rename("count").reset_index())
        out.append(_table_html(by))
        out.append("<p class='note'>top of the triage queue:</p>")
        out.append(_table_html(
            alerts.head(8)[["run_date", "layer", "severity", "message",
                            "status"]]))

    out.append("</body></html>")
    return "\n".join(out)


def write_report(path, bt, alerts=None, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_report(bt, alerts=alerts, **kwargs),
                 encoding="utf-8")
