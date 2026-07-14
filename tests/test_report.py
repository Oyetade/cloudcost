"""Report: matplotlib charts embedded as base64 PNGs in one HTML file.
The tests assert the structural contract: every chart lands as a PNG data
URI (magic bytes checked), the sections and tables are present, an
unscoreable frame is reported rather than crashed, and write_report
produces the file.
"""

import base64
from datetime import date, timedelta

import numpy as np
import pandas as pd

from catpipe import baselines as B
from catpipe import harness as H
from catpipe import report as R

PNG_MAGIC = b"\x89PNG"


def _pngs(html_doc: str) -> list[bytes]:
    out = []
    for chunk in html_doc.split('src="data:image/png;base64,')[1:]:
        out.append(base64.b64decode(chunk.split('"')[0]))
    return out


def _ledger(days=120, start=date(2025, 1, 1)):
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "run_date": [start + timedelta(days=i) for i in range(days)],
        "grp": "a", "model": "m",
        "y_true": 100 + rng.normal(0, 8, days),
        "q05": 87.0, "q50": 100.0, "q95": 113.0,
        "origin": [date(2025, 1 + i // 31, 1) for i in range(days)],
    })


def _bt():
    rng = np.random.default_rng(5)
    start = date(2024, 1, 1)
    vals = [100.0 + 30 * ((start + timedelta(days=i)).weekday() >= 5)
            + rng.normal(0, 3) for i in range(420)]
    f = pd.DataFrame({
        "run_date": [start + timedelta(days=i) for i in range(420)],
        "grp": "a", "cost": vals, "data_regime": "featured_gated",
    })
    f.attrs["group_keys"] = ["grp"]
    summary, ledger = H.run_models(f, B.all_baselines())
    return {"frame_1a": {"summary": summary, "ledger": ledger}}


class TestCharts:
    def test_series_chart_is_a_png(self):
        img = R.series_chart(_ledger(), "t")
        assert img.startswith('<img src="data:image/png;base64,')
        assert _pngs(img)[0][:4] == PNG_MAGIC

    def test_bar_chart_is_a_png_with_reference(self):
        img = R.bar_chart(["m1", "m2"], [0.8, 0.92], "cov",
                          reference=0.90, reference_label="target",
                          pct=True)
        assert _pngs(img)[0][:4] == PNG_MAGIC

    def test_nan_in_series_does_not_crash(self):
        led = _ledger(days=40)
        led.loc[led.index[5], "y_true"] = np.nan
        assert _pngs(R.series_chart(led, "t"))[0][:4] == PNG_MAGIC


class TestFullReport:
    def test_report_builds_with_all_pieces(self):
        html_doc = R.build_report(_bt(), title="CAT test report")
        assert html_doc.startswith("<!doctype html>")
        assert "frame_1a" in html_doc
        # chart titles live inside the PNGs now; the HTML text carries the
        # summary table, whose headers are the stable contract
        assert "mae_daily" in html_doc and "coverage_90" in html_doc
        assert "seasonal_naive" in html_doc
        pngs = _pngs(html_doc)
        assert len(pngs) >= 4
        assert all(p[:4] == PNG_MAGIC for p in pngs)

    def test_not_scoreable_frame_reported_not_crashed(self):
        html_doc = R.build_report(
            {"frame_1b": {"error": "no honest monthly origin exists"}})
        assert "not scoreable" in html_doc

    def test_alert_section_included_when_given(self):
        alerts = pd.DataFrame({
            "run_date": [date(2025, 3, 1)], "layer": ["L1_interval"],
            "severity": ["high"], "message": ["m"], "status": ["new"],
            "alert_id": ["abc"], "scope": ["s"], "metric": ["c"],
            "observed": [1.0], "expected": [0.5], "lo": [0.0], "hi": [0.9],
            "score": [1.2], "direction": ["above"],
        })
        html_doc = R.build_report(_bt(), alerts=alerts)
        assert "A.4 alerts" in html_doc and "triage" in html_doc

    def test_write_report_creates_the_file(self, tmp_path):
        p = tmp_path / "r.html"
        R.write_report(p, _bt())
        assert p.exists() and p.stat().st_size > 20_000
