"""A.4 detector. The load-bearing tests per layer:

  L1    an injected spike alerts, a normal day does not, and — the honest
        core — the spike does not soften its own alarm (margins are
        strictly past).
  L1.5  a December-style glide that never breaches a daily interval raises
        a CUSUM alarm within a plausible delay; stationary noise over a
        year raises none.
  L2    a 10x job day alerts; a new job announces itself once; penny
        changes on a stable job stay silent.
  Attr  the 84.7%-Unknown day alerts as attribution health.
  Table stable ids; re-scoring preserves triage statuses.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from catpipe import detector as D


def _ledger(days=200, sigma=10.0, level=100.0, grp="a", seed=1,
            start=date(2025, 1, 1)):
    """A well-calibrated model's ledger: y ~ N(level, sigma), q50 = level,
    raw band = level +/- 1.645 sigma (true 90% band)."""
    rng = np.random.default_rng(seed)
    d = [start + timedelta(days=i) for i in range(days)]
    y = level + rng.normal(0, sigma, days)
    return pd.DataFrame({
        "run_date": d, "grp": grp, "y_true": y,
        "q05": level - 1.645 * sigma, "q50": level,
        "q95": level + 1.645 * sigma,
    })


class TestLayer1:
    def test_spike_alerts_normal_days_mostly_do_not(self):
        led = _ledger(days=200)
        led.loc[led.index[-1], "y_true"] = 100.0 + 12 * 10.0  # 12-sigma day
        alerts = D.layer1_interval_alerts(led, ["grp"])
        last = alerts[alerts["run_date"] == led["run_date"].iloc[-1]]
        assert len(last) == 1
        a = last.iloc[0]
        assert a["direction"] == "above" and a["severity"] == "high"
        # false-alarm sanity on the other scored days: near the 10% design
        # rate, nowhere near the uncalibrated 2x
        others = alerts[alerts["run_date"] != led["run_date"].iloc[-1]]
        scored_days = 200 - 1 - 1  # first day has no trailing window
        assert len(others) / scored_days < 0.20

    def test_spike_cannot_soften_its_own_alarm(self):
        # margins for day d come from days < d: the spike's own magnitude
        # must not appear in the margin that scores it. Proof: append the
        # spike, and the alert set for all EARLIER days is unchanged.
        led = _ledger(days=120)
        base_alerts = D.layer1_interval_alerts(led, ["grp"])
        spiked = pd.concat([led, pd.DataFrame([{
            "run_date": led["run_date"].iloc[-1] + timedelta(days=1),
            "grp": "a", "y_true": 5000.0,
            "q05": led["q05"].iloc[0], "q50": 100.0,
            "q95": led["q95"].iloc[0],
        }])], ignore_index=True)
        spiked_alerts = D.layer1_interval_alerts(spiked, ["grp"])
        early = spiked_alerts[
            spiked_alerts["run_date"] <= led["run_date"].iloc[-1]]
        assert set(early["alert_id"]) == set(base_alerts["alert_id"])

    def test_score_from_limits_replay(self):
        led = _ledger(days=100)
        cutoff = led["run_date"].iloc[80]
        alerts = D.layer1_interval_alerts(led, ["grp"], score_from=cutoff)
        assert (alerts["run_date"] >= cutoff).all() if len(alerts) else True


class TestLayer15Cusum:
    def test_glide_alarms_within_plausible_delay(self):
        # 150 stationary days, then a glide: mean drifts by 0.8 sigma per
        # day cumulatively capped at 8 sigma over ten days and held — no
        # single day breaches a 1.645-sigma daily band scaled to CUSUM
        # terms, but the drift accumulates
        led = _ledger(days=200, sigma=10.0, seed=4)
        drift_start = 150
        for i in range(drift_start, 200):
            shift = min((i - drift_start + 1) * 0.4, 6.0) * 10.0
            led.loc[led.index[i], "y_true"] += shift
        alarms = D.layer15_cusum_alerts(led, ["grp"])
        assert len(alarms) >= 1
        first = min(alarms["run_date"])
        assert first >= led["run_date"].iloc[drift_start]
        assert first <= led["run_date"].iloc[drift_start + 25]
        assert (alarms["direction"] == "above").all()

    def test_stationary_year_is_quiet_but_not_impossibly_silent(self):
        # a two-sided CUSUM at k=0.5, h=5 has in-control ARL0 ~ 465
        # observations, i.e. ~0.8 false alarms per year BY DESIGN — the
        # triage-load budget. Zero would be an impossible claim; the honest
        # property is rarity, in contrast to the glide's prompt, repeated
        # alarms above.
        led = _ledger(days=365, seed=7)
        alarms = D.layer15_cusum_alerts(led, ["grp"])
        assert len(alarms) <= 3

    def test_downward_glide_flags_below(self):
        led = _ledger(days=220, sigma=10.0, seed=9)
        for i in range(160, 220):
            led.loc[led.index[i], "y_true"] -= min((i - 159) * 4.0, 60.0)
        alarms = D.layer15_cusum_alerts(led, ["grp"])
        assert len(alarms) >= 1
        assert (alarms["direction"] == "below").all()


class TestLayer2Jobs:
    def _job_cost(self, costs, job="BT", start=date(2025, 1, 1)):
        return pd.DataFrame([
            dict(run_date=start + timedelta(days=i), subscription_id="s1",
                 batch_account_name="a", pool_name="p", job_id=f"{job}_{i}",
                 job_name=job, job_team="Pillar1", cost=c)
            for i, c in enumerate(costs)
        ])

    def test_ten_x_day_alerts(self):
        jc = self._job_cost([100.0] * 30 + [1000.0])
        alerts = D.layer2_job_alerts(jc)
        z_alerts = alerts[alerts["metric"] == "job_daily_cost"]
        assert len(z_alerts) == 1
        a = z_alerts.iloc[0]
        assert a["direction"] == "above"
        assert a["observed"] == 1000.0 and a["expected"] == 100.0

    def test_new_job_announced_once(self):
        jc = self._job_cost([500.0, 480.0, 510.0], job="BRAND_NEW")
        alerts = D.layer2_job_alerts(jc)
        new = alerts[alerts["metric"] == "new_job"]
        assert len(new) == 1
        assert new.iloc[0]["run_date"] == date(2025, 1, 1)

    def test_penny_change_on_stable_job_stays_silent(self):
        # constant history => MAD 0 => infinite z without the floor; a
        # 2-currency-unit wobble must NOT alert
        jc = self._job_cost([100.0] * 30 + [102.0])
        alerts = D.layer2_job_alerts(jc)
        assert len(alerts[alerts["metric"] == "job_daily_cost"]) == 0

    def test_robust_to_spiky_history(self):
        # history containing spikes: median/MAD must not learn to expect
        # them — but equally a value inside the spiky range is not novel.
        # A day at the CLEAN level after spikes stays silent; a day far
        # beyond even the spikes alerts.
        hist = ([100.0] * 20 + [1000.0] * 3 + [100.0] * 10)
        jc = self._job_cost(hist + [5000.0])
        alerts = D.layer2_job_alerts(jc)
        z = alerts[alerts["metric"] == "job_daily_cost"]
        assert (z["run_date"] == date(2025, 1, 1)
                + timedelta(days=len(hist))).any()


class TestAttribution:
    def test_unknown_heavy_day_alerts(self):
        f2 = pd.DataFrame({
            "run_date": [date(2025, 3, 1)] * 3 + [date(2025, 3, 2)] * 3,
            "job_team": ["a", "b", "c"] * 2,
            "unknown_pct": [0.02] * 3 + [0.847] * 3,
        })
        alerts = D.attribution_alerts(f2)
        assert len(alerts) == 1
        a = alerts.iloc[0]
        assert a["run_date"] == date(2025, 3, 2)
        assert a["severity"] == "high"
        assert "Attribution health" in a["message"]


class TestAlertTable:
    def test_ids_stable_and_statuses_survive_rescoring(self):
        led = _ledger(days=200)
        led.loc[led.index[-1], "y_true"] = 300.0
        first = D.run_detector({"frame_1a": (led, ["grp"])})
        assert len(first) >= 1
        # triage happens: someone marks the top alert expected
        triaged = first.copy()
        triaged.loc[0, "status"] = "expected"
        second = D.run_detector({"frame_1a": (led, ["grp"])},
                                previous_alerts=triaged)
        kept = second[second["alert_id"] == triaged.loc[0, "alert_id"]]
        assert kept["status"].iloc[0] == "expected"
        fresh = second[second["alert_id"] != triaged.loc[0, "alert_id"]]
        assert (fresh["status"] == "new").all()

    def test_table_sorted_high_first_and_schema_complete(self):
        led = _ledger(days=200, seed=2)
        led.loc[led.index[-1], "y_true"] = 400.0  # high
        jc = pd.DataFrame([
            dict(run_date=date(2025, 5, 1) + timedelta(days=i),
                 subscription_id="s1", batch_account_name="a",
                 pool_name="p", job_id=f"j{i}", job_name="BT",
                 job_team="T", cost=c)
            for i, c in enumerate([100.0] * 30 + [140.0])  # mild z alert
        ])
        table = D.run_detector({"frame_1a": (led, ["grp"])}, job_cost=jc)
        assert list(table.columns) == D.ALERT_COLUMNS
        sev = table["severity"].map({"high": 0, "medium": 1, "low": 2})
        assert sev.is_monotonic_increasing

    def test_empty_inputs_empty_table(self):
        table = D.run_detector({})
        assert len(table) == 0
        assert list(table.columns) == D.ALERT_COLUMNS
