"""
min_history.py  --  how many days of data inference needs, computed from the
declared features, never written in a config.

The rule, for deepest lag L, deepest rolling window W (the feature factory
shifts by one before rolling, so a window of W needs W + 1 days), a
price-drift span D (14-day mean against the prior 28-day mean = 42), and
direct horizon h:

    days = max(L, W + 1, D + lag - 1) + h - 1

Applied to the current frames the binding feature is price_drift_lag1 at 42
days; lags (1, 7) and the shifted 28-day roll (29) sit inside it. If a
90-day roll is added next quarter the demand rises with it and the scoring
pipeline's window assertion raises, rather than quietly serving
edge-degraded features.

Feature names are parsed from the factory's naming conventions
(*_lag{n}, *_roll{n} / *_rolling{n}, *price_drift*); extend _LAG/_ROLL/
_DRIFT if the factory emits other spellings.
"""

from __future__ import annotations

import re
from typing import Iterable

DRIFT_SPAN_DAYS = 42  # 14-day mean vs the prior 28-day mean

_LAG = re.compile(r"_lag_?(\d+)$")
_ROLL = re.compile(r"_roll(?:ing)?_?(\d+)")
_DRIFT = re.compile(r"(?:^|_)(?:effective_)?price_drift")


def feature_history_days(name: str) -> int:
    """Days of history one feature needs to be fully formed at h=1.

    Drift is checked before lag: price_drift_lag1 is the 42-day drift span
    lagged by one, needing days t-42 .. t-1, i.e. 42 days of prior history
    (D + lag - 1), not the 1 day a bare lag suffix would suggest.
    """
    if _DRIFT.search(name):
        m = _LAG.search(name)
        lag = int(m.group(1)) if m else 1  # unlagged drift still ends at t
        return DRIFT_SPAN_DAYS + lag - 1
    m = _LAG.search(name)
    if m:
        return int(m.group(1))
    m = _ROLL.search(name)
    if m:
        return int(m.group(1)) + 1  # shifted-then-rolled: window ends at t-1
    return 0  # calendar leads, static attributes, categoricals


def min_history_days(feature_names: Iterable[str], horizon_days: int = 1) -> int:
    """Minimum complete per-group history to produce one fully-featured
    scoring row for a direct model at the given horizon."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    deepest = max((feature_history_days(f) for f in feature_names), default=0)
    return deepest + horizon_days - 1


def recommended_extract_days(
    feature_names: Iterable[str],
    horizon_days: int = 1,
    *,
    load_lag_days: int = 10,
    detector_window_days: int = 90,
    margin_days: int = 15,
) -> int:
    """The window actually pulled from Postgres.

    Larger than the minimum for three reasons: the load lag means the most
    recent complete day may trail today by around ten days (C8); the
    detector wants trailing 90-day windows (Layer 1 conformal, Layer 2 job
    profiles) independent of the forecaster; and rolling features computed
    on a truncated window differ at the edges, so we pull long, compute on
    the full window, and score the tail. Never pull exactly the minimum.
    """
    floor = min_history_days(feature_names, horizon_days)
    return max(floor + load_lag_days + margin_days,
               detector_window_days + load_lag_days + margin_days)
