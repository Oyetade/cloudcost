"""
transform_experiment.py -- the asinh-bias diagnosis, runnable in one command
against a live snapshot:

    PYTHONPATH=. python experiments/transform_experiment.py <snapshot_dir>

Question under test: the live backtest showed the GBM precise but biased
low at the monthly grain (estate error ~ WAPE on frames 1a and 2, i.e.
one-signed errors that never cancel), and the prime suspect is the asinh
target transform — for a right-skewed target the median sits below the
mean, so summing daily medians under-forecasts monthly totals.

Design: the same QuantileGBM, same folds, three target transforms (asinh,
log1p, none), with the seasonal naive alongside as the unbiased reference.
The verdict column is monthly_bias: if asinh is materially negative where
'none' is near zero, the transform is the mechanism and the per-frame
default should change; if all three are similarly negative, the bias lives
elsewhere (features, early stopping) and the transform is exonerated.

Output: one table per frame, models ranked by estate error, plus a verdict
line naming the transform with the smallest |monthly_bias|.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catpipe import baselines as B          # noqa: E402
from catpipe import harness as H            # noqa: E402
from catpipe import models as M             # noqa: E402
from catpipe.run_pipeline import build_ml_frames   # noqa: E402

TRANSFORMS = ("asinh", "log1p", "none")

MASKS = {
    "frame_1a": lambda f: None,
    "frame_1b": lambda f: f["post_glide"],
    "frame_2": lambda f: f["unknown_pct"] <= 0.20,
}

COLS = ["model", "mae_daily", "monthly_pct_err_estate", "monthly_bias",
        "monthly_wape", "coverage_90"]


def main(snapshot_dir: str) -> int:
    ml = build_ml_frames(snapshot_dir)
    for name, frame in ml.items():
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        gbms = [M.QuantileGBM.for_frame(frame, transform=t)
                for t in TRANSFORMS]
        for g, t in zip(gbms, TRANSFORMS):
            g.name = f"gbm_{t}"
        try:
            summary, _ = H.run_models(
                frame, gbms + [B.SeasonalNaive()],
                mask=MASKS[name](frame))
        except H.BacktestError as e:
            print(f"not scoreable: {e}")
            continue
        print(summary[COLS].round(4).to_string(index=False))
        g = summary[summary["model"].str.startswith("gbm_")]
        best = g.loc[g["monthly_bias"].abs().idxmin()]
        print(f"\nverdict: smallest |monthly_bias| is {best['model']} "
              f"({best['monthly_bias']:+.3%}); asinh row shows "
              f"{g.set_index('model').loc['gbm_asinh', 'monthly_bias']:+.3%}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
