"""Chosen config: TEST years and the study's three placebos, with every trade exited on 1-minute bars.
Placebo setups are drawn exactly as in run.py (same seed, same draw order)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from intrabar import Minutes, chained, period_of, stats  # noqa: E402
from pinetest.events import Setup, simulate  # noqa: E402
from research.rsi_divergence.run import COST, SEED, by_period, load, setups_for  # noqa: E402

OUT = Path(__file__).parent


def main():
    df4 = load("4h")
    M = Minutes(df4)
    cfg = ("hidden", 3, 3.0, "trend")
    setups = setups_for(df4, *cfg)
    o = df4["open"].to_numpy()
    risk_pct = {(s.bar, s.side): s.side * (o[s.bar + 1] - s.stop) / o[s.bar + 1] for s in setups}
    test_trades = by_period(df4, simulate(df4, setups, COST))["test"]
    ch = chained(M, df4, setups)
    per = period_of(df4, np.array([x[0] for x in ch]))
    test1m = [x for x, p in zip(ch, per) if p == "test"]
    real4 = float(np.mean([t.r for t in test_trades]))
    real1 = float(np.mean([x[2] for x in test1m]))
    years = {}
    for y in sorted({df4.index[x[0]].year for x in test1m}):
        years[str(y)] = {"4h": stats([t.r for t in test_trades if df4.index[t.bar].year == y]),
                         "1m": stats([x[2] for x in test1m if df4.index[x[0]].year == y])}
    span = (df4.index[-1] - pd.Timestamp("2022-01-01")).days / 365.25
    rng = np.random.default_rng(SEED)
    lo, hi = test_trades[0].bar, test_trades[-1].bar
    plac = {}
    for mode in ("same_bars_random_side", "random_bars_same_side", "random_bars_random_side"):
        v4, v1 = [], []
        for _ in range(300):
            out = []
            for t in test_trades:
                bar = t.bar if mode == "same_bars_random_side" else int(rng.integers(lo, hi + 1))
                side = t.side if mode == "random_bars_same_side" else int(rng.choice([-1, 1]))
                ref = o[bar + 1]
                risk = risk_pct[(t.bar, t.side)] * ref
                out.append(Setup(bar, side, "market", ref - side * risk, ref + side * cfg[2] * risk, max_hold=30))
            v4.append(np.mean([t.r for t in simulate(df4, out, COST)]))
            v1.append(np.mean([x[2] for x in chained(M, df4, out)]))
        v4, v1 = np.array(v4), np.array(v1)
        plac[mode] = {"4h_pct_below_real": round(100 * float((v4 < real4).mean()), 1),
                      "4h_placebo_median": round(float(np.median(v4)), 3),
                      "1m_pct_below_real": round(100 * float((v1 < real1).mean()), 1),
                      "1m_placebo_median": round(float(np.median(v1)), 3),
                      "mean_shift_placebo_1m_minus_4h": round(float((v1 - v4).mean()), 4)}
    res = {"test_avg_r_4h": round(real4, 3), "test_avg_r_1m": round(real1, 3), "test_n": len(test1m),
           "test_pct_per_year_at_1pct_risk_1m": round(real1 * len(test1m) / span, 1),
           "test_years": years, "placebo_test": plac}
    (OUT / "results_placebo_1m.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
