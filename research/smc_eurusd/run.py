"""SMC setups (liquidity sweep, fair value gap, order block) on EURUSD 1H, 2017-04..2018-02.

Data: data/EURUSD.csv (the hourly sample bundled with the `backtesting` PyPI package).
Parameters are chosen per pattern on the first 70% of bars (by average R) and judged on the
last 30%. Placebos keep each real trade's risk and reward:reward ratio but take random
direction at the same bars, or random bars and random direction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.events import Setup, simulate, summary
from pinetest.smc import fair_value_gaps, liquidity_sweeps, order_blocks

OUT = Path(__file__).parent
COST = 0.005   # % per side, about 0.55 pip on EURUSD: ~1.1 pip per round trip (ECN-like)
df = pd.read_csv(Path(__file__).parents[2] / "data" / "EURUSD.csv")
df.columns = [c.lower() for c in df.columns]
SPLIT = int(len(df) * 0.7)

PATTERNS = {
    "sweep": (liquidity_sweeps, {"lookback": [12, 24, 48], "rr": [1.5, 2.0, 3.0]}),
    "fvg": (fair_value_gaps, {"min_gap_atr": [0.2, 0.5, 1.0], "rr": [1.5, 2.0, 3.0]}),
    "order_block": (order_blocks, {"lookback": [12, 24, 48], "rr": [1.5, 2.0, 3.0]}),
}


def part(trades, lo, hi):
    return [t for t in trades if lo <= t.bar < hi]


def placebo(trades, rng, same_bars):
    """Market setups with each real trade's risk distance and reward ratio."""
    o = df["open"].to_numpy()
    out = []
    for t in trades:
        bar = t.bar if same_bars else int(rng.integers(30, len(df) - 2))
        side = int(rng.choice([-1, 1]))
        ref = o[bar + 1]
        out.append(Setup(bar, side, "market", ref - side * t.risk, ref + side * t.rr * t.risk, max_hold=48))
    return simulate(df, out, COST)


results = {"cost_pct_per_side": COST, "bars": len(df), "split_bar": SPLIT,
           "train": [df.iloc[0, 0], df.iloc[SPLIT - 1, 0]], "test": [df.iloc[SPLIT, 0], df.iloc[-1, 0]],
           "patterns": {}}
rng = np.random.default_rng(20261014)
for name, (fn, grid) in PATTERNS.items():
    rows = []
    keys = list(grid)
    for a in grid[keys[0]]:
        for b in grid[keys[1]]:
            params = {keys[0]: a, keys[1]: b}
            setups = fn(df, **params)
            trades = simulate(df, setups, COST)
            rows.append({"params": params, "n_setups": len(setups),
                         "train": summary(part(trades, 0, SPLIT)), "test": summary(part(trades, SPLIT, len(df))),
                         "test_cost0": summary(part(simulate(df, setups, 0.0), SPLIT, len(df)))})
    best = max(rows, key=lambda r: r["train"].get("avg_r", -9))
    setups = fn(df, **best["params"])
    trades = simulate(df, setups, COST)
    test_trades = part(trades, SPLIT, len(df))
    rr = best["params"]["rr"]
    for t in test_trades:          # attach what the placebo needs
        t.risk = abs(t.entry - next(s.stop for s in setups if s.bar == t.bar))
        t.rr = rr
    real = summary(test_trades).get("avg_r", 0.0)
    same, rand = [], []
    for _ in range(300):
        same.append(summary(placebo(test_trades, rng, True)).get("avg_r", 0.0))
        rand.append(summary(placebo(test_trades, rng, False)).get("avg_r", 0.0))
    results["patterns"][name] = {
        "grid": rows,
        "chosen_on_train": best["params"],
        "chosen_train": best["train"], "chosen_test": best["test"], "chosen_test_cost0": best["test_cost0"],
        "breakeven_win_rate_pct": round(100 / (1 + rr), 1),
        "placebo_same_bars_random_side_pct_below_real": round(100 * float(np.mean(np.array(same) < real)), 1),
        "placebo_random_bars_pct_below_real": round(100 * float(np.mean(np.array(rand) < real)), 1),
        "grid_median_test_avg_r": float(np.median([r["test"].get("avg_r", 0.0) for r in rows])),
        "grid_n_test_positive": int(sum(r["test"].get("avg_r", 0.0) > 0 for r in rows)),
    }

(OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
for name, p in results["patterns"].items():
    print(f"\n== {name}: chosen on train {p['chosen_on_train']}")
    print(f"   train {p['chosen_train']}")
    print(f"   test  {p['chosen_test']}   (cost 0: avg_r {p['chosen_test_cost0'].get('avg_r')})")
    print(f"   break-even win rate {p['breakeven_win_rate_pct']}%, placebo same bars {p['placebo_same_bars_random_side_pct_below_real']}%, random bars {p['placebo_random_bars_pct_below_real']}%")
    print(f"   grid: median test avg R {p['grid_median_test_avg_r']:.3f}, positive in {p['grid_n_test_positive']}/9")
    for r in p["grid"]:
        print(f"     {r['params']}  train {r['train'].get('avg_r')} (n {r['train'].get('n')})  test {r['test'].get('avg_r')} (n {r['test'].get('n')}, win {r['test'].get('win_rate_pct')}%)")
