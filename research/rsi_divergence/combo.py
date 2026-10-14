"""RSI divergence + MACD / EMA confirmation on BTC/USD 4h.

Same data, costs, periods, selection rule and placebos as run.py. The grid adds a
confirmation rule (pinetest.divergence.CONFIRM) and the EMA200 trend filter on top of the
divergence settings. Besides the TRAIN-chosen config, every confirmation is compared with the
plain divergence it filters (same kind, pivots and reward:risk) to see whether it helps.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run import COST, PERIODS, SEED, by_period, load, placebo  # noqa: E402

from pinetest.divergence import CONFIRM, rsi_divergences  # noqa: E402
from pinetest.events import simulate, summary  # noqa: E402

OUT = Path(__file__).parent
GRID = {"kind": ["regular", "hidden"], "lb": [3, 5], "rr": [1.0, 2.0, 3.0], "confirm": list(CONFIRM),
        "trend": [False, True]}


def setups_for(df, kind, lb, rr, confirm, trend, sides=(1, -1)):
    return rsi_divergences(df, kind=kind, left=lb, right=lb, rr=rr, confirm=confirm, trend=trend, sides=sides)


def run_grid(df):
    rows = []
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, combo))
        per = by_period(df, simulate(df, setups_for(df, **cfg), COST))
        rows.append({**cfg, **{p: summary(ts) for p, ts in per.items()}})
    return rows


def avg(row, p):
    return row[p].get("avg_r", 0.0) if row[p].get("n", 0) else np.nan


def main():
    df4 = load("4h")
    rows = run_grid(df4)
    res = {"grid": GRID, "n_configs": len(rows), "cost_pct_per_side": COST, "grid_rows": rows}

    # Does each confirmation help the divergence it filters? Pair it with confirm="none".
    base = {(r["kind"], r["lb"], r["rr"], r["trend"]): r for r in rows if r["confirm"] == "none"}
    by_rule = {}
    for rule in CONFIRM:
        rr_rows = [r for r in rows if r["confirm"] == rule]
        entry = {p: {"median_avg_r": round(float(np.nanmedian([avg(r, p) for r in rr_rows])), 3),
                     "n_positive": int(sum(avg(r, p) > 0 for r in rr_rows)),
                     "median_trades": int(np.median([r[p].get("n", 0) for r in rr_rows]))}
                 for p in PERIODS}
        if rule != "none":
            for p in PERIODS:
                diffs = [avg(r, p) - avg(base[(r["kind"], r["lb"], r["rr"], r["trend"])], p) for r in rr_rows]
                diffs = [d for d in diffs if not np.isnan(d)]
                entry[p]["better_than_plain"] = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
                entry[p]["median_gain_r"] = round(float(np.median(diffs)), 3) if diffs else None
        by_rule[rule] = entry
    res["by_confirmation"] = by_rule

    eligible = [r for r in rows if r["train"].get("n", 0) >= 30]
    best = max(eligible, key=lambda r: r["train"]["avg_r"])
    cfg = {k: best[k] for k in GRID}
    res["chosen_on_train"] = cfg
    setups = setups_for(df4, **cfg)
    risk_pct = {(s.bar, s.side): s.side * (df4["open"].iloc[s.bar + 1] - s.stop) / df4["open"].iloc[s.bar + 1]
                for s in setups}
    chosen = {f"cost_{c}": {p: summary(ts) for p, ts in by_period(df4, simulate(df4, setups, c)).items()}
              for c in (0.0, 0.05, COST)}
    test_trades = by_period(df4, simulate(df4, setups, COST))["test"]
    chosen["test_years"] = {str(y): summary([t for t in test_trades if df4.index[t.bar].year == y])
                            for y in sorted({df4.index[t.bar].year for t in test_trades})}
    rng = np.random.default_rng(SEED)
    real = summary(test_trades).get("avg_r", 0.0)
    chosen["placebo_test"] = {}
    for mode in ("same_bars_random_side", "random_bars_same_side", "random_bars_random_side"):
        vals = np.array([summary(placebo(df4, test_trades, risk_pct, cfg["rr"], rng, mode)).get("avg_r", 0.0)
                         for _ in range(300)])
        chosen["placebo_test"][mode] = round(100 * float((vals < real).mean()), 1)
    res["chosen"] = chosen

    res["grid_summary"] = {p: {"median_avg_r": round(float(np.nanmedian([avg(r, p) for r in rows])), 3),
                               "n_positive": int(sum(avg(r, p) > 0 for r in rows)),
                               "n_positive_all_three": None} for p in PERIODS}
    all3 = [r for r in rows if all(avg(r, p) > 0 for p in PERIODS)]
    res["positive_in_all_three_periods"] = [{k: r[k] for k in GRID} | {p: r[p] for p in PERIODS} for r in all3]
    for p in PERIODS:
        res["grid_summary"][p]["n_positive_all_three"] = len(all3)
    tr = pd.Series([avg(r, "train") for r in rows]).rank()
    te = pd.Series([avg(r, "test") for r in rows]).rank()
    res["grid_summary"]["spearman_train_test"] = round(float(tr.corr(te)), 3)

    for tf in ("1h", "1d"):
        df = load(tf)
        per = by_period(df, simulate(df, setups_for(df, **cfg), COST))
        res[f"tf_{tf}_chosen"] = {p: summary(ts) for p, ts in per.items()}

    (OUT / "combo_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "grid_rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
