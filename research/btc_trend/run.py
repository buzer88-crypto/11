"""Pre-registered follow-up: long/flat trend following on BTC alone.

Hypothesis fixed before looking at results: hold BTC when its trend is up (price above its
N-day SMA, or positive N-day return), otherwise cash; optionally size to a 40% annual vol
target. Parameters are chosen on TRAIN (2018-2021) only; TEST (2022-2026) and PRE (2014-2017)
are out of sample, and ETH is an out-of-asset check with the same TRAIN-chosen parameters.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.crypto_data import load_panel
from pinetest.portfolio import TEST, TRAIN, backtest, buy_and_hold, circular_shift, rebalance_on

PRE = ("2014-01-01", "2017-12-31")
OUT = Path(__file__).parent
GRID = {"signal": ["sma", "ret"], "n": [50, 100, 150, 200], "sizing": ["full", "vt40"],
        "rebal": ["D", "W"]}

panel = load_panel()
px = panel["PriceUSD"]


def weights(asset, signal, n, sizing, rebal):
    p = px[asset]
    on = (p > p.rolling(n).mean()) if signal == "sma" else (p / p.shift(n) - 1 > 0)
    size = pd.Series(1.0, index=p.index)
    if sizing == "vt40":
        vol = p.pct_change(fill_method=None).rolling(60).std() * np.sqrt(365)
        size = (0.40 / vol).clip(upper=1.0)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w[asset] = (on.astype(float) * size).where(p.notna(), 0.0).fillna(0.0)
    return rebalance_on(w, "W-SUN") if rebal == "W" else w


def m(res, period, asset="btc"):
    # Start at the asset's listing so pre-listing zero-return days do not dilute the metrics.
    start = max(pd.Timestamp(period[0]), px[asset].first_valid_index())
    x = res.metrics(start, period[1])
    return {k: round(x[k], 3) for k in ("sharpe", "cagr_pct", "total_pct", "max_dd_pct", "exposure")}


def yearly(res, years):
    return {str(y): round(100 * ((1 + res.returns.loc[str(y)]).prod() - 1), 1) for y in years}


bench = {a: backtest(buy_and_hold(px, a), px) for a in ("btc", "eth")}
rows = []
for combo in itertools.product(*GRID.values()):
    cfg = dict(zip(GRID, combo))
    row = dict(cfg)
    for a in ("btc", "eth"):
        r = backtest(weights(a, **cfg), px)
        row[a] = {"train": m(r, TRAIN, a), "test": m(r, TEST, a), "pre": m(r, PRE, a)}
    rows.append(row)

best = max(rows, key=lambda r: r["btc"]["train"]["sharpe"])
cfg = {k: best[k] for k in GRID}
w = weights("btc", **cfg)
res = backtest(w, px)
rng = np.random.default_rng(20261014)
plac = []
for _ in range(500):
    pr = backtest(circular_shift(w, px, rng), px)
    plac.append((pr.metrics(*TEST)["sharpe"], pr.metrics(*PRE)["sharpe"]))
plac = np.array(plac)
real_test, real_pre = res.metrics(*TEST)["sharpe"], res.metrics(*PRE)["sharpe"]

summary = {
    "grid": GRID, "n_configs": len(rows), "chosen_on_train": cfg,
    "chosen": {
        "btc": {"train": m(res, TRAIN), "test": m(res, TEST), "pre": m(res, PRE),
                "test_years": yearly(res, range(2022, 2027)), "pre_years": yearly(res, range(2014, 2018))},
        "eth_same_params": best["eth"],
        "placebo_percentile_test": round(100 * float((plac[:, 0] < real_test).mean()), 1),
        "placebo_percentile_pre": round(100 * float((plac[:, 1] < real_pre).mean()), 1),
        "test_sharpe_cost30": round(backtest(w, px, cost_bps=30).metrics(*TEST)["sharpe"], 3),
        "test_sharpe_cost50": round(backtest(w, px, cost_bps=50).metrics(*TEST)["sharpe"], 3),
        "test_sharpe_lag0": round(backtest(w, px, lag=0).metrics(*TEST)["sharpe"], 3),
        "test_sharpe_lag2": round(backtest(w, px, lag=2).metrics(*TEST)["sharpe"], 3),
    },
    "benchmarks": {a: {"train": m(b, TRAIN, a), "test": m(b, TEST, a), "pre": m(b, PRE, a),
                       "test_years": yearly(b, range(2022, 2027))} for a, b in bench.items()},
    "grid_btc": {
        "median_sharpe": {p: float(np.median([r["btc"][p]["sharpe"] for r in rows])) for p in ("train", "test", "pre")},
        "n_beat_bh_sharpe": {p: int(sum(r["btc"][p]["sharpe"] > bench["btc"].metrics(*P)["sharpe"] for r in rows))
                             for p, P in (("train", TRAIN), ("test", TEST), ("pre", PRE))},
        "n_better_max_dd": {p: int(sum(r["btc"][p]["max_dd_pct"] > bench["btc"].metrics(*P)["max_dd_pct"] for r in rows))
                            for p, P in (("train", TRAIN), ("test", TEST), ("pre", PRE))},
    },
    "grid_eth": {
        "median_sharpe": {p: float(np.median([r["eth"][p]["sharpe"] for r in rows])) for p in ("train", "test", "pre")},
        "n_beat_bh_sharpe": {p: int(sum(r["eth"][p]["sharpe"] > m(bench["eth"], P, "eth")["sharpe"] for r in rows))
                             for p, P in (("train", TRAIN), ("test", TEST), ("pre", PRE))},
    },
    "rows": rows,
}
(OUT / "results.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
