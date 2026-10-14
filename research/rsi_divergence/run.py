"""RSI divergence on BTC/USD 4h (Bitstamp, 2012-2026), with 1h and 1d as neighbour timeframes.

Setups follow TradingView's "RSI Divergence Indicator" (pinetest/divergence.py). Each signal
is traded at the next open with a stop beyond the pivot and a fixed reward:risk target; results
are in R (multiples of the risk taken) after 0.1% per side costs. Parameters are chosen on
TRAIN (2018-2021) by average R and judged on TEST (2022-01-01 .. latest) and PRE (2014-2017).
Placebos keep each real trade's risk (as % of price) and reward:risk but move it to random
bars and/or give it a random direction.
Run build_data.py first.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.divergence import rsi_divergences
from pinetest.events import Setup, simulate, summary

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
COST = 0.1
PERIODS = {"pre": ("2014-01-01", "2017-12-31"), "train": ("2018-01-01", "2021-12-31"),
           "test": ("2022-01-01", "2100-01-01")}
GRID = {"kind": ["regular", "hidden"], "lb": [3, 5], "rr": [1.0, 2.0, 3.0],
        "filter": ["none", "extreme", "trend"]}
SEED = 20261014


def load(tf):
    df = pd.read_csv(ROOT / "data" / f"BTCUSD_{tf}.csv", parse_dates=["date"]).set_index("date")
    return df.loc["2013-01-01":]          # Bitstamp's 2012 minutes are mostly empty


def setups_for(df, kind, lb, rr, filt, sides=(1, -1)):
    return rsi_divergences(df, kind=kind, left=lb, right=lb, rr=rr, extreme=filt == "extreme",
                           trend=filt == "trend", sides=sides)


def by_period(df, trades):
    idx = df.index
    out = {}
    for name, (a, b) in PERIODS.items():
        # Periods run through the whole last day (4h/1h bars after 00:00 included).
        lo, hi = idx.searchsorted(pd.Timestamp(a)), idx.searchsorted(pd.Timestamp(b) + pd.Timedelta(days=1))
        out[name] = [t for t in trades if lo <= t.bar < hi]
    return out


def placebo(df, trades, risk_pct, rr, rng, mode):
    o = df["open"].to_numpy()
    lo, hi = trades[0].bar, trades[-1].bar
    out = []
    for t in trades:
        bar = t.bar if mode == "same_bars_random_side" else int(rng.integers(lo, hi + 1))
        side = t.side if mode == "random_bars_same_side" else int(rng.choice([-1, 1]))
        ref = o[bar + 1]
        risk = risk_pct[(t.bar, t.side)] * ref
        out.append(Setup(bar, side, "market", ref - side * risk, ref + side * rr * risk, max_hold=30))
    return simulate(df, out, COST)


def main():
    df4 = load("4h")
    results = {"source": "Bitstamp BTC/USD 1-minute data -> 4h bars (UTC)", "cost_pct_per_side": COST,
               "periods": PERIODS, "grid": GRID, "bars_4h": len(df4)}

    # Data check: daily closes from Bitstamp vs Coin Metrics reference rate.
    d1 = load("1d")
    try:
        from pinetest.crypto_data import load_panel
        cm = load_panel()["PriceUSD"]["btc"]
        # PriceUSD for date D is the 00:00 UTC rate at the end of D, i.e. the close of the UTC day.
        both = pd.concat([d1["close"], cm], axis=1, keys=["bitstamp", "cm"], sort=True).dropna().loc["2015":]
        rets = both.pct_change().dropna()
        results["data_check"] = {"daily_return_corr_vs_coinmetrics": round(float(rets.corr().iloc[0, 1]), 4),
                                 "median_abs_close_diff_pct": round(float((both.bitstamp / both.cm - 1).abs().median() * 100), 3)}
    except Exception as e:      # the Coin Metrics archive is optional for this study
        results["data_check"] = {"skipped": repr(e)}
    full_index = pd.date_range(df4.index[0], df4.index[-1], freq="4h")
    results["data_check"]["missing_4h_bars"] = int(len(full_index) - len(df4))

    rows = []
    for kind, lb, rr, filt in itertools.product(*GRID.values()):
        trades = simulate(df4, setups_for(df4, kind, lb, rr, filt), COST)
        per = by_period(df4, trades)
        rows.append({"kind": kind, "lb": lb, "rr": rr, "filter": filt,
                     **{p: summary(ts) for p, ts in per.items()}})
    results["grid_rows"] = rows

    eligible = [r for r in rows if r["train"].get("n", 0) >= 30]
    best = max(eligible, key=lambda r: r["train"]["avg_r"])
    cfg = {k: best[k] for k in GRID}
    results["chosen_on_train"] = cfg
    setups = setups_for(df4, cfg["kind"], cfg["lb"], cfg["rr"], cfg["filter"])
    risk_pct = {}
    for s in setups:
        entry = df4["open"].iloc[s.bar + 1]
        risk_pct[(s.bar, s.side)] = s.side * (entry - s.stop) / entry
    chosen = {}
    for cost in (0.0, 0.05, COST):
        per = by_period(df4, simulate(df4, setups, cost))
        chosen[f"cost_{cost}"] = {p: summary(ts) for p, ts in per.items()}
    per = by_period(df4, simulate(df4, setups, COST))
    for side, name in ((1, "long_only"), (-1, "short_only")):
        only = by_period(df4, simulate(df4, [s for s in setups if s.side == side], COST))
        chosen[name] = {p: summary(ts) for p, ts in only.items()}
    test_trades = per["test"]
    chosen["test_years"] = {
        str(y): summary([t for t in test_trades if df4.index[t.bar].year == y])
        for y in sorted({df4.index[t.bar].year for t in test_trades})}
    years = (df4.index[-1] - pd.Timestamp(PERIODS["test"][0])).days / 365.25
    n_test = len(test_trades)
    chosen["test_trades_per_year"] = round(n_test / years, 1)
    chosen["test_return_pct_per_year_at_1pct_risk"] = round(
        summary(test_trades).get("avg_r", 0) * n_test / years, 1) if n_test else 0.0

    rng = np.random.default_rng(SEED)
    real = summary(test_trades).get("avg_r", 0.0)
    plac = {}
    for mode in ("same_bars_random_side", "random_bars_same_side", "random_bars_random_side"):
        vals = np.array([summary(placebo(df4, test_trades, risk_pct, cfg["rr"], rng, mode)).get("avg_r", 0.0)
                         for _ in range(300)])
        plac[mode] = {"pct_below_real": round(100 * float((vals < real).mean()), 1),
                      "placebo_median_avg_r": round(float(np.median(vals)), 3)}
    chosen["placebo_test"] = plac
    results["chosen"] = chosen

    for key in ("pre", "train", "test"):
        vals = [r[key].get("avg_r", 0.0) for r in rows]
        results.setdefault("grid_summary", {})[key] = {
            "median_avg_r": round(float(np.median(vals)), 3), "n_positive": int(sum(v > 0 for v in vals)),
            "n_configs": len(vals)}
    tr = [r["train"].get("avg_r", 0.0) for r in rows]
    te = [r["test"].get("avg_r", 0.0) for r in rows]
    results["grid_summary"]["spearman_train_test"] = round(float(pd.Series(tr).rank().corr(pd.Series(te).rank())), 3)

    # Neighbour timeframes: the whole grid's medians and the chosen config.
    for tf in ("1h", "1d"):
        df = load(tf)
        tf_rows = []
        for kind, lb, rr, filt in itertools.product(*GRID.values()):
            per = by_period(df, simulate(df, setups_for(df, kind, lb, rr, filt), COST))
            tf_rows.append({"kind": kind, "lb": lb, "rr": rr, "filter": filt, **{p: summary(ts) for p, ts in per.items()}})
        pick = next(r for r in tf_rows if all(r[k] == cfg[k] for k in GRID))
        results[f"tf_{tf}"] = {
            "chosen_config": {p: pick[p] for p in PERIODS},
            "grid_median_avg_r": {p: round(float(np.median([r[p].get("avg_r", 0.0) for r in tf_rows])), 3) for p in PERIODS},
            "grid_n_positive": {p: int(sum(r[p].get("avg_r", 0.0) > 0 for r in tf_rows)) for p in PERIODS},
        }

    # Buy and hold over TEST for context.
    t0 = df4.loc[PERIODS["test"][0]:]
    results["btc_buy_hold_test_pct"] = round(100 * (t0["close"].iloc[-1] / t0["open"].iloc[0] - 1), 1)

    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    show = {k: v for k, v in results.items() if k != "grid_rows"}
    print(json.dumps(show, indent=1, default=str))


if __name__ == "__main__":
    main()
