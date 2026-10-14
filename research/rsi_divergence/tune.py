"""Walk-forward tuning of RSI divergence (+ MACD/EMA confirmation) on BTC/USD 4h.

Every configuration in a random sample of the parameter space is simulated once over the whole
history; trades are then grouped by the calendar year of their signal. For each year Y the
configuration with the best record over the previous `window` years is chosen and traded in
Y only, so every reported trade is out of sample for the tuning. Selection rules: average R
(at least `min_trades` trades in the window) and t-statistic of R.

Baselines: the same walk-forward picking a random configuration each year (2000 draws), the
TradingView default (regular, RSI 14, pivots 5/5) traded every year, and BTC buy and hold.
Costs 0.1% per side; results in R and as equity risking 1% per trade.
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

OUT = Path(__file__).parent
CACHE = Path(__file__).resolve().parents[2] / "data" / "tune_cache.pkl"   # configs + R per year
SPACE = {
    "kind": ["regular", "hidden"], "rsi_len": [7, 9, 14, 21], "lb": [2, 3, 5, 8],
    "rr": [1.0, 1.5, 2.0, 3.0, 4.0], "buffer_atr": [0.0, 0.25, 0.5, 1.0], "max_hold": [12, 30, 60],
    "confirm": ["none", "macd_cross", "ema50", "ema_9_21", "ema20_reclaim"], "trend": [False, True],
    "sides": ["both", "long"], "extreme": [False, True],
}
N_SAMPLE = 2400
SEED = 20261014
YEARS = list(range(2014, 2027))
TEST_YEARS = list(range(2017, 2027))
DEFAULT = {"kind": "regular", "rsi_len": 14, "lb": 5, "rr": 2.0, "buffer_atr": 0.0, "max_hold": 30,
           "confirm": "none", "trend": False, "sides": "both", "extreme": False}

_df = None


def _init():
    global _df
    _df = load("4h")


def evaluate(cfg):
    df = _df
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"],
                             rr=cfg["rr"], buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"],
                             confirm=cfg["confirm"], trend=cfg["trend"], extreme=cfg["extreme"],
                             sides=(1, -1) if cfg["sides"] == "both" else (1,))
    years = df.index.year.to_numpy()
    by_year = {y: [] for y in YEARS}
    for t in simulate(df, setups, COST):
        y = int(years[t.bar])
        if y in by_year:
            by_year[y].append(round(t.r, 4))
    return by_year


def sample_space():
    rng = np.random.default_rng(SEED)
    keys = list(SPACE)
    seen, out = set(), [DEFAULT]
    seen.add(tuple(DEFAULT[k] for k in keys))
    while len(out) < N_SAMPLE:
        cfg = {k: SPACE[k][rng.integers(len(SPACE[k]))] for k in keys}
        key = tuple(cfg[k] for k in keys)
        if key not in seen:
            seen.add(key)
            out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()})
    return out


def stats(rs):
    r = np.asarray(rs, dtype=float)
    if len(r) == 0:
        return {"n": 0, "avg_r": 0.0, "total_r": 0.0, "t": 0.0}
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"n": int(len(r)), "avg_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2),
            "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0}


def equity_pct(rs, risk=0.01):
    return round(100 * (float(np.prod(1 + risk * np.asarray(rs, dtype=float))) - 1), 1) if len(rs) else 0.0


def walk_forward(results, window, metric, min_trades=20):
    picks, oos = {}, []
    for y in TEST_YEARS:
        train_years = [x for x in YEARS if (y - window <= x < y)] if window else [x for x in YEARS if x < y]
        best, best_score = None, -np.inf
        for i, by_year in enumerate(results):
            rs = [r for x in train_years for r in by_year[x]]
            if len(rs) < min_trades:
                continue
            s = stats(rs)
            score = s["avg_r"] if metric == "avg_r" else s["t"]
            if score > best_score:
                best, best_score = i, score
        picks[y] = best
        oos.extend(results[best][y] if best is not None else [])
    return picks, oos


def main():
    configs = sample_space()
    with ProcessPoolExecutor(3, initializer=_init) as ex:
        results = list(ex.map(evaluate, configs, chunksize=20))
    CACHE.write_bytes(pickle.dumps({"configs": configs, "results": results, "years": YEARS}))

    df = load("4h")
    close = df["close"]
    bh_year = {y: round(100 * (close.loc[str(y)].iloc[-1] / close.loc[str(y)].iloc[0] - 1), 1) for y in TEST_YEARS}
    bh_total = round(100 * (close.loc[str(TEST_YEARS[-1])].iloc[-1] / close.loc[str(TEST_YEARS[0])].iloc[0] - 1), 1)

    out = {"space": SPACE, "space_size": int(np.prod([len(v) for v in SPACE.values()])), "n_sampled": len(configs),
           "cost_pct_per_side": COST, "test_years": TEST_YEARS, "btc_buy_hold_pct_by_year": bh_year,
           "btc_buy_hold_pct_total": bh_total, "walk_forward": {}}

    rng = np.random.default_rng(SEED)
    for window in (3, 0):                       # 3-year rolling, 0 = expanding since 2014
        for metric in ("avg_r", "t"):
            picks, oos = walk_forward(results, window, metric)
            per_year = {}
            for y in TEST_YEARS:
                i = picks[y]
                rs = results[i][y] if i is not None else []
                per_year[str(y)] = {"config": configs[i] if i is not None else None, **stats(rs),
                                    "equity_pct_1pct_risk": equity_pct(rs)}
            # Nulls: pick a random configuration each year, from all configurations and from the
            # long-only ones (the picks are long-only, and BTC rose ~85x over 2017-2026).
            long_idx = [i for i, c in enumerate(configs) if c["sides"] == "long"]
            nulls = {}
            for name, pool in (("all", list(range(len(results)))), ("long_only", long_idx)):
                null = []
                for _ in range(2000):
                    rs_all = []
                    for y in TEST_YEARS:
                        rs_all.extend(results[pool[int(rng.integers(len(pool)))]][y])
                    null.append(np.mean(rs_all) if rs_all else 0.0)
                nulls[name] = np.array(null)
            real = stats(oos)
            ex2017 = [r for y in TEST_YEARS if y != 2017 for r in (results[picks[y]][y] if picks[y] is not None else [])]
            out["walk_forward"][f"window_{window or 'expanding'}_{metric}"] = {
                "oos": real, "oos_equity_pct_1pct_risk": equity_pct(oos), "per_year": per_year,
                "random_pick_median_avg_r": round(float(np.median(nulls["all"])), 3),
                "pct_random_picks_below": round(100 * float(np.mean(nulls["all"] < real["avg_r"])), 1),
                "random_long_pick_median_avg_r": round(float(np.median(nulls["long_only"])), 3),
                "pct_random_long_picks_below": round(100 * float(np.mean(nulls["long_only"] < real["avg_r"])), 1),
                "excluding_2017": {**stats(ex2017), "equity_pct_1pct_risk": equity_pct(ex2017)},
                "picked_long_only_share": round(float(np.mean([configs[picks[y]]["sides"] == "long" for y in TEST_YEARS if picks[y] is not None])), 2),
                "n_distinct_configs_picked": len({picks[y] for y in TEST_YEARS}),
            }

    default_rs = [r for y in TEST_YEARS for r in results[0][y]]
    out["tradingview_default"] = {"oos": stats(default_rs), "equity_pct_1pct_risk": equity_pct(default_rs),
                                  "per_year": {str(y): stats(results[0][y]) for y in TEST_YEARS}}

    # How good the best configuration looks with hindsight (tuned on the very years it is scored on).
    hind = [stats([r for y in TEST_YEARS for r in by_year[y]]) for by_year in results]
    k = int(np.argmax([h["avg_r"] if h["n"] >= 100 else -9 for h in hind]))
    out["hindsight_best"] = {"config": configs[k], **hind[k],
                             "equity_pct_1pct_risk": equity_pct([r for y in TEST_YEARS for r in results[k][y]])}
    all_avg = np.array([h["avg_r"] for h in hind if h["n"] >= 100])
    out["all_configs_2017_2026"] = {"n_with_100_trades": int(len(all_avg)), "median_avg_r": round(float(np.median(all_avg)), 3),
                                    "share_positive_pct": round(100 * float((all_avg > 0).mean()), 1)}

    (OUT / "tune_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
