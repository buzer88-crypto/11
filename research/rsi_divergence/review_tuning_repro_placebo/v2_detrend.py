"""Session-2 check 4 (drift-free): remove each calendar year's log trend from the 4h series and
(a) rerun the actual expanding/avg-R picks on it, (b) re-tune all 2400 sampled configs on it and
redo the four walk-forward variants. Detrending: every bar's OHLC is multiplied by
exp(-D_t), D_t = cumulative sum of the bar's calendar-year mean 4h log return (close to close),
so each year nets to ~0% while intra-year swings, volatility and divergences are kept.
Usage: python v2_detrend.py [--full]   (full re-tune runs in 1 process)."""
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

OUT = Path(__file__).parent
YEARS = list(range(2014, 2027))
TEST = list(range(2017, 2027))
PICKS = {2017: 680, 2018: 441, 2019: 441, 2020: 441, 2021: 441, 2022: 441, 2023: 441, 2024: 194, 2025: 1566, 2026: 441}


def detrend(df):
    lc = np.log(df["close"].to_numpy(float))
    step = np.r_[0.0, np.diff(lc)]
    yr = df.index.year.to_numpy()
    mu = pd.Series(step).groupby(yr).transform("mean").to_numpy()
    f = np.exp(-np.cumsum(mu))
    out = df.copy()
    for k in ("open", "high", "low", "close"):
        out[k] = df[k].to_numpy(float) * f
    return out


def run_cfg(df, cfg):
    s = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                        buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"],
                        trend=cfg["trend"], extreme=cfg["extreme"], sides=(1, -1) if cfg["sides"] == "both" else (1,))
    yr = df.index.year.to_numpy()
    out = {y: [] for y in YEARS}
    for t in simulate(df, s, COST):
        y = int(yr[t.bar])
        if y in out:
            out[y].append(round(t.r, 4))
    return out


_DF = None


def _init():
    global _DF
    _DF = detrend(load("4h"))


def _ev(cfg):
    return run_cfg(_DF, cfg)


def st(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return {"n": 0, "avg_r": 0.0, "t": 0.0}
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return {"n": int(len(a)), "avg_r": round(float(a.mean()), 3),
            "t": round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else 0.0,
            "eq_pct_1pct": round(100 * (float(np.prod(1 + 0.01 * a)) - 1), 1)}


def wf(results, window, metric, min_trades=20):
    picks = {}
    for y in TEST:
        tys = [x for x in YEARS if x < y and (window == 0 or x >= y - window)]
        best, bs = None, -np.inf
        for i, r in enumerate(results):
            a = [v for x in tys for v in r[x]]
            if len(a) < min_trades:
                continue
            s = st(a)
            sc = s["avg_r"] if metric == "avg_r" else s["t"]
            if sc > bs:
                best, bs = i, sc
        picks[y] = best
    return picks


def main():
    raw = load("4h")
    df = detrend(raw)
    cache = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
    cfgs = cache["configs"]
    res = {"year_return_pct_detrended": {int(y): round(100 * float(g.close.iloc[-1] / g.close.iloc[0] - 1), 2)
                                         for y, g in df.groupby(df.index.year)}}
    # (a) actual picks on detrended data, and the raw run for reference (must equal the cache)
    memo_d, memo_r = {}, {}
    a_all, a_ex, per = [], [], {}
    raw_check = 0.0
    for y in TEST:
        i = PICKS[y]
        if i not in memo_d:
            memo_d[i] = run_cfg(df, cfgs[i])
            memo_r[i] = run_cfg(raw, cfgs[i])
        raw_check = max(raw_check, max([abs(p - q) for p, q in zip(memo_r[i][y], cache["results"][i][y])] or [0]))
        assert len(memo_r[i][y]) == len(cache["results"][i][y])
        per[y] = {"detrended": st(memo_d[i][y]), "raw": st(memo_r[i][y])}
        a_all += memo_d[i][y]
        if y != 2017:
            a_ex += memo_d[i][y]
    res["a_picks_on_detrended"] = {"all": st(a_all), "ex2017": st(a_ex), "per_year": per,
                                   "raw_rerun_max_abs_diff_vs_cache": raw_check}
    for i in (441,):
        res[f"config_{i}_2017_2026"] = {"raw": st([v for y in TEST for v in memo_r[i][y]]),
                                        "detrended": st([v for y in TEST for v in memo_d[i][y]]),
                                        "raw_2014_2016": st([v for y in (2014, 2015, 2016) for v in memo_r[i][y]]),
                                        "detrended_2014_2016": st([v for y in (2014, 2015, 2016) for v in memo_d[i][y]])}
    print(json.dumps(res, indent=1), flush=True)
    (OUT / "v2_detrend.json").write_text(json.dumps(res, indent=1))
    if "--full" not in sys.argv:
        return
    t0 = time.time()
    with ProcessPoolExecutor(1, initializer=_init) as ex:
        results = list(ex.map(_ev, cfgs, chunksize=40))
    (OUT / "v2_detrended_cache.pkl").write_bytes(pickle.dumps(results))
    print("re-tuned in", round(time.time() - t0), "s", flush=True)
    res["b_full_retune"] = {}
    rng = np.random.default_rng(99)
    long_idx = [i for i, c in enumerate(cfgs) if c["sides"] == "long"]
    for window in (0, 3):
        for metric in ("avg_r", "t"):
            p = wf(results, window, metric)
            allr = [v for y in TEST if p[y] is not None for v in results[p[y]][y]]
            ex17 = [v for y in TEST if y != 2017 and p[y] is not None for v in results[p[y]][y]]
            null = []
            for _ in range(2000):
                rs = [v for y in TEST for v in results[long_idx[int(rng.integers(len(long_idx)))]][y]]
                null.append(np.mean(rs) if rs else 0.0)
            null = np.array(null)
            real = float(np.mean(allr)) if allr else 0.0
            res["b_full_retune"][f"{'expanding' if window == 0 else 'w3'}_{metric}"] = {
                "all": st(allr), "ex2017": st(ex17), "picks": p,
                "pick_cfgs": {y: cfgs[p[y]] for y in TEST if p[y] is not None},
                "pct_random_long_picks_below": round(100 * float((null < real).mean()), 1),
                "random_long_pick_median": round(float(np.median(null)), 3)}
    print(json.dumps(res["b_full_retune"], indent=1, default=str), flush=True)
    (OUT / "v2_detrend.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
