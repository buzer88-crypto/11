"""Check 4: drift-free rerun. Each calendar year's mean 4h log return is removed from the
price path (OHLC of every bar scaled by exp(-cumulative per-year mean log return)), so every
year nets to ~0% but intra-year shape, volatility and divergences are kept.
(a) rerun the actual picks on the detrended series; (b) redo the whole tuning (same 2400 configs)
and the expanding/avg-R walk-forward on it; (c) random-pick null on the detrended cache."""
import json, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from lib import TEST, YEARS, by_year, cache, load, trades_for

OUT = Path(__file__).parent
PICKS = {2017: 680, 2018: 441, 2019: 441, 2020: 441, 2021: 441, 2022: 441, 2023: 441, 2024: 194, 2025: 1566, 2026: 441}


def detrended():
    df = load("4h").copy()
    lr = np.log(df["close"]).diff().fillna(0.0)
    m = lr.groupby(df.index.year).transform("mean")
    f = np.exp(-m.cumsum())
    for k in ("open", "high", "low", "close"):
        df[k] = df[k] * f
    return df


_df = None


def _init():
    global _df
    _df = detrended()


def evaluate(cfg):
    yb = by_year(_df, trades_for(_df, cfg))
    return {y: [round(t.r, 4) for t in yb[y]] for y in YEARS}


def wf_expanding_avg(results, min_trades=20):
    picks, oos = {}, {}
    for y in TEST:
        best, bs = None, -np.inf
        for i, r in enumerate(results):
            rs = [v for x in YEARS if x < y for v in r[x]]
            if len(rs) < min_trades:
                continue
            s = round(float(np.mean(rs)), 3)
            if s > bs:
                best, bs = i, s
        picks[y] = best
        oos[y] = results[best][y] if best is not None else []
    return picks, oos


def st(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return {"n": 0}
    sd = a.std(ddof=1) if len(a) > 1 else 0
    return {"n": int(len(a)), "avg_r": round(float(a.mean()), 3), "t": round(float(a.mean() / (sd / np.sqrt(len(a)))), 2) if sd > 0 else 0.0,
            "eq_pct_1pct": round(100 * (np.prod(1 + 0.01 * a) - 1), 1)}


def main():
    df = detrended()
    raw = load("4h")
    yr_ret = {int(y): round(float(g.close.iloc[-1] / g.open.iloc[0] - 1) * 100, 1) for y, g in df.groupby(df.index.year)}
    yr_ret_raw = {int(y): round(float(g.close.iloc[-1] / g.open.iloc[0] - 1) * 100, 1) for y, g in raw.groupby(raw.index.year)}
    d = cache()
    res = {"yearly_return_pct_detrended": yr_ret, "yearly_return_pct_raw": yr_ret_raw}
    # (a) actual picks on detrended data
    allr, ex = [], []
    per = {}
    memo = {}
    for y in TEST:
        i = PICKS[y]
        if i not in memo:
            memo[i] = by_year(df, trades_for(df, d["configs"][i]))
        rs = [t.r for t in memo[i][y]]
        per[y] = st(rs)
        allr += rs
        if y != 2017:
            ex += rs
    # the main pick 441 over all test years on detrended vs raw
    raw441 = by_year(raw, trades_for(raw, d["configs"][441]))
    res["a_picks_on_detrended"] = {"oos": st(allr), "ex2017": st(ex), "per_year": per}
    res["config441_all_test_years"] = {"raw": st([t.r for y in TEST for t in raw441[y]]),
                                       "detrended": st([t.r for y in TEST for t in memo[441][y]]),
                                       "raw_2014_2016": st([t.r for y in (2014, 2015, 2016) for t in raw441[y]]),
                                       "detrended_2014_2016": st([t.r for y in (2014, 2015, 2016) for t in memo[441][y]])}
    print(json.dumps(res, indent=1), flush=True)
    (OUT / "c4_detrend.json").write_text(json.dumps(res, indent=1))
    if "--full" not in sys.argv:
        return
    # (b) full re-tune on detrended data
    t0 = time.time()
    cpath = OUT / "detrended_cache.pkl"
    if cpath.exists():
        results = pickle.loads(cpath.read_bytes())
    else:
        with ProcessPoolExecutor(2, initializer=_init) as ex_:
            results = list(ex_.map(evaluate, d["configs"], chunksize=20))
        cpath.write_bytes(pickle.dumps(results))
    print("tuned in", round(time.time() - t0), "s", flush=True)
    picks, oos = wf_expanding_avg(results)
    allr = [v for y in TEST for v in oos[y]]
    ex17 = [v for y in TEST if y != 2017 for v in oos[y]]
    rng = np.random.default_rng(7)
    long_idx = [i for i, c in enumerate(d["configs"]) if c["sides"] == "long"]
    nulls = {}
    for name, pool in (("all", list(range(len(results)))), ("long_only", long_idx)):
        nl = []
        for _ in range(2000):
            rs = []
            for y in TEST:
                rs += results[pool[int(rng.integers(len(pool)))]][y]
            nl.append(np.mean(rs) if rs else 0.0)
        nulls[name] = np.array(nl)
    real = float(np.mean(allr))
    res["b_full_retune_detrended_expanding_avg_r"] = {
        "oos": st(allr), "ex2017": st(ex17),
        "picks": {y: {"idx": picks[y], "config": d["configs"][picks[y]], **st(oos[y])} for y in TEST},
        "pct_random_picks_below": round(100 * float((nulls["all"] < real).mean()), 1),
        "pct_random_long_picks_below": round(100 * float((nulls["long_only"] < real).mean()), 1),
        "random_pick_median": round(float(np.median(nulls["all"])), 3),
        "random_long_pick_median": round(float(np.median(nulls["long_only"])), 3)}
    hind = []
    for r in results:
        rs = [v for y in TEST for v in r[y]]
        hind.append((np.mean(rs) if len(rs) >= 100 else np.nan))
    hind = np.array(hind)
    res["b_all_configs_detrended_2017_2026"] = {"n_with_100": int(np.isfinite(hind).sum()),
                                                "median_avg_r": round(float(np.nanmedian(hind)), 3),
                                                "share_positive_pct": round(100 * float((hind[np.isfinite(hind)] > 0).mean()), 1)}
    print(json.dumps({k: res[k] for k in ("b_full_retune_detrended_expanding_avg_r", "b_all_configs_detrended_2017_2026")}, indent=1, default=str))
    (OUT / "c4_detrend.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
