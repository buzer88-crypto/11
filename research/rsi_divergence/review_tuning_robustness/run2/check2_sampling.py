"""Check 2: sampling luck.

(a) 5 random subsets of 600 configs from the cache (plus 1000 more for a distribution), each run
    through the headline walk-forward (expanding window, avg R, min 20 trades).
(b) A fresh sample of 600 new configs (seed 777, same space, same simulator, 0.1%/side costs),
    simulated on 4h, then the same walk-forward (and the other three tune.py variants).
Run from /home/user/11 with PYTHONPATH=. ; uses 2 processes for the fresh sample.
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUTD = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wf import TEST_YEARS, YEARS, load_cache, stats, summarize, walk_forward  # noqa: E402
import tune  # noqa: E402  (evaluate/_init/SPACE only; nothing is written by importing)

FRESH_SEED = 777
N_FRESH = 600
DOMINANT = {"kind": "hidden", "rsi_len": 21, "lb": 8, "rr": 4.0, "buffer_atr": 0.0, "max_hold": 60,
            "confirm": "ema_9_21", "trend": False, "sides": "long", "extreme": False}


def fast_expanding_avg(results, idx, min_trades=20):
    """Vectorised headline rule (expanding, avg_r rounded to 3 dp, first index wins ties)."""
    idx = np.asarray(idx)
    N = np.array([[len(results[i][y]) for y in YEARS] for i in idx], dtype=float)
    S = np.array([[sum(results[i][y]) for y in YEARS] for i in idx], dtype=float)
    cN, cS = np.cumsum(N, 1), np.cumsum(S, 1)
    picks, oos = {}, {}
    for y in TEST_YEARS:
        k = YEARS.index(y) - 1                    # through year y-1
        n, s = cN[:, k], cS[:, k]
        with np.errstate(invalid="ignore", divide="ignore"):
            sc = np.where(n >= min_trades, np.round(s / n, 3), -np.inf)
        j = int(np.argmax(sc)) if np.isfinite(sc).any() else None
        picks[y] = int(idx[j]) if j is not None else None
        oos[y] = list(results[picks[y]][y]) if j is not None else []
    return picks, oos


def fresh_configs(seed, n, exclude=()):
    rng = np.random.default_rng(seed)
    keys = list(tune.SPACE)
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: tune.SPACE[k][rng.integers(len(tune.SPACE[k]))] for k in keys}
        cfg = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
        key = tuple(cfg[k] for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def main():
    configs, results = load_cache()
    dom = configs.index(DOMINANT)
    # Sanity: the vectorised rule reproduces the headline on the full cache.
    p, o = fast_expanding_avg(results, range(len(results)))
    full = summarize(p, o, configs)
    print("full cache:", full["n"], full["avg_r"], full["t"], full["avg_r_ex2017"])
    assert full["n"] == 67 and abs(full["avg_r"] - 0.581) < 1e-3

    rng = np.random.default_rng(424242)
    subsets = []
    for s in range(1005):
        idx = np.sort(rng.choice(len(configs), 600, replace=False))
        p, o = fast_expanding_avg(results, idx)
        r = summarize(p, o, configs)
        r["contains_dominant"] = bool(dom in set(idx.tolist()))
        r["dominant_picked_years"] = sum(1 for v in p.values() if v == dom)
        subsets.append(r)
    first5 = [{k: r[k] for k in ("n", "avg_r", "t", "n_ex2017", "avg_r_ex2017", "t_ex2017", "n_distinct",
                                 "long_only_share", "contains_dominant", "dominant_picked_years", "per_year_avg_r", "picks")}
              for r in subsets[:5]]
    for r in first5:
        print("subset:", r["n"], r["avg_r"], r["t"], r["n_ex2017"], r["avg_r_ex2017"], r["contains_dominant"],
              r["dominant_picked_years"])
    av = np.array([r["avg_r"] for r in subsets])
    avx = np.array([r["avg_r_ex2017"] for r in subsets])
    has = np.array([r["contains_dominant"] for r in subsets])
    dist = {"n_subsets": len(subsets),
            "avg_r": {"median": round(float(np.median(av)), 3), "q10": round(float(np.quantile(av, .1)), 3),
                      "q90": round(float(np.quantile(av, .9)), 3),
                      "share_ge_0.581_pct": round(100 * float((av >= 0.581).mean()), 1),
                      "share_positive_pct": round(100 * float((av > 0).mean()), 1)},
            "avg_r_ex2017": {"median": round(float(np.median(avx)), 3), "q10": round(float(np.quantile(avx, .1)), 3),
                             "q90": round(float(np.quantile(avx, .9)), 3),
                             "share_positive_pct": round(100 * float((avx > 0).mean()), 1)},
            "with_dominant": {"count": int(has.sum()), "median_avg_r": round(float(np.median(av[has])), 3),
                              "median_avg_r_ex2017": round(float(np.median(avx[has])), 3)},
            "without_dominant": {"count": int((~has).sum()), "median_avg_r": round(float(np.median(av[~has])), 3),
                                 "median_avg_r_ex2017": round(float(np.median(avx[~has])), 3),
                                 "share_positive_ex2017_pct": round(100 * float((avx[~has] > 0).mean()), 1)}}
    print(json.dumps(dist, indent=1))

    # Full cache with the dominant config removed.
    no_dom = [i for i in range(len(configs)) if i != dom]
    p, o = fast_expanding_avg(results, no_dom)
    without_dom = summarize(p, o, configs)
    print("cache minus dominant:", without_dom["n"], without_dom["avg_r"], without_dom["avg_r_ex2017"])

    # (b) fresh sample.
    fresh = fresh_configs(FRESH_SEED, N_FRESH)
    cache_keys = {tuple(c.values()) for c in configs}
    overlap = sum(tuple(c.values()) in cache_keys for c in fresh)
    fresh_path = OUTD / "fresh_sample_cache.pkl"
    if fresh_path.exists():
        fres = pickle.loads(fresh_path.read_bytes())["results"]
    else:
        with ProcessPoolExecutor(1, initializer=tune._init) as ex:
            fres = list(ex.map(tune.evaluate, fresh, chunksize=10))
        fresh_path.write_bytes(pickle.dumps({"configs": fresh, "results": fres, "years": YEARS, "seed": FRESH_SEED}))
    fresh_out = {"seed": FRESH_SEED, "n_configs": N_FRESH, "overlap_with_cache": overlap,
                 "contains_dominant": DOMINANT in fresh, "variants": {}}
    for window in (0, 3):
        for metric in ("avg_r", "t"):
            for mt in (10, 20, 30):
                p, o = walk_forward(fres, window, metric, mt)
                r = summarize(p, o, fresh)
                r["picked_configs"] = {y: fresh[i] for y, i in p.items() if i is not None}
                fresh_out["variants"][f"window_{window or 'expanding'}_{metric}_min{mt}"] = r
                print("fresh", window or "exp", metric, mt, r["n"], r["avg_r"], r["t"], r["n_ex2017"],
                      r["avg_r_ex2017"], r["n_distinct"], r["long_only_share"])
    # Null for the fresh sample: random long-only config per year.
    long_idx = [i for i, c in enumerate(fresh) if c["sides"] == "long"]
    nr = np.random.default_rng(1)
    null = []
    for _ in range(4000):
        rs = [r for y in TEST_YEARS for r in fres[long_idx[int(nr.integers(len(long_idx)))]][y]]
        null.append(np.mean(rs) if rs else 0.0)
    null = np.array(null)
    h = fresh_out["variants"]["window_expanding_avg_r_min20"]
    fresh_out["headline_rule_pct_random_long_below"] = round(100 * float(np.mean(null < h["avg_r"])), 1)
    fresh_out["random_long_null_median"] = round(float(np.median(null)), 3)
    print("fresh headline pct random long below:", fresh_out["headline_rule_pct_random_long_below"])

    out = {"full_cache_headline_check": {k: full[k] for k in ("n", "avg_r", "t", "avg_r_ex2017")},
           "subsets_first5": first5, "subsets_distribution": dist,
           "cache_minus_dominant": {k: without_dom[k] for k in ("n", "avg_r", "t", "n_ex2017", "avg_r_ex2017", "t_ex2017",
                                                                "per_year_avg_r", "per_year_n")},
           "cache_minus_dominant_picks": {y: configs[i] for y, i in without_dom["picks"].items() if i is not None},
           "fresh_sample": fresh_out}
    (OUTD / "check2_sampling.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
