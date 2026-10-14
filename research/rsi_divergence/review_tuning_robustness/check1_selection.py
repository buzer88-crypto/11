"""Check 1: selection sensitivity of the walk-forward (from data/tune_cache.pkl).

Grid: min_trades {10,15,20,30,40} x window {2,3,4,5,expanding} x metric {avg_r,t,total_r,median_r}.
Also: expanding-window start year 2015/2016, and the OOS result of the rank-k pick (k=1..20)
and of the top-k basket under the headline rule (expanding, avg_r, min 20).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wf import TEST_YEARS, YEARS, load_cache, score, stats, summarize, train_years, walk_forward  # noqa: E402

OUT = Path(__file__).parent


def main():
    configs, results = load_cache()
    rows = []
    for metric in ("avg_r", "t", "total_r", "median_r"):
        for window in (2, 3, 4, 5, 0):
            for mt in (10, 15, 20, 30, 40):
                picks, oos = walk_forward(results, window, metric, mt)
                s = summarize(picks, oos, configs)
                rows.append({"metric": metric, "window": window or "expanding", "min_trades": mt, **s})
                print(metric, window or "exp", mt, s["n"], s["avg_r"], s["t"], s["n_ex2017"], s["avg_r_ex2017"],
                      s["n_distinct"], flush=True)

    # Start-year sensitivity of the expanding window (headline metric/min_trades).
    start_rows = []
    for start in (2014, 2015, 2016):
        for mt in (10, 15, 20):
            picks, oos = walk_forward(results, 0, "avg_r", mt, start=start)
            s = summarize(picks, oos, configs)
            start_rows.append({"start": start, "min_trades": mt, **s})
            print("start", start, mt, s["n"], s["avg_r"], s["avg_r_ex2017"], flush=True)

    # Rank-k picks and top-k baskets under the headline rule.
    rank_rows = []
    ranked = {}
    for y in TEST_YEARS:
        ty = train_years(y, 0)
        sc = []
        for i, by in enumerate(results):
            rs = [r for x in ty for r in by[x]]
            if len(rs) >= 20:
                sc.append((score(rs, "avg_r"), -i, i))
        sc.sort(reverse=True)
        ranked[y] = [i for _, _, i in sc]
    for k in range(1, 21):
        rs = [r for y in TEST_YEARS for r in results[ranked[y][k - 1]][y]]
        ex = [r for y in TEST_YEARS if y != 2017 for r in results[ranked[y][k - 1]][y]]
        rank_rows.append({"rank": k, **stats(rs), "avg_r_ex2017": stats(ex)["avg_r"], "n_ex2017": len(ex)})
    basket = []
    for k in (1, 3, 5, 10, 25, 50, 100):
        # Equal weight per config: average of each config's yearly OOS mean R, then over all trades pooled.
        rs = [r for y in TEST_YEARS for i in ranked[y][:k] for r in results[i][y]]
        ex = [r for y in TEST_YEARS if y != 2017 for i in ranked[y][:k] for r in results[i][y]]
        basket.append({"top_k": k, **stats(rs), "avg_r_ex2017": stats(ex)["avg_r"], "n_ex2017": len(ex)})

    # Null: a random config each year (all / long-only), same as tune.py, 4000 draws.
    rng = np.random.default_rng(7)
    long_idx = [i for i, c in enumerate(configs) if c["sides"] == "long"]
    nulls = {}
    for name, pool in (("all", list(range(len(results)))), ("long_only", long_idx)):
        v, vex = [], []
        for _ in range(4000):
            ch = {y: pool[int(rng.integers(len(pool)))] for y in TEST_YEARS}
            rs = [r for y in TEST_YEARS for r in results[ch[y]][y]]
            ex = [r for y in TEST_YEARS if y != 2017 for r in results[ch[y]][y]]
            v.append(np.mean(rs) if rs else 0.0)
            vex.append(np.mean(ex) if ex else 0.0)
        nulls[name] = (np.array(v), np.array(vex))
    for r in rows + start_rows:
        r["pct_random_long_below"] = round(100 * float(np.mean(nulls["long_only"][0] < r["avg_r"])), 1)
        r["pct_random_long_below_ex2017"] = round(100 * float(np.mean(nulls["long_only"][1] < r["avg_r_ex2017"])), 1)

    avg = np.array([r["avg_r"] for r in rows])
    avg_ex = np.array([r["avg_r_ex2017"] for r in rows])
    head = next(r for r in rows if r["metric"] == "avg_r" and r["window"] == "expanding" and r["min_trades"] == 20)
    summary = {
        "n_variants": len(rows),
        "headline": {k: head[k] for k in ("n", "avg_r", "t", "n_ex2017", "avg_r_ex2017")},
        "variants_avg_r": {"median": round(float(np.median(avg)), 3), "mean": round(float(avg.mean()), 3),
                           "q25": round(float(np.quantile(avg, .25)), 3), "q75": round(float(np.quantile(avg, .75)), 3),
                           "share_ge_headline_pct": round(100 * float((avg >= head["avg_r"]).mean()), 1),
                           "share_positive_pct": round(100 * float((avg > 0).mean()), 1),
                           "share_t_ge_2_pct": round(100 * float(np.mean([r["t"] >= 2 for r in rows])), 1)},
        "variants_avg_r_ex2017": {"median": round(float(np.median(avg_ex)), 3), "mean": round(float(avg_ex.mean()), 3),
                                  "q25": round(float(np.quantile(avg_ex, .25)), 3), "q75": round(float(np.quantile(avg_ex, .75)), 3),
                                  "share_positive_pct": round(100 * float((avg_ex > 0).mean()), 1),
                                  "share_t_ex2017_ge_2_pct": round(100 * float(np.mean([r["t_ex2017"] >= 2 for r in rows])), 1)},
        "by_metric": {m: {"median_avg_r": round(float(np.median([r["avg_r"] for r in rows if r["metric"] == m])), 3),
                          "median_avg_r_ex2017": round(float(np.median([r["avg_r_ex2017"] for r in rows if r["metric"] == m])), 3)}
                      for m in ("avg_r", "t", "total_r", "median_r")},
        "rank_of_headline_among_variants": int((avg > head["avg_r"]).sum()) + 1,
        "random_long_pick_null": {"median": round(float(np.median(nulls["long_only"][0])), 3),
                                  "p95": round(float(np.quantile(nulls["long_only"][0], .95)), 3),
                                  "median_ex2017": round(float(np.median(nulls["long_only"][1])), 3),
                                  "p95_ex2017": round(float(np.quantile(nulls["long_only"][1], .95)), 3)},
        "random_all_pick_null": {"median": round(float(np.median(nulls["all"][0])), 3),
                                 "p95": round(float(np.quantile(nulls["all"][0], .95)), 3)},
    }
    out = {"summary": summary, "rows": rows, "start_year_rows": start_rows, "rank_k_rows": rank_rows,
           "top_k_baskets": basket, "configs_picked": {str(i): configs[i] for i in
                                                        sorted({p for r in rows + start_rows for p in r["picks"].values() if p is not None})}}
    (OUT / "check1_selection.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    print("rank-k:", [(r["rank"], r["n"], r["avg_r"], r["avg_r_ex2017"]) for r in rank_rows])
    print("baskets:", [(r["top_k"], r["n"], r["avg_r"], r["t"], r["avg_r_ex2017"]) for r in basket])


if __name__ == "__main__":
    main()
