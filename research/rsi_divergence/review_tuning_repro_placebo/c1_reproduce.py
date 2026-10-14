"""Check 1: independent reproduction of the walk-forward from data/tune_cache.pkl."""
import json, pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
d = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
configs, results, years = d["configs"], d["results"], d["years"]
print(len(configs), len(results), years)
TEST = list(range(2017, 2027))

# matrix of per-config per-year sums and counts
n = len(results)
S = np.zeros((n, len(years))); C = np.zeros((n, len(years))); SS = np.zeros((n, len(years)))
for i, r in enumerate(results):
    for j, y in enumerate(years):
        a = np.asarray(r[y], float)
        S[i, j] = a.sum(); C[i, j] = len(a); SS[i, j] = (a ** 2).sum()

def tstat(a):
    a = np.asarray(a, float)
    if len(a) < 2: return 0.0
    sd = a.std(ddof=1)
    return a.mean() / (sd / np.sqrt(len(a))) if sd > 0 else 0.0

def wf(window, metric, min_trades=20, round_score=True):
    picks = {}
    for y in TEST:
        cols = [j for j, x in enumerate(years) if (x < y and (window == 0 or x >= y - window))]
        s = S[:, cols].sum(1); c = C[:, cols].sum(1); ss = SS[:, cols].sum(1)
        with np.errstate(all="ignore"):
            mean = s / c
            var = (ss - c * mean ** 2) / (c - 1)
            t = mean / np.sqrt(var / c)
        score = mean if metric == "avg_r" else t
        if round_score:  # tune.py ranks on rounded values (3dp avg, 2dp t), first index wins ties
            score = np.round(score, 3 if metric == "avg_r" else 2)
        score = np.where(c >= min_trades, score, -np.inf)
        score = np.where(np.isnan(score), -np.inf, score)
        picks[y] = int(np.argmax(score))
    oos = {y: results[picks[y]][y] for y in TEST}
    return picks, oos

def st(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "avg_r": round(float(a.mean()), 3) if len(a) else 0, "t": round(tstat(a), 2),
            "eq_pct_1pct": round(100 * (np.prod(1 + 0.01 * a) - 1), 1) if len(a) else 0}

out = {}
for window in (3, 0):
    for metric in ("avg_r", "t"):
        for rnd in (True, False):
            picks, oos = wf(window, metric, round_score=rnd)
            allr = [r for y in TEST for r in oos[y]]
            ex17 = [r for y in TEST if y != 2017 for r in oos[y]]
            key = f"w{window or 'exp'}_{metric}_{'rounded' if rnd else 'exact'}"
            out[key] = {"oos": st(allr), "ex2017": st(ex17),
                        "per_year": {y: {"pick": picks[y], **st(oos[y])} for y in TEST},
                        "n_distinct": len(set(picks.values()))}
            print(key, out[key]["oos"], "ex17", out[key]["ex2017"], "picks", [picks[y] for y in TEST])

# detail for expanding avg_r: show configs and training avg R
picks, oos = wf(0, "avg_r")
detail = {}
for y in TEST:
    i = picks[y]
    cols = [j for j, x in enumerate(years) if x < y]
    tr = S[i, cols].sum() / C[i, cols].sum()
    # runner-ups
    s = S[:, cols].sum(1); c = C[:, cols].sum(1)
    with np.errstate(all="ignore"):
        m = np.where(c >= 20, s / c, -np.inf)
    top = np.argsort(-m)[:5]
    detail[y] = {"pick": i, "config": configs[i], "train_avg_r": round(float(tr), 3), "train_n": int(C[i, cols].sum()),
                 "oos": st(oos[y]), "oos_r": oos[y],
                 "top5_train": [(int(k), round(float(m[k]), 3), int(c[k])) for k in top]}
    print(y, i, configs[i], detail[y]["train_avg_r"], detail[y]["train_n"], st(oos[y]), detail[y]["top5_train"])
out["expanding_avg_r_detail"] = detail
(OUT / "c1_reproduce.json").write_text(json.dumps(out, indent=1, default=str))
