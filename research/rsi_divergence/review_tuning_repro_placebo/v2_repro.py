"""Session-2 check 1: reproduce the walk-forward picks / OOS numbers from data/tune_cache.pkl
with independent code (plain loops; tune.py ranks rounded scores, first index wins ties)."""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
d = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
cfgs, res, years = d["configs"], d["results"], d["years"]
TEST = list(range(2017, 2027))
print("n configs", len(cfgs), "years", years)


def tstat(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(len(a)))) if sd > 0 else 0.0


def score(a, metric, rounded):
    if metric == "avg_r":
        v = float(np.mean(a))
        return round(v, 3) if rounded else v
    v = tstat(a)
    return round(v, 2) if rounded else v


def wf(window, metric, rounded=True, min_trades=20):
    picks = {}
    for y in TEST:
        tr_years = [x for x in years if x < y and (window == 0 or x >= y - window)]
        best, bs = None, -np.inf
        for i, r in enumerate(res):
            a = [v for x in tr_years for v in r[x]]
            if len(a) < min_trades:
                continue
            s = score(a, metric, rounded)
            if s > bs:
                best, bs = i, s
        picks[y] = best
    return picks


def summ(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "avg_r": round(float(a.mean()), 3) if len(a) else 0.0, "t": round(tstat(a), 2),
            "eq_pct_1pct": round(100 * (float(np.prod(1 + 0.01 * a)) - 1), 1)}


out = {}
for window in (0, 3):
    for metric in ("avg_r", "t"):
        for rounded in (True, False):
            p = wf(window, metric, rounded)
            allr = [v for y in TEST for v in res[p[y]][y]]
            ex = [v for y in TEST if y != 2017 for v in res[p[y]][y]]
            key = f"{'expanding' if window == 0 else 'w3'}_{metric}_{'rounded' if rounded else 'exact'}"
            out[key] = {"picks": p, "all": summ(allr), "ex2017": summ(ex),
                        "per_year": {y: summ(res[p[y]][y]) for y in TEST},
                        "long_only": all(cfgs[p[y]]["sides"] == "long" for y in TEST)}
            print(key, p, out[key]["all"], "ex2017", out[key]["ex2017"], flush=True)

# configs of the expanding/avg-R picks and their training record
p = out["expanding_avg_r_rounded"]["picks"]
out["expanding_avg_r_configs"] = {}
for y in TEST:
    i = p[y]
    tr = [v for x in years if x < y for v in res[i][x]]
    out["expanding_avg_r_configs"][y] = {"idx": i, "cfg": cfgs[i], "train_n": len(tr), "train_avg_r": round(float(np.mean(tr)), 3),
                                         "oos": res[i][y]}
    print(y, i, cfgs[i], "train", len(tr), round(float(np.mean(tr)), 3), "oos", summ(res[i][y]))
# share of total R from 2017
allr = [v for y in TEST for v in res[p[y]][y]]
r17 = sum(res[p[2017]][2017])
out["share_total_R_2017"] = round(r17 / sum(allr), 3)
print("2017 share of total R", out["share_total_R_2017"], "total", round(sum(allr), 2))
(OUT / "v2_repro.json").write_text(json.dumps(out, indent=1, default=str))
