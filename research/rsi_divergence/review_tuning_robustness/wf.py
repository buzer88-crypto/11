"""Shared walk-forward helpers for the robustness review (reads data/tune_cache.pkl only).

Mirrors tune.walk_forward exactly for metrics avg_r and t (scores rounded like tune.stats,
first index wins ties), and adds total_r and median_r.
"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "tune_cache.pkl"
YEARS = list(range(2014, 2027))
TEST_YEARS = list(range(2017, 2027))


def load_cache():
    d = pickle.loads(CACHE.read_bytes())
    return d["configs"], d["results"]


def stats(rs):
    r = np.asarray(rs, dtype=float)
    if len(r) == 0:
        return {"n": 0, "avg_r": 0.0, "total_r": 0.0, "t": 0.0}
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"n": int(len(r)), "avg_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2),
            "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0}


def score(rs, metric):
    r = np.asarray(rs, dtype=float)
    if metric == "avg_r":
        return round(float(r.mean()), 3)
    if metric == "t":
        sd = r.std(ddof=1) if len(r) > 1 else 0.0
        return round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0
    if metric == "total_r":
        return round(float(r.sum()), 2)
    if metric == "median_r":
        return round(float(np.median(r)), 3)
    raise ValueError(metric)


def train_years(y, window, start=2014):
    if window:
        return [x for x in YEARS if y - window <= x < y and x >= start]
    return [x for x in YEARS if start <= x < y]


def walk_forward(results, window, metric, min_trades=20, idx=None, start=2014, test_years=TEST_YEARS):
    """Returns picks {year: config index or None} and the OOS R list per year."""
    idx = range(len(results)) if idx is None else idx
    picks, oos = {}, {}
    for y in test_years:
        ty = train_years(y, window, start)
        best, best_score = None, -np.inf
        for i in idx:
            by_year = results[i]
            rs = [r for x in ty for r in by_year[x]]
            if len(rs) < min_trades:
                continue
            s = score(rs, metric)
            if s > best_score:
                best, best_score = i, s
        picks[y] = best
        oos[y] = list(results[best][y]) if best is not None else []
    return picks, oos


def summarize(picks, oos, configs=None):
    all_r = [r for y in oos for r in oos[y]]
    ex = [r for y in oos if y != 2017 for r in oos[y]]
    s, se = stats(all_r), stats(ex)
    out = {"n": s["n"], "avg_r": s["avg_r"], "t": s["t"], "total_r": s["total_r"],
           "n_ex2017": se["n"], "avg_r_ex2017": se["avg_r"], "t_ex2017": se["t"],
           "n_distinct": len({p for p in picks.values() if p is not None}),
           "per_year_avg_r": {y: (round(float(np.mean(oos[y])), 3) if oos[y] else None) for y in oos},
           "per_year_n": {y: len(oos[y]) for y in oos}}
    if configs is not None:
        out["long_only_share"] = round(float(np.mean([configs[p]["sides"] == "long" for p in picks.values() if p is not None])), 2)
        out["picks"] = {y: p for y, p in picks.items()}
    return out
