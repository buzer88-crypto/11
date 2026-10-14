"""Shared helpers: vectorised walk-forward over per-config/per-year (count, sum, sum of squares),
a vectorised long/short trade simulator matching pinetest.events.simulate for market entries,
and small stats helpers."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
YEARS = np.arange(2014, 2027)
TEST_YEARS = np.arange(2017, 2027)
COST = 0.1 / 100
DOMINANT = {"kind": "hidden", "rsi_len": 21, "lb": 8, "rr": 4.0, "buffer_atr": 0.0, "max_hold": 60,
            "confirm": "ema_9_21", "trend": False, "sides": "long", "extreme": False}


def load_cache():
    return pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())


def load_df(tf="4h"):
    df = pd.read_csv(ROOT / "data" / f"BTCUSD_{tf}.csv", parse_dates=["date"]).set_index("date")
    return df.loc["2013-01-01":]


def arrays_from_results(results, years=YEARS):
    C, Y = len(results), len(years)
    N, S, Q = np.zeros((C, Y)), np.zeros((C, Y)), np.zeros((C, Y))
    for c, by_year in enumerate(results):
        for j, y in enumerate(years):
            r = np.asarray(by_year[int(y)], dtype=float)
            N[c, j], S[c, j], Q[c, j] = len(r), r.sum(), (r * r).sum()
    return N, S, Q


def _tstat(n, s, q):
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / n
        var = (q - n * m * m) / (n - 1)
        t = m / np.sqrt(var / n)
    t = np.where((n > 1) & (var > 1e-18), t, 0.0)
    return np.nan_to_num(t)


def walk_forward(N, S, Q, window, metric, min_trades=20, test_idx=None, round_scores=True):
    """window: int years or 0 = expanding (all earlier years in the arrays). metric: avg_r | t | total_r.
    Returns picks (config index per test year, -1 if none) and OOS (n, sum, sumsq)."""
    Y = N.shape[1]
    if test_idx is None:
        test_idx = np.arange(Y - len(TEST_YEARS), Y)
    cN, cS, cQ = np.cumsum(N, 1), np.cumsum(S, 1), np.cumsum(Q, 1)

    def wsum(cA, a, b):           # sum over columns [a, b)
        return cA[:, b - 1] - (cA[:, a - 1] if a > 0 else 0)

    picks, on, os_, oq = [], 0.0, 0.0, 0.0
    for yi in test_idx:
        a = max(0, yi - window) if window else 0
        n, s, q = wsum(cN, a, yi), wsum(cS, a, yi), wsum(cQ, a, yi)
        with np.errstate(divide="ignore", invalid="ignore"):
            if metric == "avg_r":
                sc = s / n
                sc = np.round(sc, 3) if round_scores else sc
            elif metric == "t":
                sc = _tstat(n, s, q)
                sc = np.round(sc, 2) if round_scores else sc
            elif metric == "total_r":
                sc = s.copy()
            else:
                raise ValueError(metric)
        sc = np.where(n >= min_trades, sc, -np.inf)
        k = int(np.argmax(sc)) if np.isfinite(sc).any() else -1
        picks.append(k)
        if k >= 0:
            on += N[k, yi]
            os_ += S[k, yi]
            oq += Q[k, yi]
    return np.array(picks), (on, os_, oq)


def oos_stats(nsq):
    n, s, q = nsq
    return {"n": int(n), "avg_r": float(s / n) if n else 0.0, "t": float(_tstat(n, s, q)) if n else 0.0}


REPORTED = [(3, "avg_r"), (3, "t"), (0, "avg_r"), (0, "t")]


def sim_market(o, h, l, c, fill, side, risk_pct, rr, hold, cost=COST, maxw=60):
    """Vectorised version of pinetest.events.simulate for market entries (no one-at-a-time logic).
    fill: fill bar (signal bar + 1). Returns exit_bar, R."""
    n = len(c)
    fill = np.asarray(fill, int)
    side = np.asarray(side, float)
    e = o[fill]
    risk = np.asarray(risk_pct, float) * e
    stop = e - side * risk
    target = e + side * np.asarray(rr, float) * risk
    hold = np.asarray(hold, int)
    k = np.arange(maxw)
    idx = fill[:, None] + k[None, :]
    valid = (k[None, :] < hold[:, None]) & (idx <= n - 1)
    idx = np.minimum(idx, n - 1)
    lo, hi, op = l[idx], h[idx], o[idx]
    longs = side[:, None] > 0
    hs = np.where(longs, lo <= stop[:, None], hi >= stop[:, None])
    ht = np.where(longs, hi >= target[:, None], lo <= target[:, None])
    ev = (hs | ht) & valid
    anyev = ev.any(1)
    first = np.argmax(ev, 1)
    rows = np.arange(len(fill))
    oj = op[rows, first]
    gap_stop = side * (oj - stop) <= 0
    gap_tgt = side * (oj - target) >= 0
    px = np.where(gap_stop | gap_tgt, oj, np.where(hs[rows, first], stop, target))
    last = np.minimum(fill + hold - 1, n - 1)
    exit_bar = np.where(anyev, fill + first, last)
    exit_px = np.where(anyev, px, c[last])
    r = (side * (exit_px - e) - cost * (e + exit_px)) / risk
    return exit_bar, r


def q(x, p):
    return float(np.quantile(np.asarray(x, float), p))
