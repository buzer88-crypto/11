"""Shared helpers for the reproduction / timing-placebo review."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
from run import COST, load  # noqa: E402,F401

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

TEST = list(range(2017, 2027))
YEARS = list(range(2014, 2027))


def cache():
    return pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())


def trades_for(df, cfg):
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"],
                             rr=cfg["rr"], buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"],
                             confirm=cfg["confirm"], trend=cfg["trend"], extreme=cfg["extreme"],
                             sides=(1, -1) if cfg["sides"] == "both" else (1,))
    return simulate(df, setups, COST)


def by_year(df, trades):
    yrs = df.index.year.to_numpy()
    out = {y: [] for y in YEARS}
    for t in trades:
        y = int(yrs[t.bar])
        if y in out:
            out[y].append(t)
    return out


def sim_long_independent(o, h, l, c, sig_bars, risk_pct, rr, max_hold, cost=COST):
    """Vectorised copy of events._exit for independent long market trades (no one-at-a-time).
    sig_bars: (..., k) signal bars; entry at o[bar+1]; stop entry*(1-risk_pct); target entry+rr*risk.
    Returns (R, exit_bar, fill_bar)."""
    n = len(c)
    sig_bars = np.asarray(sig_bars)
    fill = sig_bars + 1
    entry = o[fill]
    risk = entry * risk_pct
    stop = entry - risk
    target = entry + rr * risk
    shape = fill.shape
    r = np.full(shape, np.nan)
    exit_bar = np.full(shape, -1)
    done = np.zeros(shape, bool)
    exit_px = np.full(shape, np.nan)
    for k in range(max_hold):
        j = np.minimum(fill + k, n - 1)
        active = ~done & (fill + k <= n - 1)
        oj, hj, lj = o[j], h[j], l[j]
        # gap at the open (on the fill bar o==entry so neither triggers)
        g_stop = active & (oj - stop <= 0)
        exit_px = np.where(g_stop, oj, exit_px); exit_bar = np.where(g_stop, j, exit_bar); done |= g_stop
        active &= ~g_stop
        g_tgt = active & (oj - target >= 0)
        exit_px = np.where(g_tgt, oj, exit_px); exit_bar = np.where(g_tgt, j, exit_bar); done |= g_tgt
        active &= ~g_tgt
        hs = active & (lj <= stop)
        exit_px = np.where(hs, stop, exit_px); exit_bar = np.where(hs, j, exit_bar); done |= hs
        active &= ~hs
        ht = active & (hj >= target)
        exit_px = np.where(ht, target, exit_px); exit_bar = np.where(ht, j, exit_bar); done |= ht
    last = np.minimum(fill + max_hold - 1, n - 1)
    exit_px = np.where(done, exit_px, c[last]); exit_bar = np.where(done, exit_bar, last)
    costv = cost / 100 * (entry + exit_px)
    r = (exit_px - entry - costv) / risk
    return r, exit_bar, fill
