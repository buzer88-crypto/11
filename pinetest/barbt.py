"""Bar-by-bar signal backtester with Freqtrade-like exits, for porting published strategies.

Conventions:
- `enter[i]` / `exit_[i]` are known at the close of bar i; orders fill at the open of bar i+1.
- One position at a time, full stake; returns are fractions of the stake after costs.
- Exits, checked on every bar from the entry bar on (stop first when several trigger):
    stoploss   fixed stop at entry * (1 - stoploss) for longs (mirrored for shorts)
    trailing   stop at best price * (1 - trailing) once the trade is up `trailing_offset`
               (0 = always on); the trailing level only moves in the trade's favour
    roi        Freqtrade minimal_roi as [(bars_since_entry, profit), ...]: from that many bars
               after entry, exit once the price reaches entry * (1 + profit)
    exit_      signal exit at the next open; max_hold: exit at the close of that bar
  A bar that opens beyond the stop fills at the open; a ROI level an open has already passed
  fills at the open. Freqtrade's ROI of 0 means "exit at any profit", handled the same way.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: int
    entry: float
    exit: float
    reason: str
    ret: float        # return on the stake after costs


def backtest(df: pd.DataFrame, enter, exit_=None, side: int = 1, stoploss: float | None = None,
             roi: list | None = None, trailing: float | None = None, trailing_offset: float = 0.0,
             max_hold: int | None = None, cost_pct: float = 0.1) -> list[Trade]:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(c)
    enter = np.asarray(enter, dtype=bool)
    exit_ = np.zeros(n, dtype=bool) if exit_ is None else np.asarray(exit_, dtype=bool)
    roi = sorted(roi or [])
    cost = cost_pct / 100
    trades: list[Trade] = []
    i = 0
    while i < n - 1:
        if not enter[i]:
            i += 1
            continue
        e_bar, entry = i + 1, o[i + 1]
        best = entry
        stop = entry * (1 - side * stoploss) if stoploss else None
        done = None
        for j in range(e_bar, n):
            # 1) Price levels inside bar j: stop (fixed or trailing), then ROI.
            stops = [s for s in (stop, _trail(best, side, trailing, trailing_offset, entry)) if s is not None]
            eff = (max(stops) if side > 0 else min(stops)) if stops else None
            if eff is not None:
                if j > e_bar and side * (o[j] - eff) <= 0:
                    done = (j, o[j], "stop")
                elif side * (l[j] if side > 0 else h[j]) <= side * eff:
                    done = (j, eff, "stop")
            if done is None and roi:
                target = _roi_target(roi, j - e_bar, entry, side)
                if target is not None:
                    if j > e_bar and side * (o[j] - target) >= 0:
                        done = (j, o[j], "roi")
                    elif side * ((h[j] if side > 0 else l[j]) - target) >= 0:
                        done = (j, max(target, o[j]) if side > 0 else min(target, o[j]), "roi")
            if done is None and max_hold and j - e_bar + 1 >= max_hold:
                done = (j, c[j], "time")
            if done is None and exit_[j] and j + 1 < n:
                done = (j + 1, o[j + 1], "signal")
            if done is None and j == n - 1:
                done = (j, c[j], "end")
            if done:
                break
            best = max(best, h[j]) if side > 0 else min(best, l[j])
        x_bar, price, reason = done
        ret = side * (price - entry) / entry - cost * (1 + price / entry)
        trades.append(Trade(i, x_bar, side, entry, price, reason, ret))
        i = x_bar              # new entry signals count from the exit bar's close on
    return trades


def _trail(best, side, trailing, offset, entry):
    if not trailing:
        return None
    if side * (best - entry) / entry < offset:
        return None
    return best * (1 - side * trailing)


def _roi_target(roi, held, entry, side):
    """Freqtrade picks the entry with the largest key not above the time held."""
    active = [p for k, p in roi if k <= held]
    if not active:
        return None
    return entry * (1 + side * active[-1])


def summary(trades: list[Trade], df: pd.DataFrame, start=None, end=None) -> dict:
    idx = df.index
    lo = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    hi = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end) + pd.Timedelta(days=1))
    ts = [t for t in trades if lo <= t.entry_bar < hi]
    r = np.array([t.ret for t in ts])
    bars = sum(t.exit_bar - t.entry_bar for t in ts)
    if len(r) == 0:
        return {"n": 0, "total_pct": 0.0, "avg_pct": 0.0, "win_pct": 0.0, "exposure_pct": 0.0}
    return {"n": int(len(r)), "total_pct": round(100 * (float(np.prod(1 + r)) - 1), 1),
            "avg_pct": round(100 * float(r.mean()), 3), "win_pct": round(100 * float((r > 0).mean()), 1),
            "exposure_pct": round(100 * bars / max(hi - lo, 1), 1)}


def timing_placebo(df: pd.DataFrame, trades: list[Trade], rng: np.random.Generator, start=None, end=None,
                   draws: int = 500, cost_pct: float = 0.1) -> np.ndarray:
    """Average trade return of random-timed copies: same count, sides and holding lengths as the
    real trades in [start, end], entries at random bars of that window, exits at the open of
    the bar the real trade's length later. Returns the array of placebo average returns."""
    idx = df.index
    lo = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    hi = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end) + pd.Timedelta(days=1))
    ts = [t for t in trades if lo <= t.entry_bar < hi]
    if not ts:
        return np.zeros(draws)
    o = df["open"].to_numpy(dtype=float)
    lens = np.array([max(t.exit_bar - t.entry_bar, 1) for t in ts])
    sides = np.array([t.side for t in ts])
    cost = cost_pct / 100
    out = np.empty(draws)
    for k in range(draws):
        e = rng.integers(lo + 1, max(hi - lens.max() - 1, lo + 2), size=len(ts))
        x = np.minimum(e + lens, len(o) - 1)
        rel = o[x] / o[e]
        out[k] = float(np.mean(sides * (rel - 1) - cost * (1 + rel)))
    return out
