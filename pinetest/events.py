"""Trade-by-trade simulator for setups with their own entry, stop and target prices.

Each setup is known at the close of its signal bar i. Entries:
- "market": fills at the open of bar i+1.
- "limit": rests from bar i+1 for `valid` bars; a long fills when the low reaches the price
  (at the open if the bar opens below it), a short when the high does. It is cancelled if the
  target trades first or it expires.
Exits, checked from the fill bar on: stop, target, or the close of the `max_hold`-th bar.
Conservative assumptions: if stop and target are both inside one bar, the stop wins; a limit
fill bar that also reaches the stop is a loss; an open beyond a level fills at the open.
Only one setup is live at a time: setups arriving while one is pending or open are skipped.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Setup:
    bar: int          # signal bar (known at its close)
    side: int         # +1 long, -1 short
    entry: str        # "market" or "limit"
    stop: float
    target: float
    price: float = np.nan   # limit price (ignored for market)
    valid: int = 24         # bars a limit order rests
    max_hold: int = 48      # bars a position is held at most


@dataclass
class Trade:
    bar: int
    side: int
    fill_bar: int
    entry: float
    exit_bar: int
    exit: float
    reason: str
    r: float          # result in units of initial risk, after costs


def simulate(df: pd.DataFrame, setups: list[Setup], cost_pct: float = 0.0) -> list[Trade]:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(c)
    trades: list[Trade] = []
    busy_until = -1
    for s in sorted(setups, key=lambda s: s.bar):
        if s.bar <= busy_until or s.bar + 1 >= n:
            continue
        fill_bar, entry = _fill(s, o, h, l, n)
        if fill_bar is None:
            busy_until = min(s.bar + s.valid, n - 1)
            continue
        risk = s.side * (entry - s.stop)
        if risk <= 0:      # filled at or beyond the stop (e.g. a gap): skip the setup
            busy_until = fill_bar
            continue
        exit_bar, exit_, reason = _exit(s, fill_bar, entry, o, h, l, c, n)
        cost = cost_pct / 100 * (entry + exit_)
        r = (s.side * (exit_ - entry) - cost) / risk
        trades.append(Trade(s.bar, s.side, fill_bar, entry, exit_bar, exit_, reason, r))
        busy_until = exit_bar
    return trades


def _fill(s, o, h, l, n):
    if s.entry == "market":
        return s.bar + 1, o[s.bar + 1]
    for j in range(s.bar + 1, min(s.bar + 1 + s.valid, n)):
        if s.side > 0:
            if h[j] >= s.target and l[j] > s.price:
                return None, None              # ran to target without us
            if l[j] <= s.price:
                return j, min(o[j], s.price)
        else:
            if l[j] <= s.target and h[j] < s.price:
                return None, None
            if h[j] >= s.price:
                return j, max(o[j], s.price)
    return None, None


def _exit(s, fill_bar, entry, o, h, l, c, n):
    last = min(fill_bar + s.max_hold - 1, n - 1)
    for j in range(fill_bar, last + 1):
        market_fill_bar = j == fill_bar and s.entry == "market"
        if j > fill_bar or market_fill_bar:
            # Gaps at the open (not possible on a limit fill bar: we filled inside it).
            if s.side * (o[j] - s.stop) <= 0:
                return j, o[j], "stop"
            if s.side * (o[j] - s.target) >= 0:
                return j, o[j], "target"
        hit_stop = (l[j] <= s.stop) if s.side > 0 else (h[j] >= s.stop)
        hit_target = (h[j] >= s.target) if s.side > 0 else (l[j] <= s.target)
        if hit_stop:
            return j, s.stop, "stop"
        if hit_target and not (j == fill_bar and s.entry == "limit"):
            return j, s.target, "target"
    return last, c[last], "time"


def summary(trades: list[Trade]) -> dict:
    r = np.array([t.r for t in trades])
    if len(r) == 0:
        return {"n": 0}
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    return {
        "n": len(r),
        "win_rate_pct": round(100 * float((r > 0).mean()), 1),
        "avg_r": round(float(r.mean()), 3),
        "total_r": round(float(r.sum()), 1),
        "t_stat": round(float(r.mean() / (std / np.sqrt(len(r)))), 2) if std > 0 else 0.0,
    }
