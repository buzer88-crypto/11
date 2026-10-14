"""Mechanical definitions of popular "Smart Money Concepts" setups, as pinetest.events.Setup lists.

SMC is usually drawn by eye; these rules fix one reasonable reading so it can be tested.
Every rule uses only bars up to and including the signal bar.
"""
import numpy as np
import pandas as pd

from .events import Setup


def atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 24, rr: float = 2.0, buffer_atr: float = 0.1,
                     max_hold: int = 48) -> list[Setup]:
    """Price trades through the lowest low of the previous `lookback` bars but closes back above
    it (stops taken, then rejected): long at the next open, stop just under the sweep's low.
    Mirrored for highs."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr(df)
    prior_low = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    prior_high = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    out = []
    for i in range(len(c) - 1):
        if np.isnan(a[i]) or np.isnan(prior_low[i]):
            continue
        if l[i] < prior_low[i] and c[i] > prior_low[i]:
            stop = l[i] - buffer_atr * a[i]
            risk = o[i + 1] - stop
            out.append(Setup(i, 1, "market", stop, o[i + 1] + rr * risk, max_hold=max_hold))
        elif h[i] > prior_high[i] and c[i] < prior_high[i]:
            stop = h[i] + buffer_atr * a[i]
            risk = stop - o[i + 1]
            out.append(Setup(i, -1, "market", stop, o[i + 1] - rr * risk, max_hold=max_hold))
    return out


def fair_value_gaps(df: pd.DataFrame, min_gap_atr: float = 0.3, rr: float = 2.0, buffer_atr: float = 0.1,
                    valid: int = 24, max_hold: int = 48) -> list[Setup]:
    """Bullish FVG at bar i: low[i] > high[i-2] (a gap the middle candle left untraded), at least
    `min_gap_atr` ATR wide. Limit buy at the gap's midpoint, stop under the gap. Mirrored."""
    h, l = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
    a = atr(df)
    out = []
    for i in range(2, len(h)):
        if np.isnan(a[i]):
            continue
        if l[i] - h[i - 2] >= min_gap_atr * a[i]:
            top, bottom = l[i], h[i - 2]
            price, stop = (top + bottom) / 2, bottom - buffer_atr * a[i]
            out.append(Setup(i, 1, "limit", stop, price + rr * (price - stop), price, valid, max_hold))
        elif l[i - 2] - h[i] >= min_gap_atr * a[i]:
            top, bottom = l[i - 2], h[i]
            price, stop = (top + bottom) / 2, top + buffer_atr * a[i]
            out.append(Setup(i, -1, "limit", stop, price - rr * (stop - price), price, valid, max_hold))
    return out


def order_blocks(df: pd.DataFrame, lookback: int = 24, search: int = 10, rr: float = 2.0,
                 buffer_atr: float = 0.1, valid: int = 24, max_hold: int = 48) -> list[Setup]:
    """Break of structure up at bar i: close above the highest high of the previous `lookback`
    bars. The order block is the last down candle (close < open) within `search` bars before i.
    Limit buy at the block's high, stop under its low. Mirrored for breaks down."""
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = atr(df)
    prior_high = pd.Series(h).rolling(lookback).max().shift(1).to_numpy()
    prior_low = pd.Series(l).rolling(lookback).min().shift(1).to_numpy()
    out = []
    for i in range(len(c)):
        if np.isnan(a[i]) or np.isnan(prior_high[i]):
            continue
        if c[i] > prior_high[i]:
            for j in range(i - 1, max(i - 1 - search, -1), -1):
                if c[j] < o[j]:
                    price, stop = h[j], l[j] - buffer_atr * a[i]
                    if price < c[i]:
                        out.append(Setup(i, 1, "limit", stop, price + rr * (price - stop), price, valid, max_hold))
                    break
        elif c[i] < prior_low[i]:
            for j in range(i - 1, max(i - 1 - search, -1), -1):
                if c[j] > o[j]:
                    price, stop = l[j], h[j] + buffer_atr * a[i]
                    if price > c[i]:
                        out.append(Setup(i, -1, "limit", stop, price - rr * (stop - price), price, valid, max_hold))
                    break
    return out
