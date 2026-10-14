"""RSI divergence setups, following TradingView's built-in "RSI Divergence Indicator".

Pivots are found on the RSI (ta.pivotlow/ta.pivothigh with `left`/`right` bars), so a pivot at
bar p is only confirmed at bar p + right, and the signal is that confirmation bar. Price is
compared at the same pivot bars, against the previous RSI pivot 5..60 bars earlier.

- regular bullish: price lower low,  RSI higher low   -> long  (reversal)
- hidden bullish:  price higher low, RSI lower low    -> long  (continuation)
- regular bearish: price higher high, RSI lower high  -> short
- hidden bearish:  price lower high, RSI higher high  -> short
"""
import numpy as np
import pandas as pd

from .events import Setup
from .indicators import ema, rsi
from .smc import atr


def pivots(x: np.ndarray, left: int, right: int, low: bool) -> np.ndarray:
    """Bool array marking pivot bars p (strict extreme over [p-left, p+right], like ta.pivotlow).
    Usable at bar p + right at the earliest."""
    n = len(x)
    out = np.zeros(n, dtype=bool)
    for p in range(left, n - right):
        v = x[p]
        if np.isnan(v):
            continue
        window = np.r_[x[p - left:p], x[p + 1:p + right + 1]]
        if np.isnan(window).any():
            continue
        out[p] = (v < window).all() if low else (v > window).all()
    return out


def rsi_divergences(df: pd.DataFrame, kind: str = "regular", rsi_len: int = 14, left: int = 5,
                    right: int = 5, range_lower: int = 5, range_upper: int = 60, rr: float = 2.0,
                    extreme: bool = False, trend: bool = False, sides: tuple = (1, -1),
                    buffer_atr: float = 0.1, max_hold: int = 30) -> list[Setup]:
    """Setups at each divergence confirmation bar; market entry at the next open.

    Stop beyond the pivot's price extreme (plus `buffer_atr` ATR), target `rr` x risk.
    extreme: require the RSI pivot below 30 (bullish) / above 70 (bearish).
    trend: require close vs EMA200 on the signal bar to agree with the trade direction.
    """
    if kind not in ("regular", "hidden"):
        raise ValueError(kind)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    osc = rsi(df["close"], rsi_len).to_numpy()
    a = atr(df)
    ema200 = ema(df["close"], 200).to_numpy()
    out = []
    for side, is_low in ((1, True), (-1, False)):
        if side not in sides:
            continue
        piv = np.flatnonzero(pivots(osc, left, right, is_low))
        price = l if is_low else h
        for prev, cur in zip(piv[:-1], piv[1:]):
            i = cur + right                      # confirmation (signal) bar
            if i + 1 >= len(c) or not (range_lower <= cur - prev <= range_upper):
                continue
            osc_up, price_up = osc[cur] > osc[prev], price[cur] > price[prev]
            if is_low:
                ok = (not price_up and osc_up) if kind == "regular" else (price_up and not osc_up)
            else:
                ok = (price_up and not osc_up) if kind == "regular" else (not price_up and osc_up)
            if not ok or np.isnan(a[i]):
                continue
            if extreme and not (osc[cur] < 30 if is_low else osc[cur] > 70):
                continue
            if trend and (np.isnan(ema200[i]) or side * (c[i] - ema200[i]) <= 0):
                continue
            stop = price[cur] - side * buffer_atr * a[i]
            risk = side * (o[i + 1] - stop)
            if risk <= 0:
                continue                          # price already beyond the stop: no trade
            out.append(Setup(i, side, "market", stop, o[i + 1] + side * rr * risk, max_hold=max_hold))
    return sorted(out, key=lambda s: s.bar)
