"""RSI divergence setups, following TradingView's built-in "RSI Divergence Indicator".

Pivots are found on the RSI (ta.pivotlow/ta.pivothigh with `left`/`right` bars), so a pivot at
bar p is only confirmed at bar p + right, and the signal is that confirmation bar. Price is
compared at the same pivot bars, against the previous RSI pivot. The range check follows
Pine's `ta.barssince(plFound[1])`, which is the pivot distance minus 1, so the default 5..60
accepts pivots 6..61 bars apart. Comparisons are strict, as in the indicator.

- regular bullish: price lower low,  RSI higher low   -> long  (reversal)
- hidden bullish:  price higher low, RSI lower low    -> long  (continuation)
- regular bearish: price higher high, RSI lower high  -> short
- hidden bearish:  price lower high, RSI higher high  -> short
"""
import numpy as np
import pandas as pd

from .events import Setup
from .indicators import ema, macd, rsi
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
                    buffer_atr: float = 0.1, max_hold: int = 30, confirm: str = "none",
                    wait: int = 6) -> list[Setup]:
    """Setups at each divergence confirmation bar; market entry at the next open.

    Stop beyond the pivot's price extreme (plus `buffer_atr` ATR), target `rr` x risk.
    extreme: require the RSI pivot below 30 (bullish) / above 70 (bearish).
    trend: require close vs EMA200 on the signal bar to agree with the trade direction.
    confirm (all known at the bar's close; "long" wording, mirrored for shorts):
      macd_cross     MACD(12,26,9) line above its signal line on the signal bar
      macd_hist      MACD histogram rising on the signal bar
      macd_div       the MACD line diverges too at the same two pivot bars
      ema50          close above EMA50
      ema_9_21       EMA9 above EMA21
      ema20_reclaim  wait up to `wait` bars for a close above EMA20 and enter on the next
                     open; cancelled if the stop trades first
    """
    if confirm not in CONFIRM:
        raise ValueError(confirm)
    if kind not in ("regular", "hidden"):
        raise ValueError(kind)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    osc = rsi(df["close"], rsi_len).to_numpy()
    a = atr(df)
    ema200 = ema(df["close"], 200).to_numpy()
    line, sig, hist = (x.to_numpy() for x in macd(df["close"]))
    e9, e20, e21, e50 = (ema(df["close"], n).to_numpy() for n in (9, 20, 21, 50))
    out = []
    for side, is_low in ((1, True), (-1, False)):
        if side not in sides:
            continue
        piv = np.flatnonzero(pivots(osc, left, right, is_low))
        price = l if is_low else h
        for prev, cur in zip(piv[:-1], piv[1:]):
            i = cur + right                      # confirmation (signal) bar
            if i + 1 >= len(c) or not (range_lower <= cur - prev - 1 <= range_upper):
                continue
            osc_up, osc_down = osc[cur] > osc[prev], osc[cur] < osc[prev]
            price_up, price_down = price[cur] > price[prev], price[cur] < price[prev]
            if is_low:
                ok = (price_down and osc_up) if kind == "regular" else (price_up and osc_down)
            else:
                ok = (price_up and osc_down) if kind == "regular" else (price_down and osc_up)
            if not ok or np.isnan(a[i]):
                continue
            if extreme and not (osc[cur] < 30 if is_low else osc[cur] > 70):
                continue
            if trend and (np.isnan(ema200[i]) or side * (c[i] - ema200[i]) <= 0):
                continue
            stop = price[cur] - side * buffer_atr * a[i]
            if confirm == "macd_cross" and not side * (line[i] - sig[i]) > 0:
                continue
            if confirm == "macd_hist" and not side * (hist[i] - hist[i - 1]) > 0:
                continue
            # Double divergence: the MACD line moves between the two pivots the same way RSI does.
            if confirm == "macd_div" and not (line[cur] > line[prev]) == osc_up:
                continue
            if confirm == "macd_div" and (np.isnan(line[cur]) or np.isnan(line[prev])):
                continue
            if confirm == "ema50" and not side * (c[i] - e50[i]) > 0:
                continue
            if confirm == "ema_9_21" and not side * (e9[i] - e21[i]) > 0:
                continue
            if confirm == "ema20_reclaim":
                i = _reclaim(i, side, stop, c, h, l, e20, wait)
                if i is None or i + 1 >= len(c):
                    continue
            risk = side * (o[i + 1] - stop)
            if risk <= 0:
                continue                          # price already beyond the stop: no trade
            out.append(Setup(i, side, "market", stop, o[i + 1] + side * rr * risk, max_hold=max_hold))
    return sorted(out, key=lambda s: s.bar)


CONFIRM = ("none", "macd_cross", "macd_hist", "macd_div", "ema50", "ema_9_21", "ema20_reclaim")


def _reclaim(i, side, stop, c, h, l, e20, wait):
    """First bar j in (i, i+wait] whose close is on the trade's side of EMA20 (the signal bar
    itself counts if it already is). None if the stop trades first or nothing happens."""
    if side * (c[i] - e20[i]) > 0:
        return i
    for j in range(i + 1, min(i + 1 + wait, len(c))):
        if (l[j] <= stop) if side > 0 else (h[j] >= stop):
            return None
        if side * (c[j] - e20[j]) > 0:
            return j
    return None
