"""Independent re-implementation of TradingView's built-in "RSI Divergence Indicator".

Does NOT import pinetest/divergence.py (or pinetest/indicators.py). It emulates Pine's bar-by-bar
execution literally:

    osc     = ta.rsi(close, 14)
    plFound = not na(ta.pivotlow(osc, lbL, lbR))
    phFound = not na(ta.pivothigh(osc, lbL, lbR))
    _inRange(cond) => bars = ta.barssince(cond == true); rangeLower <= bars and bars <= rangeUpper
    oscHL    = osc[lbR] > ta.valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
    priceLL  = low[lbR] < ta.valuewhen(plFound, low[lbR], 1)
    bull     = priceLL and oscHL and plFound                      (regular bullish)
    oscLL    = osc[lbR] < ta.valuewhen(plFound, osc[lbR], 1) and _inRange(plFound[1])
    priceHL  = low[lbR] > ta.valuewhen(plFound, low[lbR], 1)
    hbull    = priceHL and oscLL and plFound                      (hidden bullish)
    oscLH    = osc[lbR] < ta.valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
    priceHH  = high[lbR] > ta.valuewhen(phFound, high[lbR], 1)
    bear     = priceHH and oscLH and phFound                      (regular bearish)
    oscHH    = osc[lbR] > ta.valuewhen(phFound, osc[lbR], 1) and _inRange(phFound[1])
    priceLH  = high[lbR] < ta.valuewhen(phFound, high[lbR], 1)
    hbear    = priceLH and oscHH and phFound                      (hidden bearish)

Pivot tie conventions (`ties`):
  "left_strict"  : centre strictly beyond the LEFT bars, ties allowed with the RIGHT bars
                   (the loop in TradingView's v4 User Manual pivot example; on a flat top the
                   leftmost bar is the pivot).  Used as the reference.
  "strict"       : strictly beyond all 2*lb neighbours (what pinetest/divergence.py does).
  "right_strict" : ties allowed with the LEFT bars, strict vs the RIGHT bars.
"""
import math

import numpy as np


def pine_rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    """ta.rsi: rma(max(change,0)) / rma(-min(change,0)); rma seeded with the SMA of its first
    `length` non-na inputs (ta.change is na on bar 0)."""
    n = len(close)
    out = np.full(n, np.nan)
    alpha = 1.0 / length
    up = dn = math.nan
    ups, dns = [], []
    for t in range(1, n):
        ch = close[t] - close[t - 1]
        u, d = max(ch, 0.0), max(-ch, 0.0)
        if math.isnan(up):
            ups.append(u)
            dns.append(d)
            if len(ups) == length:
                up, dn = sum(ups) / length, sum(dns) / length
        else:
            up = alpha * u + (1 - alpha) * up
            dn = alpha * d + (1 - alpha) * dn
        if not math.isnan(up):
            out[t] = 100.0 if dn == 0 else (0.0 if up == 0 else 100.0 - 100.0 / (1.0 + up / dn))
    return out


def _pivot_at(x, t, lbL, lbR, low, ties):
    """Value of ta.pivotlow/pivothigh evaluated on bar t (centre t-lbR), or nan."""
    c = t - lbR
    if c - lbL < 0:
        return math.nan
    v = x[c]
    if math.isnan(v):
        return math.nan
    left = x[c - lbL:c]
    right = x[c + 1:t + 1]
    if np.isnan(left).any() or np.isnan(right).any():
        return math.nan
    s = 1.0 if low else -1.0       # compare s*v < s*neighbour
    L, R, V = s * left, s * right, s * v
    if ties == "strict":
        ok = (V < L).all() and (V < R).all()
    elif ties == "left_strict":
        ok = (V < L).all() and (V <= R).all()
    elif ties == "right_strict":
        ok = (V <= L).all() and (V < R).all()
    else:
        raise ValueError(ties)
    return v if ok else math.nan


def tv_divergences(o, h, l, c, lbL=5, lbR=5, range_lower=5, range_upper=60, rsi_len=14,
                   ties="left_strict"):
    """Bar-by-bar Pine emulation. Returns dict type -> list of (signal_bar, pivot_bar, prev_pivot_bar)."""
    n = len(c)
    osc = pine_rsi(c, rsi_len)
    out = {"regular_bull": [], "hidden_bull": [], "regular_bear": [], "hidden_bear": []}
    state = {}
    for is_low in (True, False):
        found = np.zeros(n, dtype=bool)
        vw_osc, vw_price, vw_bar = [], [], []     # history for ta.valuewhen
        last_cond_true = None                     # for ta.barssince(found[1])
        for t in range(n):
            found[t] = not math.isnan(_pivot_at(osc, t, lbL, lbR, is_low, ties))
            # ta.barssince(found[1]) is evaluated on every bar (Pine v5 eager evaluation)
            prev_found = found[t - 1] if t >= 1 else False
            if prev_found:
                last_cond_true = t
            bars = None if last_cond_true is None else t - last_cond_true
            in_range = bars is not None and range_lower <= bars <= range_upper
            if found[t]:
                vw_osc.append(osc[t - lbR])
                vw_price.append(l[t - lbR] if is_low else h[t - lbR])
                vw_bar.append(t - lbR)
            if not found[t] or len(vw_osc) < 2:
                continue
            o_now, o_prev = osc[t - lbR], vw_osc[-2]
            p_now, p_prev = vw_price[-1], vw_price[-2]
            rec = (t, t - lbR, vw_bar[-2])
            if is_low:
                if p_now < p_prev and o_now > o_prev and in_range:
                    out["regular_bull"].append(rec)
                if p_now > p_prev and o_now < o_prev and in_range:
                    out["hidden_bull"].append(rec)
            else:
                if p_now > p_prev and o_now < o_prev and in_range:
                    out["regular_bear"].append(rec)
                if p_now < p_prev and o_now > o_prev and in_range:
                    out["hidden_bear"].append(rec)
        state["pl" if is_low else "ph"] = found
    return out, osc, state


# ---- Independent trade construction and simulator (mirrors the study's stated rules) ----

def sma_atr(h, l, c, n=14):
    """Study's ATR: simple mean of true range over 14 bars (NaN for the first bars)."""
    N = len(c)
    tr = np.full(N, np.nan)
    tr[0] = h[0] - l[0]
    for t in range(1, N):
        tr[t] = max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1]))
    out = np.full(N, np.nan)
    cs = np.cumsum(tr)
    for t in range(n - 1, N):
        out[t] = (cs[t] - (cs[t - n] if t >= n else 0.0)) / n
    return out


def pine_ema(x, n):
    out = np.full(len(x), np.nan)
    a = 2 / (n + 1)
    out[n - 1] = x[:n].mean()
    for t in range(n, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def build_trades(sig, o, h, l, c, osc, atr, ema200, side, rr, filt, buffer_atr=0.1):
    """sig: list of (signal_bar, pivot_bar, prev_bar). Returns list of dicts (one per setup)."""
    out = []
    n = len(c)
    for i, p, _ in sig:
        if i + 1 >= n or np.isnan(atr[i]):
            continue
        if filt == "extreme" and not (osc[p] < 30 if side > 0 else osc[p] > 70):
            continue
        if filt == "trend" and (np.isnan(ema200[i]) or side * (c[i] - ema200[i]) <= 0):
            continue
        ext = l[p] if side > 0 else h[p]
        stop = ext - side * buffer_atr * atr[i]
        entry = o[i + 1]
        risk = side * (entry - stop)
        if risk <= 0:
            continue
        out.append(dict(bar=i, side=side, stop=stop, target=entry + side * rr * risk))
    return out


def simulate_ref(setups, o, h, l, c, cost_pct=0.1, max_hold=30, one_at_a_time=True):
    """Market entry at o[bar+1]; per bar: gap through stop/target fills at the open, else stop
    first if the bar's range touches it (conservative), else target; time exit at the close of
    the max_hold-th bar. Costs cost_pct% of notional on entry and on exit."""
    n = len(c)
    trades = []
    busy = -1
    for s in sorted(setups, key=lambda d: (d["bar"], -d["side"])):
        i, side, stop, tgt = s["bar"], s["side"], s["stop"], s["target"]
        if one_at_a_time and i <= busy:
            continue
        f = i + 1
        e = o[f]
        risk = side * (e - stop)
        if risk <= 0:
            continue
        last = min(f + max_hold - 1, n - 1)
        res = None
        for j in range(f, last + 1):
            if side * (o[j] - stop) <= 0:
                res = (j, o[j], "stop"); break
            if side * (o[j] - tgt) >= 0:
                res = (j, o[j], "target"); break
            lo_hit = (l[j] <= stop) if side > 0 else (h[j] >= stop)
            hi_hit = (h[j] >= tgt) if side > 0 else (l[j] <= tgt)
            if lo_hit:
                res = (j, stop, "stop"); break
            if hi_hit:
                res = (j, tgt, "target"); break
        if res is None:
            res = (last, c[last], "time")
        xb, x, why = res
        r = (side * (x - e) - cost_pct / 100 * (e + x)) / risk
        trades.append(dict(bar=i, side=side, fill_bar=f, entry=e, exit_bar=xb, exit=x, reason=why,
                           r=r, stop=stop, target=tgt, risk=risk))
        busy = xb
    return trades
