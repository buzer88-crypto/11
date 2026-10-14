"""Trader-practice review: RSI divergence detectors, entries and exits as commonly taught.

See PREREG.txt (written before any run). Run from /home/user/11 with PYTHONPATH=. .
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.divergence import pivots
from pinetest.indicators import ema, rsi
from pinetest.smc import atr

ROOT = Path(__file__).resolve().parents[3]
PERIODS = {"pre": ("2014-01-01", "2017-12-31"), "train": ("2018-01-01", "2021-12-31"),
           "test": ("2022-01-01", "2100-01-01")}
BUFFER_ATR = 0.1
RANGE = (5, 60)
CONFIRM_WAIT = 10


def load(tf="4h"):
    df = pd.read_csv(ROOT / "data" / f"BTCUSD_{tf}.csv", parse_dates=["date"]).set_index("date")
    return df.loc["2013-01-01":]


class Data:
    def __init__(self, df):
        self.df = df
        self.o, self.h, self.l, self.c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.rsi = rsi(df["close"], 14).to_numpy()
        self.atr = atr(df)
        self.ema200 = ema(df["close"], 200).to_numpy()
        self.n = len(df)
        self.idx = df.index
        self.bounds = {}
        for name, (a, b) in PERIODS.items():
            self.bounds[name] = (int(self.idx.searchsorted(pd.Timestamp(a))),
                                 int(self.idx.searchsorted(pd.Timestamp(b), side="right")))
        self._cache = {}

    def period_of(self, bar):
        for name, (lo, hi) in self.bounds.items():
            if lo <= bar < hi:
                return name
        return None


@dataclass
class Event:
    i: int          # signal (confirmation) bar
    side: int
    cur: int        # second pivot bar
    prev: int       # first pivot bar
    rsi_prev: float
    rsi_cur: float
    stop_piv: float
    stop_swing: float


def divergences(D: Data, base: str, lb: int, side: int, kind: str) -> list[Event]:
    """base 'rsi': pivots on RSI, price compared at those bars (TV indicator).
    base 'price': pivots on price low/high, RSI compared at those bars."""
    key = (base, lb, side, kind)
    if key in D._cache:
        return D._cache[key]
    is_low = side > 0
    price = D.l if is_low else D.h
    src = D.rsi if base == "rsi" else price
    piv = np.flatnonzero(pivots(src, lb, lb, is_low))
    out = []
    for prev, cur in zip(piv[:-1], piv[1:]):
        i = cur + lb
        if i + 1 >= D.n or not (RANGE[0] <= cur - prev <= RANGE[1]):
            continue
        r0, r1 = D.rsi[prev], D.rsi[cur]
        if np.isnan(r0) or np.isnan(r1) or np.isnan(D.atr[i]):
            continue
        osc_up, price_up = r1 > r0, price[cur] > price[prev]
        if is_low:
            ok = (not price_up and osc_up) if kind == "regular" else (price_up and not osc_up)
        else:
            ok = (price_up and not osc_up) if kind == "regular" else (not price_up and osc_up)
        if not ok:
            continue
        buf = side * BUFFER_ATR * D.atr[i]
        stop_piv = price[cur] - buf
        swing = D.l[cur - lb:i + 1].min() if is_low else D.h[cur - lb:i + 1].max()
        out.append(Event(i, side, cur, prev, r0, r1, stop_piv, swing - buf))
    D._cache[key] = out
    return out


SETUPS = ["regbull", "regbull_os", "regular", "regular_ext", "hidden_trend"]


def candidates(D: Data, detector: str, lb: int, setup: str) -> list[tuple]:
    """(i, side, stop, cur) candidates, sorted by signal bar."""
    base = "rsi" if detector.startswith("rsi") else "price"
    use_swing = detector == "rsi_swingstop"
    kind = "hidden" if setup == "hidden_trend" else "regular"
    sides = (1,) if setup.startswith("regbull") else (1, -1)
    out = []
    for side in sides:
        for e in divergences(D, base, lb, side, kind):
            if setup in ("regbull_os", "regular_ext"):
                if not (e.rsi_prev < 30 if side > 0 else e.rsi_prev > 70):
                    continue
            if setup == "hidden_trend":
                if np.isnan(D.ema200[e.i]) or side * (D.c[e.i] - D.ema200[e.i]) <= 0:
                    continue
            out.append((e.i, side, e.stop_swing if use_swing else e.stop_piv, e.cur))
    out.sort(key=lambda t: (t[0], -t[1]))
    return out


def opposite_bars(D: Data, detector: str, lb: int) -> dict:
    """Bool arrays of confirmation bars of regular divergences, keyed by the side they oppose."""
    base = "rsi" if detector.startswith("rsi") else "price"
    key = ("opp", base, lb)
    if key not in D._cache:
        bull = np.zeros(D.n, bool)
        bear = np.zeros(D.n, bool)
        for e in divergences(D, base, lb, 1, "regular"):
            bull[e.i] = True
        for e in divergences(D, base, lb, -1, "regular"):
            bear[e.i] = True
        D._cache[key] = {1: bear, -1: bull}     # a long exits on a bearish divergence
    return D._cache[key]


EXITS = ["rr1", "rr2", "rr3", "rsi50", "rsi7030", "oppdiv", "chand2", "chand3", "time12", "time30", "time60"]


def exit_spec(name):
    if name.startswith("rr"):
        return {"rr": float(name[2:]), "cap": 30}
    if name.startswith("time"):
        return {"cap": int(name[4:])}
    if name == "rsi50":
        return {"rsi": (50.0, 50.0), "cap": 60}
    if name == "rsi7030":
        return {"rsi": (70.0, 30.0), "cap": 60}
    if name == "oppdiv":
        return {"opp": True, "cap": 60}
    if name.startswith("chand"):
        return {"chand": float(name[5:]), "cap": 60}
    raise ValueError(name)


@dataclass
class Trade:
    bar: int        # signal bar (period assignment)
    side: int
    fill: int
    entry: float
    exit_bar: int
    exit: float
    reason: str
    risk: float     # price distance entry -> initial stop

    def r(self, cost_pct):
        return (self.side * (self.exit - self.entry) - cost_pct / 100 * (self.entry + self.exit)) / self.risk


def run_exit(D: Data, side, fill, entry, stop, spec, opp=None):
    o, h, l, c, n = D.o, D.h, D.l, D.c, D.n
    risk = side * (entry - stop)
    tgt = entry + side * spec["rr"] * risk if "rr" in spec else None
    last = min(fill + spec["cap"] - 1, n - 1)
    st = stop
    ext = h[fill] if side > 0 else l[fill]
    flag = False
    lvl = None
    if "rsi" in spec:
        lvl = spec["rsi"][0] if side > 0 else spec["rsi"][1]
    for j in range(fill, last + 1):
        if flag:
            return j, o[j], "signal"
        if side * (o[j] - st) <= 0:
            return j, o[j], "stop" if st == stop else "trail"
        if tgt is not None and side * (o[j] - tgt) >= 0:
            return j, o[j], "target"
        if (l[j] <= st) if side > 0 else (h[j] >= st):
            return j, st, "stop" if st == stop else "trail"
        if tgt is not None and ((h[j] >= tgt) if side > 0 else (l[j] <= tgt)):
            return j, tgt, "target"
        if lvl is not None and not np.isnan(D.rsi[j]) and side * (D.rsi[j] - lvl) >= 0:
            flag = True
        if opp is not None and opp[j]:
            flag = True
        if "chand" in spec:
            k = spec["chand"]
            if side > 0:
                ext = max(ext, h[j])
                st = max(st, ext - k * D.atr[j])
            else:
                ext = min(ext, l[j])
                st = min(st, ext + k * D.atr[j])
    return last, c[last], "time"


def simulate(D: Data, cands, entry_mode, exit_name, detector, lb):
    spec = exit_spec(exit_name)
    opp_all = opposite_bars(D, detector, lb) if spec.get("opp") else None
    o, h, l, c, n = D.o, D.h, D.l, D.c, D.n
    trades = []
    busy = -1
    for i, side, stop, cur in cands:
        if i <= busy or i + 1 >= n:
            continue
        if entry_mode == "next_open":
            fill = i + 1
        else:
            trig = h[cur] if side > 0 else l[cur]
            fill = None
            end = min(i + CONFIRM_WAIT, n - 2)
            j = i
            while j <= end:
                if j > i and ((l[j] <= stop) if side > 0 else (h[j] >= stop)):
                    break                       # stopped out before confirmation
                if side * (c[j] - trig) > 0:
                    fill = j + 1
                    break
                j += 1
            if fill is None:
                busy = min(j, end)
                continue
        entry = o[fill]
        risk = side * (entry - stop)
        if risk <= 0:
            if entry_mode != "next_open":
                busy = fill
            continue
        xb, xp, why = run_exit(D, side, fill, entry, stop, spec, opp_all[side] if opp_all else None)
        trades.append(Trade(i, side, fill, entry, xb, xp, why, risk))
        busy = xb
    return trades


def stats(rs):
    r = np.asarray(rs, float)
    if len(r) == 0:
        return {"n": 0, "avg_r": 0.0, "sharpe": 0.0, "t": 0.0, "win": 0.0, "total_r": 0.0}
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"n": int(len(r)), "avg_r": round(float(r.mean()), 4),
            "sharpe": round(float(r.mean() / sd), 4) if sd > 0 else 0.0,
            "t": round(float(r.mean() / sd * np.sqrt(len(r))), 2) if sd > 0 else 0.0,
            "win": round(100 * float((r > 0).mean()), 1), "total_r": round(float(r.sum()), 1)}


def by_period(D: Data, trades):
    out = {p: [] for p in PERIODS}
    for t in trades:
        p = D.period_of(t.bar)
        if p:
            out[p].append(t)
    return out


def placebo(D: Data, real, exit_name, detector, lb, rng):
    """Random bars in [first, last] real signal bar, same side and risk % as each real trade,
    same exit rule, market entry at the next open, one at a time."""
    if not real:
        return []
    spec = exit_spec(exit_name)
    opp_all = opposite_bars(D, detector, lb) if spec.get("opp") else None
    lo, hi = min(t.bar for t in real), max(t.bar for t in real)
    hi = min(hi, D.n - 2)
    bars = rng.integers(lo, hi + 1, size=len(real))
    plan = sorted(((int(b), t.side, t.risk / t.entry) for b, t in zip(bars, real)), key=lambda x: x[0])
    out = []
    busy = -1
    for b, side, rp in plan:
        if b <= busy:
            continue
        fill = b + 1
        entry = D.o[fill]
        stop = entry * (1 - side * rp)
        xb, xp, why = run_exit(D, side, fill, entry, stop, spec, opp_all[side] if opp_all else None)
        out.append(Trade(b, side, fill, entry, xb, xp, why, side * (entry - stop)))
        busy = xb
    return out
