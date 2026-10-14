"""Intrabar truth for the RSI-divergence study: re-simulate every trade minute by minute.

For each config, the study's 4h simulator (pinetest.events.simulate) gives the trades. Each
trade is then replayed on Bitstamp 1-minute bars from the entry (open of 4h fill bar = open of
its first minute) to the end of the 30th 4h bar:
- stop / target: the first minute whose low/high reaches the level; a minute that opens
  beyond the level fills at that minute's open (stop worse, target better);
- if stop and target are both inside one minute (and the minute did not open beyond either),
  the outcome is still ambiguous: counted, and resolved as stop (main) or target (alt);
- otherwise the time exit at the close of the 30th 4h bar (= close of its last minute).
The trade set is also re-derived with the one-trade-at-a-time rule using the minute exits.
Entry slippage: 4h open vs prices in the first minutes after the open.
Run from /home/user/11 with PYTHONPATH=. ; writes results here.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_minutes import load as load_minutes  # noqa: E402
from pinetest.events import simulate, summary  # noqa: E402
from research.rsi_divergence.run import COST, GRID, PERIODS, by_period, load, setups_for  # noqa: E402

OUT = Path(__file__).parent
TOP = [("hidden", 3, 3.0, "trend"), ("hidden", 5, 3.0, "trend"), ("hidden", 5, 2.0, "trend"),
       ("hidden", 5, 3.0, "none")]


class Minutes:
    def __init__(self, df4):
        m = load_minutes()
        self.ts, self.o, self.h, self.l, self.c, self.v = (m[k] for k in ("ts", "open", "high", "low", "close", "volume"))
        self.t0 = int(self.ts[0])
        assert (np.diff(self.ts) == 60).all(), "minute series must be complete"
        bar_ts = (df4.index.astype("int64") // 10**9).to_numpy() if df4.index.dtype.kind == "M" else None
        # pandas 3 may store datetime64[us]: convert robustly
        bar_ts = np.array([int(t.timestamp()) for t in df4.index], dtype=np.int64)
        self.bar_start = (bar_ts - self.t0) // 60          # minute index of each 4h bar's first minute
        assert (self.bar_start >= 0).all()
        self.bar_end = np.r_[self.bar_start[1:], len(self.ts)]   # exclusive

    def check_bars(self, df4):
        """4h OHLC must equal the minute aggregates."""
        s, e = self.bar_start, self.bar_end
        o = self.o[s]
        c = self.c[e - 1]
        h = np.maximum.reduceat(self.h, s)
        l = np.minimum.reduceat(self.l, s)
        d = {k: float(np.nanmax(np.abs(a - df4[k].to_numpy()))) for k, a in
             (("open", o), ("high", h), ("low", l), ("close", c))}
        return d


def minute_exit(M, side, entry, stop, target, fill_bar, last_bar):
    a, b = M.bar_start[fill_bar], M.bar_end[last_bar]
    O, H, L = M.o[a:b], M.h[a:b], M.l[a:b]
    if side > 0:
        sh, th = L <= stop, H >= target
    else:
        sh, th = H >= stop, L <= target
    js = int(np.argmax(sh)) if sh.any() else 10**12
    jt = int(np.argmax(th)) if th.any() else 10**12
    amb = False
    if js == jt == 10**12:
        return b - 1, float(M.c[b - 1]), "time", amb
    if js < jt:
        j, reason = js, "stop"
    elif jt < js:
        j, reason = jt, "target"
    else:                                   # both inside the same minute
        j = js
        if side * (O[j] - stop) <= 0:
            reason = "stop"
        elif side * (O[j] - target) >= 0:
            reason = "target"
        else:
            reason, amb = "stop", True
    if reason == "stop":
        px = O[j] if side * (O[j] - stop) <= 0 else stop
    else:
        px = O[j] if side * (O[j] - target) >= 0 else target
    return a + j, float(px), reason, amb


def r_of(side, entry, exit_, risk, cost=COST):
    return (side * (exit_ - entry) - cost / 100 * (entry + exit_)) / risk


def replay(M, df4, setups, trades):
    """Replay the 4h simulator's trades on minutes (same trade set)."""
    n = len(df4)
    rows = []
    stop_of = {(s.bar, s.side): s for s in setups}
    for t in trades:
        s = stop_of[(t.bar, t.side)]
        last = min(t.fill_bar + s.max_hold - 1, n - 1)
        risk = t.side * (t.entry - s.stop)
        mi, px, reason, amb = minute_exit(M, t.side, t.entry, s.stop, s.target, t.fill_bar, last)
        r_main = r_of(t.side, t.entry, px, risk)
        r_alt = r_of(t.side, t.entry, s.target, risk) if amb else r_main
        rows.append({"signal_bar": t.bar, "signal_time": str(df4.index[t.bar]), "side": t.side,
                     "entry": t.entry, "stop": s.stop, "target": s.target, "risk_pct": 100 * risk / t.entry,
                     "exit_bar_4h": t.exit_bar, "exit_time_4h": str(df4.index[t.exit_bar]), "exit_4h": t.exit,
                     "reason_4h": t.reason, "r_4h": t.r,
                     "exit_time_1m": str(pd.Timestamp(int(M.ts[mi]), unit="s")), "exit_bar_1m": int(np.searchsorted(M.bar_start, mi, side="right") - 1),
                     "exit_1m": px, "reason_1m": reason, "ambiguous_minute": amb, "r_1m": r_main, "r_1m_alt": r_alt})
    return pd.DataFrame(rows)


def chained(M, df4, setups):
    """Full one-trade-at-a-time simulation with minute exits (re-derives the trade set)."""
    o = df4["open"].to_numpy(float)
    n = len(df4)
    busy, out = -1, []
    for s in sorted(setups, key=lambda s: s.bar):
        if s.bar <= busy or s.bar + 1 >= n:
            continue
        f, entry = s.bar + 1, o[s.bar + 1]
        risk = s.side * (entry - s.stop)
        if risk <= 0:
            busy = f
            continue
        last = min(f + s.max_hold - 1, n - 1)
        mi, px, reason, amb = minute_exit(M, s.side, entry, s.stop, s.target, f, last)
        xb = int(np.searchsorted(M.bar_start, mi, side="right") - 1)
        out.append((s.bar, s.side, r_of(s.side, entry, px, risk), reason))
        busy = xb
    return out


def period_of(df4, bars):
    idx = df4.index
    lab = np.full(len(bars), "", dtype=object)
    for name, (a, b) in PERIODS.items():
        lo, hi = idx.searchsorted(pd.Timestamp(a)), idx.searchsorted(pd.Timestamp(b), side="right")
        lab[(bars >= lo) & (bars < hi)] = name
    return lab


def stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "avg_r": round(float(x.mean()), 3), "win_pct": round(100 * float((x > 0).mean()), 1),
            "t": round(float(x.mean() / (sd / np.sqrt(len(x)))), 2) if sd > 0 else 0.0}


def slippage(M, df4, tr):
    """Entry: 4h open vs the first minutes after the open, signed so positive = adverse."""
    o4 = df4["open"].to_numpy(float)
    c4 = df4["close"].to_numpy(float)
    res = {}
    rows = []
    for _, t in tr.iterrows():
        f = int(t.signal_bar) + 1
        a = M.bar_start[f]
        side, entry, risk = int(t.side), float(t.entry), float(t.entry * t.risk_pct / 100)
        row = {"first_min_open_minus_4h_open": M.o[a] - o4[f], "prev_close_gap": side * (o4[f] - c4[f - 1]) / o4[f] * 100}
        for k in (1, 5, 15, 60):
            sl = slice(a, a + k)
            tp = (M.h[sl] + M.l[sl] + M.c[sl]) / 3
            vol = M.v[sl]
            vwap = (tp * vol).sum() / vol.sum() if vol.sum() > 0 else M.c[a + k - 1]
            worst = M.h[sl].max() if side > 0 else M.l[sl].min()
            row[f"vwap{k}_pct"] = side * (vwap - entry) / entry * 100
            row[f"vwap{k}_R"] = side * (vwap - entry) / risk
            row[f"close{k}_pct"] = side * (M.c[a + k - 1] - entry) / entry * 100
            row[f"worst{k}_pct"] = side * (worst - entry) / entry * 100
            row[f"worst{k}_R"] = side * (worst - entry) / risk
        row["first_min_volume_zero"] = bool(M.v[a] == 0)
        row["first_min_range_pct"] = (M.h[a] - M.l[a]) / entry * 100
        row["risk_pct"] = float(t.risk_pct)
        row["period"] = t.period
        rows.append(row)
    s = pd.DataFrame(rows)
    for per, g in [("all", s)] + list(s.groupby("period")):
        d = {"n": len(g), "max_abs_first_minute_open_vs_4h_open": float(g.first_min_open_minus_4h_open.abs().max()),
             "median_risk_pct": round(float(g.risk_pct.median()), 3),
             "share_first_minute_zero_volume": round(float(g.first_min_volume_zero.mean()), 3),
             "median_first_minute_range_pct": round(float(g.first_min_range_pct.median()), 4),
             "signal_close_to_next_open_adverse_pct_mean": round(float(g.prev_close_gap.mean()), 4)}
        for k in (1, 5, 15, 60):
            d[f"vwap_first_{k}m"] = {"mean_pct": round(float(g[f"vwap{k}_pct"].mean()), 4),
                                     "median_pct": round(float(g[f"vwap{k}_pct"].median()), 4),
                                     "mean_R": round(float(g[f"vwap{k}_R"].mean()), 4),
                                     "median_R": round(float(g[f"vwap{k}_R"].median()), 4)}
            d[f"close_after_{k}m_mean_pct"] = round(float(g[f"close{k}_pct"].mean()), 4)
            d[f"worst_price_first_{k}m"] = {"mean_pct": round(float(g[f"worst{k}_pct"].mean()), 4),
                                            "median_pct": round(float(g[f"worst{k}_pct"].median()), 4),
                                            "mean_R": round(float(g[f"worst{k}_R"].mean()), 4)}
        res[per] = d
    return res


def main():
    df4 = load("4h")
    M = Minutes(df4)
    results = {"bar_check_max_abs_diff": M.check_bars(df4)}
    configs = list(itertools.product(*GRID.values()))
    grid = []
    for cfg in configs:
        kind, lb, rr, filt = cfg
        setups = setups_for(df4, kind, lb, rr, filt)
        trades = simulate(df4, setups, COST)
        tr = replay(M, df4, setups, trades)
        tr["period"] = period_of(df4, tr.signal_bar.to_numpy())
        ch = chained(M, df4, setups)
        ch_bars = np.array([x[0] for x in ch])
        ch_per = period_of(df4, ch_bars)
        row = {"kind": kind, "lb": lb, "rr": rr, "filter": filt,
               "same_trade_set_as_4h": [(x[0], x[1]) for x in ch] == [(t.bar, t.side) for t in trades]}
        for p in PERIODS:
            g = tr[tr.period == p]
            row[p] = {"n": len(g), "avg_r_4h": round(float(g.r_4h.mean()), 3) if len(g) else None,
                      "avg_r_1m": round(float(g.r_1m.mean()), 3) if len(g) else None,
                      "avg_r_1m_alt": round(float(g.r_1m_alt.mean()), 3) if len(g) else None,
                      "outcome_changed": int((g.reason_4h != g.reason_1m).sum()),
                      "stop_to_target": int(((g.reason_4h == "stop") & (g.reason_1m == "target")).sum()),
                      "ambiguous_minutes": int(g.ambiguous_minute.sum()),
                      "exit_price_changed_same_reason": int(((g.reason_4h == g.reason_1m) & ((g.exit_4h - g.exit_1m).abs() > 1e-9)).sum()),
                      "chained_1m": stats([x[2] for x, q in zip(ch, ch_per) if q == p])}
        grid.append(row)
        if cfg in TOP:
            name = f"{kind}_lb{lb}_rr{rr:g}_{filt}"
            tr.to_csv(OUT / f"trades_{name}.csv", index=False)
            detail = {"config": name}
            for p in PERIODS:
                g = tr[tr.period == p]
                ch_g = g[g.reason_4h != g.reason_1m]
                detail[p] = {"4h": stats(g.r_4h), "1m": stats(g.r_1m), "1m_alt_ambiguous_as_target": stats(g.r_1m_alt),
                             "changed": ch_g[["signal_time", "side", "reason_4h", "r_4h", "reason_1m", "exit_time_1m", "r_1m"]].round(3).to_dict("records"),
                             "stop_gap_fills_worse": int(((g.reason_1m == "stop") & (g.side * (g.exit_1m - g.stop) < -1e-9)).sum()),
                             "mean_r_delta_same_reason": round(float((g.r_1m - g.r_4h)[g.reason_4h == g.reason_1m].mean()), 4),
                             "long": {"4h": stats(g[g.side > 0].r_4h), "1m": stats(g[g.side > 0].r_1m)},
                             "short": {"4h": stats(g[g.side < 0].r_4h), "1m": stats(g[g.side < 0].r_1m)}}
                # How many 4h bars had both stop and target inside (the conservative rule fired)?
            detail["slippage"] = slippage(M, df4, tr)
            results.setdefault("top_configs", {})[name] = detail
    results["grid"] = grid
    gs = {}
    for p in PERIODS:
        a4 = [r[p]["avg_r_4h"] or 0.0 for r in grid]
        a1 = [r[p]["avg_r_1m"] or 0.0 for r in grid]
        gs[p] = {"median_avg_r_4h": round(float(np.median(a4)), 3), "median_avg_r_1m": round(float(np.median(a1)), 3),
                 "n_positive_4h": int(sum(v > 0 for v in a4)), "n_positive_1m": int(sum(v > 0 for v in a1)),
                 "trades": int(sum(r[p]["n"] for r in grid)),
                 "outcome_changed": int(sum(r[p]["outcome_changed"] for r in grid)),
                 "ambiguous_minutes": int(sum(r[p]["ambiguous_minutes"] for r in grid)),
                 "mean_shift_avg_r": round(float(np.mean(np.array(a1) - np.array(a4))), 4)}
    results["grid_summary"] = gs
    results["all_same_trade_sets"] = all(r["same_trade_set_as_4h"] for r in grid)
    elig = [r for r in grid if r["train"]["n"] >= 30]
    best1m = max(elig, key=lambda r: r["train"]["avg_r_1m"])
    results["chosen_on_train_with_1m_exits"] = {k: best1m[k] for k in ("kind", "lb", "rr", "filter", "pre", "train", "test")}
    (OUT / "results_intrabar.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps({k: v for k, v in results.items() if k != "grid"}, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
