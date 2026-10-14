"""Robustness checks on the minute replay (run intrabar.py first).

- both_in_bar: 4h stop exits whose exit bar also reached the target (the conservative rule fired)
  and how the minutes resolved them.
- touch-only targets: target exits where price went beyond the target by < 0.02% (a resting limit
  might not fill on a touch).
- wick stops: stop exits where price stayed beyond the stop for <= 1 / <= 5 minutes and the trade
  would have reached the target later (on another venue such a wick might not exist).
- stop fill slippage: stop fills at the worse of the stop and the trigger minute's close.
- sensitivity: stop only on a 1-minute CLOSE beyond the stop (not the study's rule; shown to bound
  how much single-venue wicks matter).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from intrabar import TOP, Minutes, period_of, r_of, stats  # noqa: E402
from pinetest.events import simulate  # noqa: E402
from research.rsi_divergence.run import COST, GRID, PERIODS, load, setups_for  # noqa: E402

OUT = Path(__file__).parent


def analyse(M, df4, setups, trades):
    n = len(df4)
    h4, l4 = df4["high"].to_numpy(float), df4["low"].to_numpy(float)
    st = {(s.bar, s.side): s for s in setups}
    rows = []
    for t in trades:
        s = st[(t.bar, t.side)]
        side, entry, stop, target = t.side, t.entry, s.stop, s.target
        risk = side * (entry - stop)
        last = min(t.fill_bar + s.max_hold - 1, n - 1)
        a, b = M.bar_start[t.fill_bar], M.bar_end[last]
        O, H, L, C = M.o[a:b], M.h[a:b], M.l[a:b], M.c[a:b]
        adv = L if side > 0 else H            # adverse extreme
        fav = H if side > 0 else L
        sh = side * (adv - stop) <= 0
        th = side * (fav - target) >= 0
        js = int(np.argmax(sh)) if sh.any() else None
        jt = int(np.argmax(th)) if th.any() else None
        j = t.exit_bar
        both_in_bar = t.reason == "stop" and (h4[j] >= target if side > 0 else l4[j] <= target)
        rec = {"bar": t.bar, "side": side, "reason_4h": t.reason, "r_4h": t.r, "both_in_bar": both_in_bar}
        # stop-fill slippage (pessimistic): worse of stop and the trigger minute's close/open
        if js is not None and (jt is None or js < jt):
            fill = O[js] if side * (O[js] - stop) <= 0 else stop
            pess = fill if side * (C[js] - fill) >= 0 else C[js]
            rec["stop_slip_R"] = side * (fill - pess) / risk
            # time spent beyond the stop
            beyond = side * (C - stop) <= 0
            k = js
            run = 0
            while k < len(C) and (sh[k] if k == js else side * (adv[k] - stop) <= 0):
                run += 1
                k += 1
            rec["stop_wick_minutes"] = run
            rec["target_later"] = bool(jt is not None)
            # close-based stop: first minute closing beyond the stop
            jc = int(np.argmax(beyond)) if beyond.any() else None
        else:
            jc = None
            beyond = side * (C - stop) <= 0
            jc = int(np.argmax(beyond)) if beyond.any() else None
        if jt is not None and (js is None or jt < js):
            rec["target_excess_pct"] = float(side * (fav[jt:].max() - target) / target * 100) if side > 0 else float(side * (fav[jt:].min() - target) / target * 100)
            # excess within the target minute alone
            rec["target_minute_excess_pct"] = float(side * (fav[jt] - target) / target * 100)
        # close-based stop variant
        if jc is not None and (jt is None or jc < jt):
            r_close = r_of(side, entry, C[jc], risk)
        elif jt is not None:
            px = O[jt] if side * (O[jt] - target) >= 0 else target
            r_close = r_of(side, entry, px, risk)
        else:
            r_close = r_of(side, entry, C[-1], risk)
        rec["r_close_stop"] = r_close
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df4 = load("4h")
    M = Minutes(df4)
    out = {}
    grid_both = {p: 0 for p in PERIODS}
    grid_trades = {p: 0 for p in PERIODS}
    for cfg in itertools.product(*GRID.values()):
        kind, lb, rr, filt = cfg
        setups = setups_for(df4, kind, lb, rr, filt)
        trades = simulate(df4, setups, COST)
        d = analyse(M, df4, setups, trades)
        d["period"] = period_of(df4, d.bar.to_numpy())
        for p in PERIODS:
            g = d[d.period == p]
            grid_both[p] += int(g.both_in_bar.sum())
            grid_trades[p] += len(g)
        if cfg in TOP:
            name = f"{kind}_lb{lb}_rr{rr:g}_{filt}"
            res = {}
            for p in PERIODS:
                g = d[d.period == p]
                stops = g[g.reason_4h == "stop"]
                tg = g[g.target_minute_excess_pct.notna()] if "target_minute_excess_pct" in g else g.iloc[:0]
                res[p] = {
                    "n": len(g), "stop_exits": len(stops), "both_in_same_4h_bar": int(g.both_in_bar.sum()),
                    "stop_slip_pessimistic_mean_R": round(float(stops.stop_slip_R.mean()), 4),
                    "stops_wick_le_1min": int((stops.stop_wick_minutes <= 1).sum()),
                    "stops_wick_le_5min": int((stops.stop_wick_minutes <= 5).sum()),
                    "stops_wick_le_5min_then_target": int(((stops.stop_wick_minutes <= 5) & stops.target_later).sum()),
                    "target_exits": len(tg),
                    "targets_touched_lt_0.02pct_in_minute": int((tg.target_minute_excess_pct < 0.02).sum()),
                    "targets_never_exceeded_0.02pct_after": int((tg.target_excess_pct < 0.02).sum()),
                    "avg_r_4h": round(float(g.r_4h.mean()), 3),
                    "avg_r_stop_on_1m_close": round(float(g.r_close_stop.mean()), 3),
                    "avg_r_pessimistic_stop_fill": round(float((g.r_4h - g.stop_slip_R.fillna(0)).mean()), 3),
                }
            out[name] = res
    out["grid_both_in_same_4h_bar"] = {p: f"{grid_both[p]} of {grid_trades[p]} trades" for p in PERIODS}
    (OUT / "results_extras.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
