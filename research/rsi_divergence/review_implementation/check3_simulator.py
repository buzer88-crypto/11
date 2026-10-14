"""Check 3: simulator. (1) Print a sample of real chosen-config trades bar by bar for hand checks.
(2) Rebuild 4h bars independently from the 1-minute source (read as data only) and compare with
data/BTCUSD_4h.csv. (3) Replay every chosen-config trade on 1-minute bars (same stop, target,
30 x 4h hold, 0.1%/side) to measure what the 4h conservative rules (stop first when both levels
are inside one bar) cost. (4) Count what the one-trade-at-a-time rule skips and what trading
every setup independently would give."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.events import Setup, simulate, summary

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("study_run", HERE.parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
df = run.load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
idx = df.index
setups = run.setups_for(df, "hidden", 3, 3.0, "trend")
trades = simulate(df, setups, 0.1)
out = {}

# ---------- (1) hand-check sample ----------
lines = []


def show(t, s):
    lines.append(f"\n=== signal {idx[t.bar]} side {t.side:+d} fill {idx[t.fill_bar]} entry {t.entry} "
                 f"stop {s.stop:.2f} target {s.target:.2f} -> {t.reason} at {t.exit} on {idx[t.exit_bar]} r={t.r:.4f}")
    risk = t.side * (t.entry - s.stop)
    exp_r = (t.side * (t.exit - t.entry) - 0.001 * (t.entry + t.exit)) / risk
    lines.append(f"    risk {risk:.2f}; target-entry = {t.side*(s.target-t.entry):.2f} = {t.side*(s.target-t.entry)/risk:.3f} x risk; "
                 f"recomputed r = {exp_r:.4f}")
    for j in range(t.fill_bar, t.exit_bar + 1):
        sh = (l[j] <= s.stop) if t.side > 0 else (h[j] >= s.stop)
        th = (h[j] >= s.target) if t.side > 0 else (l[j] <= s.target)
        lines.append(f"    {j - t.fill_bar + 1:2d} {idx[j]} O {o[j]:.2f} H {h[j]:.2f} L {l[j]:.2f} C {c[j]:.2f}"
                     f"{'  STOP-touch' if sh else ''}{'  TARGET-touch' if th else ''}")


smap = {(s.bar, s.side): s for s in setups}
test_lo = idx.searchsorted(pd.Timestamp("2022-01-01"))
picked = {}
for t in trades:
    if t.bar < test_lo:
        continue
    s = smap[(t.bar, t.side)]
    j = t.exit_bar
    both = ((l[j] <= s.stop) if t.side > 0 else (h[j] >= s.stop)) and ((h[j] >= s.target) if t.side > 0 else (l[j] <= s.target))
    gap = t.reason == "stop" and t.exit != s.stop
    for tag, cond in (("target_long", t.reason == "target" and t.side > 0), ("target_short", t.reason == "target" and t.side < 0),
                      ("stop_long", t.reason == "stop" and t.side > 0 and not gap), ("stop_short", t.reason == "stop" and t.side < 0 and not gap),
                      ("time", t.reason == "time"), ("gap_stop", gap), ("both_in_bar", both)):
        if cond and tag not in picked:
            picked[tag] = t
for tag, t in picked.items():
    lines.append(f"\n##### {tag}")
    show(t, smap[(t.bar, t.side)])
# one-at-a-time: show a setup skipped because a trade was open
busy = [(t.fill_bar, t.exit_bar) for t in trades]
skipped = [s for s in setups if not any(t.bar == s.bar for t in trades)]
out["setups_total"] = len(setups)
out["trades_taken"] = len(trades)
out["setups_skipped_one_at_a_time"] = len(skipped)
if skipped:
    s = next(x for x in skipped if x.bar >= test_lo)
    t = next(t for t in trades if t.bar <= s.bar <= t.exit_bar)
    lines.append(f"\n##### skipped setup at {idx[s.bar]} side {s.side:+d}: trade from signal {idx[t.bar]} "
                 f"(side {t.side:+d}) was open until {idx[t.exit_bar]}")
out["same_bar_stop_and_target_exits"] = sum(
    1 for t in trades if t.reason == "stop" and
    (((h[t.exit_bar] >= smap[(t.bar, t.side)].target) if t.side > 0 else (l[t.exit_bar] <= smap[(t.bar, t.side)].target))))
out["gap_stop_exits"] = sum(1 for t in trades if t.reason == "stop" and t.exit != smap[(t.bar, t.side)].stop)
out["gap_target_exits"] = sum(1 for t in trades if t.reason == "target" and t.exit != smap[(t.bar, t.side)].target)
out["reasons"] = {r: sum(t.reason == r for t in trades) for r in ("stop", "target", "time")}

# ---------- (4) one-at-a-time vs every setup independently ----------
indep = []
for s in setups:
    indep += simulate(df, [s], 0.1)
per_seq = run.by_period(df, trades)
per_ind = run.by_period(df, indep)
out["one_at_a_time"] = {p: summary(v) for p, v in per_seq.items()}
out["every_setup_independent"] = {p: summary(v) for p, v in per_ind.items()}
# signal on the exit bar of the previous trade is skipped by `s.bar <= busy_until`
out["setups_on_previous_exit_bar"] = sum(1 for s in setups for t in trades if s.bar == t.exit_bar and s.bar != t.bar)

# ---------- (2) independent 4h rebuild from 1-minute data ----------
SRC = Path("/home/user/ff137/bitstamp-btcusd-minute-data/data")
m1 = pd.read_csv(SRC / "historical" / "btcusd_bitstamp_1min_2012-2025.csv.gz")
m2 = pd.read_csv(SRC / "updates" / "btcusd_bitstamp_1min_latest.csv")
m = pd.concat([m1, m2], ignore_index=True)
out["minute_rows_raw"] = len(m)
out["minute_duplicate_timestamps"] = int(m["timestamp"].duplicated().sum())
dup_conflict = m[m["timestamp"].duplicated(keep=False)].groupby("timestamp")[["open", "high", "low", "close"]].nunique().max(axis=1)
out["duplicate_timestamps_with_conflicting_prices"] = int((dup_conflict > 1).sum())
m = m.drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
out["historical_file_sorted"] = bool(m1["timestamp"].is_monotonic_increasing)
b = m["timestamp"] // 14400 * 14400
rb = m.groupby(b).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
rb.index = pd.to_datetime(rb.index, unit="s")
rb = rb.loc["2013-01-01":]
common = rb.index.intersection(df.index)
diff = (rb.loc[common, ["open", "high", "low", "close"]] - df.loc[common, ["open", "high", "low", "close"]]).abs().max(axis=1)
out["rebuild_4h"] = {"bars_rebuilt": len(rb), "bars_in_file": len(df), "common": len(common),
                     "bars_with_any_ohlc_diff": int((diff > 1e-9).sum())}
print(out["rebuild_4h"])

# ---------- (3) 1-minute replay of every chosen-config trade ----------
ts = m["timestamp"].to_numpy()
mo, mh, ml, mc = (m[k].to_numpy(float) for k in ("open", "high", "low", "close"))
bar_start = (idx.astype("int64") // 10**9).to_numpy() if idx.dtype.kind == "M" else None
bar_start = np.array([int(x.timestamp()) for x in idx])
rep = []
for t in trades:
    s = smap[(t.bar, t.side)]
    side = t.side
    a = np.searchsorted(ts, bar_start[t.fill_bar])
    last = min(t.fill_bar + 29, len(idx) - 1)
    z = np.searchsorted(ts, bar_start[last] + 14400)
    e = mo[a]
    assert abs(e - t.entry) < 1e-9, (idx[t.bar], e, t.entry)
    risk = side * (e - s.stop)
    res = None
    for k in range(a, z):
        if side * (mo[k] - s.stop) <= 0:
            res = (mo[k], "stop", k); break
        if side * (mo[k] - s.target) >= 0:
            res = (mo[k], "target", k); break
        sh = (ml[k] <= s.stop) if side > 0 else (mh[k] >= s.stop)
        th = (mh[k] >= s.target) if side > 0 else (ml[k] <= s.target)
        if sh:
            res = (s.stop, "stop", k); break
        if th:
            res = (s.target, "target", k); break
    if res is None:
        res = (mc[z - 1], "time", z - 1)
    x, why, k = res
    r = (side * (x - e) - 0.001 * (e + x)) / risk
    rep.append({"bar": t.bar, "side": side, "r4h": t.r, "r1m": r, "why4h": t.reason, "why1m": why})
rep = pd.DataFrame(rep)
rep["period"] = pd.cut(rep["bar"], [-1, idx.searchsorted(pd.Timestamp("2014-01-01")) - 1,
                                     idx.searchsorted(pd.Timestamp("2018-01-01")) - 1,
                                     idx.searchsorted(pd.Timestamp("2022-01-01")) - 1, 10**9],
                       labels=["2013", "pre", "train", "test"])
out["minute_replay"] = {
    "trades": len(rep), "outcome_changed": int((rep.why4h != rep.why1m).sum()),
    "r_changed_more_than_0.01": int(((rep.r4h - rep.r1m).abs() > 0.01).sum()),
    "by_period": {p: {"n": int(len(g)), "avg_r_4h": round(float(g.r4h.mean()), 3), "avg_r_1m": round(float(g.r1m.mean()), 3)}
                  for p, g in rep.groupby("period", observed=True)},
    "changes": rep[rep.why4h != rep.why1m].assign(date=lambda d: [str(idx[b]) for b in d.bar]).to_dict("records"),
}
print(json.dumps({k: v for k, v in out.items()}, indent=1, default=str))
(HERE / "check3_trades_by_hand.txt").write_text("\n".join(lines) + "\n")
(HERE / "check3_simulator.json").write_text(json.dumps(out, indent=1, default=str))
