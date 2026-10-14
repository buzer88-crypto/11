"""Check 4 (+ simulator cross-check from 3): re-run the study's grid and chosen config with the
study code, compare with results.json, then re-run everything with
  (a) the study's setups through an independent simulator (trade-by-trade equality),
  (b) the independent Pine emulation (left-strict pivot ties, barssince range = 6..61 pivot
      distance) through the independent trade builder + simulator,
  (c) the study code with only the range fixed (range_lower=6, range_upper=61).
Nothing outside review_implementation/ is written."""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tv_reimpl import build_trades, pine_ema, pine_rsi, sma_atr, simulate_ref, tv_divergences  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate, summary  # noqa: E402

spec = importlib.util.spec_from_file_location("study_run", HERE.parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)          # main() is guarded; nothing is written

df = run.load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
idx = df.index
stored = json.loads((HERE.parent / "results.json").read_text())
out = {}


def per_period(trs, get_bar=lambda t: t.bar):
    res = {}
    for name, (a, b) in run.PERIODS.items():
        lo, hi = idx.searchsorted(pd.Timestamp(a)), idx.searchsorted(pd.Timestamp(b), side="right")
        r = np.array([t["r"] if isinstance(t, dict) else t.r for t in trs
                      if lo <= (t["bar"] if isinstance(t, dict) else t.bar) < hi])
        if len(r) == 0:
            res[name] = {"n": 0}
            continue
        sd = r.std(ddof=1) if len(r) > 1 else 0
        res[name] = {"n": int(len(r)), "avg_r": round(float(r.mean()), 3),
                     "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else 0.0}
    return res


# ---------- study code, full grid, compare to results.json ----------
grid_mismatch = 0
study_rows = []
sim_mismatch = {"configs_with_trade_differences": 0, "trades_compared": 0}
for kind, lb, rr, filt in itertools.product(*run.GRID.values()):
    setups = run.setups_for(df, kind, lb, rr, filt)
    trades = simulate(df, setups, run.COST)
    per = {p: summary(ts) for p, ts in run.by_period(df, trades).items()}
    row = {"kind": kind, "lb": lb, "rr": rr, "filter": filt, **per}
    srow = next(r for r in stored["grid_rows"] if all(r[k] == row[k] for k in ("kind", "lb", "rr", "filter")))
    if any(srow[p] != per[p] for p in run.PERIODS):
        grid_mismatch += 1
    study_rows.append(row)
    # (a) the same setups through the independent simulator
    mine = simulate_ref([dict(bar=s.bar, side=s.side, stop=s.stop, target=s.target) for s in setups], o, h, l, c,
                        cost_pct=run.COST)
    sim_mismatch["trades_compared"] += len(trades)
    same = len(mine) == len(trades) and all(
        (m["bar"], m["fill_bar"], m["exit_bar"], m["reason"]) == (t.bar, t.fill_bar, t.exit_bar, t.reason)
        and abs(m["r"] - t.r) < 1e-9 for m, t in zip(mine, trades))
    sim_mismatch["configs_with_trade_differences"] += 0 if same else 1
out["study_grid_rows_mismatching_results_json"] = grid_mismatch
out["independent_simulator_vs_events_simulate"] = sim_mismatch
eligible = [r for r in study_rows if r["train"].get("n", 0) >= 30]
best = max(eligible, key=lambda r: r["train"]["avg_r"])
out["study_chosen_on_train"] = {k: best[k] for k in ("kind", "lb", "rr", "filter")}
out["study_chosen_result"] = {p: best[p] for p in run.PERIODS}
print("study grid mismatches vs results.json:", grid_mismatch)
print("simulator cross-check:", sim_mismatch)
print("chosen:", out["study_chosen_on_train"], out["study_chosen_result"])


def grid_summary(rows):
    s = {}
    for p in run.PERIODS:
        v = [r[p].get("avg_r", 0.0) for r in rows]
        s[p] = {"median_avg_r": round(float(np.median(v)), 3), "n_positive": int(sum(x > 0 for x in v))}
    return s


out["study_grid_summary"] = grid_summary(study_rows)

# ---------- (b) independent Pine emulation end-to-end ----------
osc = pine_rsi(c)
atr = sma_atr(h, l, c)
ema200 = pine_ema(c, 200)
TYPES = {("regular", 1): "regular_bull", ("hidden", 1): "hidden_bull",
         ("regular", -1): "regular_bear", ("hidden", -1): "hidden_bear"}
variants = {}
for ties in ("left_strict", "strict"):
    sigs = {lb: tv_divergences(o, h, l, c, lbL=lb, lbR=lb, ties=ties)[0] for lb in (3, 5)}
    rows = []
    for kind, lb, rr, filt in itertools.product(*run.GRID.values()):
        setups = []
        for side in (1, -1):
            setups += build_trades(sigs[lb][TYPES[(kind, side)]], o, h, l, c, osc, atr, ema200, side, rr, filt)
        trs = simulate_ref(setups, o, h, l, c, cost_pct=run.COST)
        rows.append({"kind": kind, "lb": lb, "rr": rr, "filter": filt, **per_period(trs)})
    elig = [r for r in rows if r["train"].get("n", 0) >= 30]
    b = max(elig, key=lambda r: r["train"]["avg_r"])
    pick = next(r for r in rows if (r["kind"], r["lb"], r["rr"], r["filter"]) == ("hidden", 3, 3.0, "trend"))
    tvdefault = [r for r in rows if (r["kind"], r["lb"], r["filter"]) == ("regular", 5, "none")]
    variants[f"tv_emulation_{ties}"] = {
        "chosen_on_train": {k: b[k] for k in ("kind", "lb", "rr", "filter")},
        "chosen_result": {p: b[p] for p in run.PERIODS},
        "study_chosen_config_result": {p: pick[p] for p in run.PERIODS},
        "tv_default_regular_lb5_no_filter": {r["rr"]: {p: r[p] for p in run.PERIODS} for r in tvdefault},
        "grid_summary": grid_summary(rows),
        "rows": rows,
    }
    print(ties, "chosen", variants[f"tv_emulation_{ties}"]["chosen_on_train"], variants[f"tv_emulation_{ties}"]["chosen_result"])
    print(ties, "study's config", variants[f"tv_emulation_{ties}"]["study_chosen_config_result"])
    print(ties, "grid", variants[f"tv_emulation_{ties}"]["grid_summary"])

# ---------- (c) study code with the 6..61 range ----------
rows = []
for kind, lb, rr, filt in itertools.product(*run.GRID.values()):
    ss = rsi_divergences(df, kind=kind, left=lb, right=lb, rr=rr, extreme=filt == "extreme",
                         trend=filt == "trend", range_lower=6, range_upper=61)
    per = {p: summary(ts) for p, ts in run.by_period(df, simulate(df, ss, run.COST)).items()}
    rows.append({"kind": kind, "lb": lb, "rr": rr, "filter": filt, **per})
elig = [r for r in rows if r["train"].get("n", 0) >= 30]
b = max(elig, key=lambda r: r["train"]["avg_r"])
pick = next(r for r in rows if (r["kind"], r["lb"], r["rr"], r["filter"]) == ("hidden", 3, 3.0, "trend"))
variants["study_code_range_6_61"] = {
    "chosen_on_train": {k: b[k] for k in ("kind", "lb", "rr", "filter")},
    "chosen_result": {p: b[p] for p in run.PERIODS},
    "study_chosen_config_result": {p: pick[p] for p in run.PERIODS},
    "grid_summary": grid_summary(rows), "rows": rows}
print("range fix chosen", variants["study_code_range_6_61"]["chosen_on_train"], variants["study_code_range_6_61"]["chosen_result"])
print("range fix study's config", variants["study_code_range_6_61"]["study_chosen_config_result"])
print("range fix grid", variants["study_code_range_6_61"]["grid_summary"])
out["variants"] = variants
(HERE / "check4_rerun.json").write_text(json.dumps(out, indent=1, default=str))
