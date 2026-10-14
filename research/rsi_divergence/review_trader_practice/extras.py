"""Additional checks (NOT pre-registered; labelled as such in the report):
1. per-config placebo percentile for all 660 configs (100 reps, random bars, same side/risk/exit);
2. diagnostics: how often 'confirm' == next open, how often rsi50 exits on the fill bar;
3. regbull (long-only, most taught) family detail;
4. sensitivity: study's 'current pivot < 30' filter; cap 120 for indicator exits on the pick.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tp_lib as T

OUT = Path(__file__).parent
P = list(T.PERIODS)
rng = np.random.default_rng(777)
D = T.Data(T.load("4h"))
g = pd.read_csv(OUT / "grid_rows.csv")
res = {}

# 1. per-config placebo percentiles
REPS = 100
pct = {p: [] for p in P}
pl_med = {p: [] for p in P}
for _, row in g.iterrows():
    key = (row.detector, int(row.lb), row.setup, row.entry, row.exit)
    tr = T.simulate(D, T.candidates(D, key[0], key[1], key[2]), key[3], key[4], key[0], key[1])
    per = T.by_period(D, tr)
    for p in P:
        real = T.stats([t.r(0.1) for t in per[p]])["avg_r"]
        if not per[p]:
            pct[p].append(np.nan); pl_med[p].append(np.nan); continue
        v = np.array([T.stats([t.r(0.1) for t in T.placebo(D, per[p], key[4], key[0], key[1], rng)])["avg_r"]
                      for _ in range(REPS)])
        pct[p].append(100 * float((v < real).mean()))
        pl_med[p].append(float(np.median(v)))
for p in P:
    g[f"{p}_pctile"] = pct[p]
    g[f"{p}_placebo_med"] = pl_med[p]
    g[f"{p}_excess"] = g[f"{p}_avg_r_c0.1"] - g[f"{p}_placebo_med"]
g.to_csv(OUT / "grid_rows_with_placebo.csv", index=False)
res["per_config_placebo"] = {
    "reps": REPS,
    **{f"{p}_n_beat95": int((g[f"{p}_pctile"] >= 95).sum()) for p in P},
    **{f"{p}_median_pctile": float(g[f"{p}_pctile"].median()) for p in P},
    **{f"{p}_median_excess_avg_r_vs_placebo": round(float(g[f"{p}_excess"].median()), 3) for p in P},
    "n_beat90_all_three": int(((g.pre_pctile >= 90) & (g.train_pctile >= 90) & (g.test_pctile >= 90)).sum()),
    "n_beat50_all_three": int(((g.pre_pctile > 50) & (g.train_pctile > 50) & (g.test_pctile > 50)).sum()),
    "expected_beat50_all_three_if_null": round(len(g) / 8, 1),
    "test_n_beat95_expected_if_null": round(len(g) * 0.05, 1),
    "excess_test_by_setup": g.groupby("setup")["test_excess"].median().round(3).to_dict(),
    "excess_test_by_exit": g.groupby("exit")["test_excess"].median().round(3).to_dict(),
    "excess_test_by_entry": g.groupby("entry")["test_excess"].median().round(3).to_dict(),
    "excess_test_by_detector": g.groupby("detector")["test_excess"].median().round(3).to_dict(),
    "excess_pre_by_setup": g.groupby("setup")["pre_excess"].median().round(3).to_dict(),
    "excess_train_by_setup": g.groupby("setup")["train_excess"].median().round(3).to_dict(),
}
all3 = g[(g["pre_avg_r_c0.1"] > 0) & (g["train_avg_r_c0.1"] > 0) & (g["test_avg_r_c0.1"] > 0)]
res["all_three_positive_vs_placebo"] = all3[["detector", "lb", "setup", "entry", "exit", "pre_avg_r_c0.1", "train_avg_r_c0.1",
                                             "test_avg_r_c0.1", "test_n_c0.1", "pre_pctile", "train_pctile", "test_pctile"]].round(3).to_dict("records")

# 2. diagnostics
diag = {}
for det, lb, setup in itertools.product(["rsi_pivstop", "price_pivot"], [3, 5], ["regbull", "regular"]):
    cands = T.candidates(D, det, lb, setup)
    same = cancelled = 0
    o, h, l, c = D.o, D.h, D.l, D.c
    for i, side, stop, cur in cands:
        trig = h[cur] if side > 0 else l[cur]
        if side * (c[i] - trig) > 0:
            same += 1
    trs = T.simulate(D, cands, "next_open", "rsi50", det, lb)
    imm = sum(1 for t in trs if t.exit_bar == t.fill + 1 and t.reason == "signal")
    rsi_at_sig = np.array([D.rsi[t.bar] for t in trs if t.side > 0])
    diag[f"{det}|{lb}|{setup}"] = {"n_candidates": len(cands), "pct_confirmed_on_signal_bar": round(100 * same / max(len(cands), 1), 1),
                                   "rsi50_trades": len(trs), "pct_rsi50_exit_after_one_bar": round(100 * imm / max(len(trs), 1), 1),
                                   "median_rsi_at_signal_long": round(float(np.median(rsi_at_sig)), 1) if len(rsi_at_sig) else None}
res["diagnostics"] = diag

# 3. regbull family, test years + side splits not needed (long-only); list all rows
rb = g[g.setup == "regbull"]
res["regbull_family"] = {
    "n_configs": len(rb),
    **{f"{p}_median_avg_r": round(float(rb[f"{p}_avg_r_c0.1"].median()), 3) for p in P},
    **{f"{p}_median_placebo": round(float(rb[f"{p}_placebo_med"].median()), 3) for p in P},
    **{f"{p}_median_pctile": float(rb[f"{p}_pctile"].median()) for p in P},
    "n_all_three_positive": int(((rb["pre_avg_r_c0.1"] > 0) & (rb["train_avg_r_c0.1"] > 0) & (rb["test_avg_r_c0.1"] > 0)).sum()),
    "train_pick": rb[rb["train_n_c0.1"] >= 30].sort_values("train_avg_r_c0.1").iloc[-1][
        ["detector", "lb", "entry", "exit", "pre_avg_r_c0.1", "train_avg_r_c0.1", "test_avg_r_c0.1", "test_n_c0.1",
         "test_t_c0.1", "pre_pctile", "train_pctile", "test_pctile"]].to_dict(),
}
rbo = g[g.setup == "regbull_os"]
res["regbull_os_family"] = {
    **{f"{p}_median_avg_r": round(float(rbo[f"{p}_avg_r_c0.1"].median()), 3) for p in P},
    **{f"{p}_median_pctile": float(rbo[f"{p}_pctile"].median()) for p in P},
    "train_pick": rbo[rbo["train_n_c0.1"] >= 30].sort_values("train_avg_r_c0.1").iloc[-1][
        ["detector", "lb", "entry", "exit", "pre_avg_r_c0.1", "train_avg_r_c0.1", "train_n_c0.1", "test_avg_r_c0.1",
         "test_n_c0.1", "test_pctile"]].to_dict() if (rbo["train_n_c0.1"] >= 30).any() else None,
    "median_train_n": float(rbo["train_n_c0.1"].median()),
}

# 4a. study's extreme filter: CURRENT RSI pivot < 30, regular bullish long only, rsi_pivstop
sens = {}
for lb, ex, entry in itertools.product([3, 5], ["rr2", "chand2", "rsi50", "time30"], ["next_open", "confirm"]):
    cands = [(e.i, 1, e.stop_piv, e.cur) for e in T.divergences(D, "rsi", lb, 1, "regular") if e.rsi_cur < 30]
    per = T.by_period(D, T.simulate(D, cands, entry, ex, "rsi_pivstop", lb))
    sens[f"lb{lb}|{entry}|{ex}"] = {p: T.stats([t.r(0.1) for t in per[p]]) for p in P}
res["regbull_current_pivot_lt30"] = {k: {p: (v[p]["n"], v[p]["avg_r"]) for p in P} for k, v in sens.items()}

# 4b. pick with longer caps for time / indicator exits
pick = ("price_pivot", 3, "regular", "confirm")
cap = {}
for ex, capv in [("time60", 60), ("time90", 90), ("time120", 120), ("time30", 30)]:
    T_spec = ex
    per = T.by_period(D, T.simulate(D, T.candidates(D, *pick[:3]), pick[3], ex, pick[0], pick[1]))
    cap[ex] = {p: T.stats([t.r(0.1) for t in per[p]]) for p in P}
res["pick_time_exit_sensitivity"] = {k: {p: (v[p]["n"], v[p]["avg_r"], v[p]["t"]) for p in P} for k, v in cap.items()}

(OUT / "results_extras.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str)[:12000])
