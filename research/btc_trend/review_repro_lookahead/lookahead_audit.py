"""Check 2: lookahead audit by truncation / perturbation of run.py's own `weights` function and of
pinetest.portfolio.backtest / rebalance_on. run.py is NOT imported (it would rerun and rewrite
results.json); its `weights` function source is extracted with ast and exec'd against truncated data.
"""
import ast
import itertools
import json
import numpy as np
import pandas as pd

from pinetest.crypto_data import load_panel
from pinetest.portfolio import backtest, rebalance_on

SRC = "/home/user/11/research/btc_trend/run.py"
tree = ast.parse(open(SRC).read())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "weights")
fn_src = ast.get_source_segment(open(SRC).read(), fn)

px_full = load_panel()["PriceUSD"][["btc", "eth"]]  # only the columns the strategy touches


def make_weights(px):
    ns = {"np": np, "pd": pd, "px": px, "rebalance_on": rebalance_on}
    exec(fn_src, ns)
    return ns["weights"]


wf_full = make_weights(px_full)
GRID = list(itertools.product(["sma", "ret"], [50, 100, 150, 200], ["full", "vt40"], ["D", "W"]))
rng = np.random.default_rng(7)
idx = px_full.index
cand = idx[(idx >= "2012-01-01") & (idx <= "2026-05-20")]
cuts = sorted(set(rng.choice(cand, 60, replace=False)) | {pd.Timestamp("2021-12-31"), pd.Timestamp("2022-01-01"),
                                                         pd.Timestamp("2017-12-31"), pd.Timestamp("2022-06-15")})
report = {"n_cuts": len(cuts), "configs": len(GRID)}

# (a) weights truncation test for all 32 configs (BTC and ETH)
mism = {}
for cfg in GRID:
    cfgd = dict(zip(["signal", "n", "sizing", "rebal"], cfg))
    for a in ("btc", "eth"):
        full = wf_full(a, **cfgd)
        bad = 0
        bad_lastrow_only = 0
        for c in cuts:
            tr = make_weights(px_full.loc[:c])(a, **cfgd)
            f = full.loc[:c]
            eq = (tr.fillna(-9) == f.fillna(-9)).all(axis=1) | np.isclose(tr.fillna(-9), f.fillna(-9), rtol=0, atol=1e-12).all(axis=1)
            if not eq.all():
                if (~eq).sum() == 1 and not eq.iloc[-1]:
                    bad_lastrow_only += 1
                else:
                    bad += 1
        mism[f"{a}_{'_'.join(map(str, cfg))}"] = (bad, bad_lastrow_only)
report["weights_trunc_mismatch_cuts_excluding_lastrow"] = sum(v[0] for v in mism.values())
report["weights_trunc_mismatch_only_last_row"] = {k: v[1] for k, v in mism.items() if v[1]}

# (b) future-perturbation test on the chosen config: scramble prices after the cut, compare weights before the cut
chosen = dict(signal="sma", n=50, sizing="vt40", rebal="D")
w_full = wf_full("btc", **chosen)
res_full = backtest(w_full, px_full)
pert_bad_w, pert_bad_r = 0, 0
for c in cuts:
    pp = px_full.copy()
    after = pp.index > c
    pp.loc[after, "btc"] = pp.loc[after, "btc"] * np.exp(rng.normal(0, 0.2, after.sum()))
    w2 = make_weights(pp)("btc", **chosen)
    if not np.allclose(w2.loc[:c].to_numpy(), w_full.loc[:c].to_numpy(), atol=1e-12):
        pert_bad_w += 1
    r2 = backtest(w2, pp).returns
    # returns up to and including the cut use only prices <= cut
    if not np.allclose(r2.loc[:c].to_numpy(), res_full.returns.loc[:c].to_numpy(), atol=1e-12):
        pert_bad_r += 1
report["future_perturbation_weight_mismatches"] = pert_bad_w
report["future_perturbation_return_mismatches"] = pert_bad_r

# (c) backtest truncation: run on truncated prices+weights; returns must match the full run
bt_bad = 0
for c in cuts:
    r = backtest(w_full.loc[:c], px_full.loc[:c]).returns
    if not np.allclose(r.to_numpy(), res_full.returns.loc[:c].to_numpy(), atol=1e-12):
        bt_bad += 1
report["backtest_truncation_mismatches"] = bt_bad

# (d) impulse response: change the weight on a single day t0; returns must change only from t0+lag+1 on
t0 = pd.Timestamp("2023-03-01")
for lag in (0, 1, 2):
    w = pd.DataFrame(0.0, index=px_full.index, columns=px_full.columns)
    w2 = w.copy()
    w2.loc[t0:, "btc"] = 1.0
    d = (backtest(w2, px_full, lag=lag).returns - backtest(w, px_full, lag=lag).returns).abs()
    first = d[d > 0].index[0]
    report[f"impulse_lag{lag}_first_return_affected"] = f"{first.date()} (= t0 + {(first - t0).days} days; the day-t0+{lag} return is the cost of the trade at close t0+{lag})"

# (e) rolling-window audit: SMA and vol at t vs explicit windows ending at t
p = px_full["btc"]
sma = p.rolling(50).mean()
vol = p.pct_change(fill_method=None).rolling(60).std() * np.sqrt(365)
t = px_full.index.get_loc(pd.Timestamp("2022-05-10"))
report["sma50_check"] = bool(np.isclose(sma.iloc[t], p.iloc[t - 49:t + 1].mean()))
report["vol60_check"] = bool(np.isclose(vol.iloc[t], (p.iloc[t - 60:t + 1].pct_change().iloc[1:]).std() * np.sqrt(365)))
report["vol_ddof"] = 1

# (f) rebalance_on edge: keeps the last row of each W-SUN period of the AVAILABLE data
w = pd.DataFrame({"btc": np.arange(len(idx), dtype=float)}, index=idx)
rb_full = rebalance_on(w)
wed = pd.Timestamp("2024-03-13")  # a Wednesday
rb_tr = rebalance_on(w.loc[:wed])
report["rebalance_on_last_partial_week"] = {
    "cut": str(wed.date()), "weekday": wed.day_name(),
    "truncated_keeps_cut_row": bool(rb_tr.loc[wed].notna().all()),
    "full_keeps_cut_row": bool(rb_full.loc[wed].notna().all()),
    "nonsunday_rows_kept_in_full_history": [str(d.date()) for d in rb_full.dropna().index if d.dayofweek != 6],
}

# (g) timestamp convention: PriceUSD(D) vs ReferenceRateUSD (00:00 UTC fix stamped with its date)
d = pd.read_csv("/home/user/coinmetrics/data/csv/btc.csv", usecols=["time", "PriceUSD", "ReferenceRateUSD", "AssetEODCompletionTime"])
d.index = pd.to_datetime(d.pop("time"))
both = d.dropna(subset=["ReferenceRateUSD"])
cmp = pd.DataFrame({"PriceUSD_D": d["PriceUSD"], "RefRate_D": d["ReferenceRateUSD"], "RefRate_D+1": d["ReferenceRateUSD"].shift(-1)}).dropna()
report["timestamp"] = {
    "days_with_ReferenceRateUSD": len(both),
    "max_rel_diff_PriceUSD_D_vs_RefRate_D+1": float(((cmp.PriceUSD_D - cmp["RefRate_D+1"]).abs() / cmp.PriceUSD_D).max()),
    "median_rel_diff_PriceUSD_D_vs_RefRate_D": float(((cmp.PriceUSD_D - cmp.RefRate_D).abs() / cmp.PriceUSD_D).median()),
    "EOD_completion_hours_after_D_00UTC_2022plus_min": float(((pd.to_datetime(d["AssetEODCompletionTime"], unit="s") - d.index).dt.total_seconds() / 3600).loc["2022":].min()),
}
print(json.dumps(report, indent=1, default=str))
json.dump(report, open("/home/user/11/research/btc_trend/review_repro_lookahead/lookahead_audit.json", "w"), indent=1, default=str)
