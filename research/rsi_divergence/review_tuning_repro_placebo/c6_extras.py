"""Extras for checks 2-4:
(a) placebo comparison stratified by year x next-60-bar direction (real vs placebo within cell),
(b) excess R over a plain 60-bar hold from the same entry (stop/target/exit-management value),
    real vs timing placebo,
(c) timing placebo on the detrended series for the picks' detrended trades."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lib import COST, TEST, by_year, cache, load, sim_long_independent, trades_for
from c4_detrend import detrended, PICKS

OUT = Path(__file__).parent
D = 20000
rng = np.random.default_rng(20261016)
cfgs = cache()["configs"]

def pct_rank(real, null):
    null = np.asarray(null); null = null[np.isfinite(null)]
    return {"real": round(float(real), 3), "null_mean": round(float(null.mean()), 3),
            "null_p05": round(float(np.quantile(null, 0.05)), 3), "null_p95": round(float(np.quantile(null, 0.95)), 3),
            "pct_null_below_real": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4)}

def run(df, tr):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c); yrs = df.index.year.to_numpy()
    R = np.empty((D, len(tr))); F = np.empty((D, len(tr))); H60 = np.empty((D, len(tr)))
    col = 0
    for y in TEST:
        sub = tr[tr.year == y]; k = len(sub)
        if k == 0:
            continue
        cand = np.flatnonzero((yrs == y) & (np.arange(n) + 1 <= n - 1))
        bars = np.stack([rng.choice(cand, size=k, replace=False) for _ in range(D)])
        rp = np.broadcast_to(sub.risk_pct.to_numpy(), (D, k))
        r, _, fill = sim_long_independent(o, h, l, c, bars, rp, 4.0, 60)
        last = np.minimum(fill + 59, n - 1)
        R[:, col:col + k] = r
        F[:, col:col + k] = c[last] / o[fill] - 1
        H60[:, col:col + k] = (c[last] - o[fill] - COST / 100 * (o[fill] + c[last])) / (rp * o[fill])
        col += k
    fill = tr.fill_bar.to_numpy(); last = np.minimum(fill + 59, n - 1)
    f_real = c[last] / o[fill] - 1
    h60_real = (c[last] - o[fill] - COST / 100 * (o[fill] + c[last])) / (tr.risk_pct.to_numpy() * o[fill])
    return R, F, H60, f_real, h60_real

def analyse(tr, R, F, H60, f_real, h60_real):
    res = {}
    for lab, m in (("all", np.ones(len(tr), bool)), ("ex2017", (tr.year != 2017).to_numpy())):
        r = tr.r.to_numpy()[m]
        # (a) stratified by year x direction: expected placebo R for each real trade = mean placebo R
        # in its (year, up/down) cell, per draw
        yr = tr.year.to_numpy()[m]; up = f_real[m] > 0
        Rm, Fm = R[:, m], F[:, m]
        exp_draw = np.zeros(D); cell_out = {}
        for y in np.unique(yr):
            cols = yr == y
            for d_ in (True, False):
                k_real = int(((yr == y) & (up == d_)).sum())
                if k_real == 0:
                    continue
                sel = (Fm[:, cols] > 0) == d_
                vals = np.where(sel, Rm[:, cols], np.nan)
                with np.errstate(all="ignore"):
                    per_draw = np.nanmean(vals, 1)
                # draws with no placebo in this cell: fall back to the pooled cell mean
                per_draw = np.where(np.isfinite(per_draw), per_draw, np.nanmean(vals))
                exp_draw += k_real * per_draw
                cell_out[f"{y}_{'up' if d_ else 'down'}"] = {"n_real": k_real, "real_avg_r": round(float(r[(yr == y) & (up == d_)].mean()), 3),
                                                             "placebo_avg_r": round(float(np.nanmean(vals)), 3)}
        exp_draw /= len(r)
        res[lab] = {"stratified_year_x_dir": pct_rank(r.mean(), exp_draw), "cells": cell_out,
                    # (b) excess over 60-bar hold from the same entry
                    "excess_over_hold60": pct_rank((r - h60_real[m]).mean(), (Rm - H60[:, m]).mean(1)),
                    "real_hold60_avg_r": round(float(h60_real[m].mean()), 3),
                    "placebo_hold60_avg_r": round(float(H60[:, m].mean()), 3),
                    "timing_placebo": pct_rank(r.mean(), Rm.mean(1))}
    return res

out = {}
df = load("4h")
tr = pd.read_csv(OUT / "trades_expanding_avg_r.csv")
R, F, H60, f_real, h60_real = run(df, tr)
out["raw"] = analyse(tr, R, F, H60, f_real, h60_real)
print("raw", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in out["raw"].items()}, indent=0), flush=True)

# (c) detrended: regenerate the picks' trades on the detrended series, then the same placebo there
dd = detrended()
rows = []
from pinetest.divergence import rsi_divergences
for y in TEST:
    i = PICKS[y]; cfg = cfgs[i]
    ts = by_year(dd, trades_for(dd, cfg))[y]
    st = {s.bar: s.stop for s in rsi_divergences(dd, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                                                 buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"],
                                                 trend=cfg["trend"], extreme=cfg["extreme"], sides=(1,))}
    for t in ts:
        rows.append({"year": y, "pick": i, "bar": t.bar, "fill_bar": t.fill_bar, "exit_bar": t.exit_bar, "entry": t.entry,
                     "exit": t.exit, "reason": t.reason, "r": t.r, "risk_pct": (t.entry - st[t.bar]) / t.entry})
trd = pd.DataFrame(rows)
trd.to_csv(OUT / "trades_picks_detrended.csv", index=False)
R, F, H60, f_real, h60_real = run(dd, trd)
out["detrended"] = analyse(trd, R, F, H60, f_real, h60_real)
print("detrended", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"} for k, v in out["detrended"].items()}, indent=0))
(OUT / "c6_extras.json").write_text(json.dumps(out, indent=1, default=str))
