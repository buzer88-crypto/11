"""Check 3: is circular_shift a fair placebo? Independent placebos for SMA50 / vt40 / daily on BTC.

P1  circular_shift from the harness, more draws / other seeds; also report the placebos' exposure.
P2  spell shuffle: inside each period, the real trend on/off series is cut into its on-spells and
    off-spells, which are shuffled separately and re-interleaved (same number of switches, same
    on-day fraction, same spell-length distribution); the REAL real-time vol-target size is kept.
    This randomises only the trend timing.
P3  within-period circular shift of the on/off series only (real-time size kept).
P4  stationary block bootstrap of BTC daily log returns inside the period (mean block 5 / 20 / 60 d),
    real prices as warm-up; strategy and buy&hold are run on the same synthetic path, statistic =
    Sharpe(strategy) - Sharpe(buy&hold) and maxDD(strategy) - maxDD(buy&hold).
P5  paired stationary bootstrap (mean block 20 d) of the realised daily returns: CI of the Sharpe
    difference strategy - buy&hold, and of strategy - constant-weight (same mean exposure).
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/11/research/btc_trend/review_repro_lookahead")
from repro import run  # own single-asset backtester (verified identical to the harness)
from pinetest.crypto_data import load_panel
from pinetest.portfolio import backtest, circular_shift

ANN = 365
PER = {"test": ("2022-01-01", "2026-05-24"), "pre": ("2014-01-01", "2017-12-31")}
px = load_panel()["PriceUSD"]
btc = px["btc"].loc["2010-07-18":]
idx = btc.index


def sig_size(p):
    on = (p > p.rolling(50).mean()).astype(float)
    vol = p.pct_change(fill_method=None).rolling(60).std() * np.sqrt(ANN)
    size = (0.40 / vol).clip(upper=1.0)
    return on, size


def sharpe(x):
    return float(x.mean() / x.std(ddof=1) * np.sqrt(ANN))


def mdd(x):
    eq = np.cumprod(1 + np.asarray(x))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


on, size = sig_size(btc)
w_real = (on * size).where(btc.notna(), 0.0).fillna(0.0)
r_real, e_real = run(btc, w_real)
bh_r, bh_e = run(btc, pd.Series(np.where(btc.notna(), 1.0, 0.0), idx))
out = {}
real = {k: {"sharpe": sharpe(r_real.loc[a:b]), "bh_sharpe": sharpe(bh_r.loc[a:b]), "mdd": mdd(r_real.loc[a:b]),
            "bh_mdd": mdd(bh_r.loc[a:b]), "exposure": float(e_real.loc[a:b].mean())} for k, (a, b) in PER.items()}
out["real"] = real

# ---- P1: harness circular_shift -------------------------------------------------------------
W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
W["btc"] = w_real.reindex(px.index).fillna(0.0)
Wb = W[["btc"]]
pxb = px[["btc"]]
for seed in (20261014, 1, 2):
    rng = np.random.default_rng(seed)
    s = {k: [] for k in PER}
    ex = {k: [] for k in PER}
    for _ in range(1000):
        pr = backtest(circular_shift(Wb, pxb, rng), pxb)
        for k, P in PER.items():
            m = pr.metrics(*P)
            s[k].append(m["sharpe"]); ex[k].append(m["exposure"])
    out[f"P1_circular_shift_seed{seed}"] = {
        k: {"pct_rank": round(100 * float(np.mean(np.array(s[k]) < real[k]["sharpe"])), 1),
            "placebo_median_sharpe": round(float(np.median(s[k])), 3),
            "placebo_p95_sharpe": round(float(np.quantile(s[k], 0.95)), 3),
            "placebo_exposure_median": round(float(np.median(ex[k])), 3),
            "placebo_exposure_p5_p95": [round(float(np.quantile(ex[k], q)), 3) for q in (0.05, 0.95)],
            "frac_placebo_beating_bh_sharpe": round(float(np.mean(np.array(s[k]) > real[k]["bh_sharpe"])), 3)}
        for k in PER}


# ---- P2 / P3: spell shuffle and within-period rotation, real-time size kept --------------------
def spells(x):
    x = np.asarray(x)
    cut = np.flatnonzero(np.diff(x) != 0) + 1
    segs = np.split(x, cut)
    return [(int(s[0]), len(s)) for s in segs]


def spell_shuffle(x, rng):
    sp = spells(x)
    ons = [l for v, l in sp if v == 1]
    offs = [l for v, l in sp if v == 0]
    rng.shuffle(ons); rng.shuffle(offs)
    if len(ons) > len(offs):
        start = 1
    elif len(offs) > len(ons):
        start = 0
    else:
        start = int(rng.integers(2))
    seq, a, b, cur = [], list(ons), list(offs), start
    while a or b:
        src = a if cur == 1 else b
        seq.append(np.full(src.pop(), cur, float))
        cur = 1 - cur
    return np.concatenate(seq)


rng = np.random.default_rng(11)
onv, sizev = on.to_numpy(), size.fillna(0.0).to_numpy()
for k, (a, b) in PER.items():
    ia, ib = idx.get_loc(pd.Timestamp(a)), idx.get_loc(pd.Timestamp(b))
    lo = ia - 2  # weights decided at ia-2 are held over day ia (lag 1)
    seg = onv[lo:ib + 1]
    n_sw = int((np.diff(seg) != 0).sum())
    res = {"P2": [], "P3": []}
    exps = {"P2": [], "P3": []}
    for _ in range(2000):
        for name in ("P2", "P3"):
            if name == "P2":
                s = spell_shuffle(seg, rng)
            else:
                s = np.roll(seg, int(rng.integers(30, len(seg) - 30)))
            o = onv.copy(); o[lo:ib + 1] = s
            w = pd.Series(o * sizev, idx)
            r, e = run(btc, w)
            res[name].append(sharpe(r.loc[a:b])); exps[name].append(float(e.loc[a:b].mean()))
    for name in ("P2", "P3"):
        arr = np.array(res[name])
        out.setdefault(f"{name}_{'spell_shuffle' if name == 'P2' else 'within_period_rotation'}_real_size", {})[k] = {
            "n_switches_real": n_sw, "on_frac_real": round(float(seg.mean()), 3),
            "pct_rank": round(100 * float(np.mean(arr < real[k]["sharpe"])), 1),
            "placebo_median_sharpe": round(float(np.median(arr)), 3),
            "placebo_p95_sharpe": round(float(np.quantile(arr, 0.95)), 3),
            "placebo_exposure_median": round(float(np.median(exps[name])), 3),
            "frac_placebo_beating_bh_sharpe": round(float(np.mean(arr > real[k]["bh_sharpe"])), 3)}


# ---- P4: stationary block bootstrap of returns (destroys trend persistence beyond ~L days) ---------
def stat_boot_idx(n, L, rng):
    out_i = np.empty(n, int)
    i = rng.integers(n)
    for t in range(n):
        if t == 0 or rng.random() < 1.0 / L:
            i = rng.integers(n)
        else:
            i = (i + 1) % n
        out_i[t] = i
    return out_i


lr = np.log(btc).diff()
for k, (a, b) in PER.items():
    ia, ib = idx.get_loc(pd.Timestamp(a)), idx.get_loc(pd.Timestamp(b))
    if np.isnan(btc.iloc[ib]):
        ib -= 1  # last panel day has no BTC price
    warm = btc.iloc[ia - 250:ia]
    pr = lr.iloc[ia:ib + 1].to_numpy()
    real_ds = real[k]["sharpe"] - real[k]["bh_sharpe"]
    real_dd = real[k]["mdd"] - real[k]["bh_mdd"]
    for L in (5, 20, 60):
        rng = np.random.default_rng(100 + L)
        ds, dd, ss = [], [], []
        for _ in range(1000):
            j = stat_boot_idx(len(pr), L, rng)
            path = warm.iloc[-1] * np.exp(np.cumsum(pr[j]))
            p = pd.concat([warm, pd.Series(path, index=idx[ia:ib + 1])])
            o, sz = sig_size(p)
            w = (o * sz).fillna(0.0)
            r, _ = run(p, w)
            br, _ = run(p, pd.Series(1.0, p.index))
            rs, brs = r.iloc[250:], br.iloc[250:]
            ds.append(sharpe(rs) - sharpe(brs)); dd.append(mdd(rs) - mdd(brs)); ss.append(sharpe(rs))
        ds, dd = np.array(ds), np.array(dd)
        out.setdefault("P4_block_bootstrap_paths", {}).setdefault(k, {})[f"L{L}"] = {
            "real_sharpe_diff": round(real_ds, 3), "pct_rank_sharpe_diff": round(100 * float(np.mean(ds < real_ds)), 1),
            "null_median_sharpe_diff": round(float(np.median(ds)), 3), "null_p95_sharpe_diff": round(float(np.quantile(ds, 0.95)), 3),
            "frac_null_sharpe_diff_gt0": round(float(np.mean(ds > 0)), 3),
            "real_mdd_diff_pp": round(100 * real_dd, 1), "null_median_mdd_diff_pp": round(100 * float(np.median(dd)), 1),
            "pct_rank_mdd_diff": round(100 * float(np.mean(dd < real_dd)), 1)}

# ---- P5: paired bootstrap CI of realised Sharpe differences ------------------------------------
for k, (a, b) in PER.items():
    s = r_real.loc[a:b].to_numpy(); bh = bh_r.loc[a:b].to_numpy()
    expo = real[k]["exposure"]
    cw, _ = run(btc, pd.Series(np.where(btc.notna(), expo, 0.0), idx))
    cwv = cw.loc[a:b].to_numpy()
    rng = np.random.default_rng(5)
    d1, d2 = [], []
    for _ in range(2000):
        j = stat_boot_idx(len(s), 20, rng)
        sh = lambda v: v.mean() / v.std(ddof=1) * np.sqrt(ANN)
        d1.append(sh(s[j]) - sh(bh[j])); d2.append(sh(s[j]) - sh(cwv[j]))
    d1, d2 = np.array(d1), np.array(d2)
    out.setdefault("P5_paired_bootstrap_sharpe_diff_L20", {})[k] = {
        "vs_bh_point": round(real[k]["sharpe"] - real[k]["bh_sharpe"], 3),
        "vs_bh_ci90": [round(float(np.quantile(d1, q)), 3) for q in (0.05, 0.95)],
        "vs_bh_p_le0": round(float(np.mean(d1 <= 0)), 3),
        "vs_constweight_point": round(real[k]["sharpe"] - sh(cwv), 3),
        "vs_constweight_ci90": [round(float(np.quantile(d2, q)), 3) for q in (0.05, 0.95)],
        "vs_constweight_p_le0": round(float(np.mean(d2 <= 0)), 3)}

print(json.dumps(out, indent=1))
json.dump(out, open("/home/user/11/research/btc_trend/review_repro_lookahead/placebo.json", "w"), indent=1)
