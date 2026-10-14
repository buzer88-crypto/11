"""Session-2 checks 2-4b on the expanding/avg-R walk-forward picks.

2. Regenerate each year's pick trades (rsi_divergences + simulate), verify against the cache.
   Timing placebo per year: the same number of long trades at random 4h signal bars of that
   year (entry next open), each with a real trade's risk as % of entry, the pick's rr and
   max_hold, same costs and exit rules.
   P1 'uniform'     independent draws (trades may overlap), 20000 draws
   P2 'nonoverlap'  trades placed one by one, never overlapping (like one-at-a-time), 4000 draws
   P3 'filtered'    P1 restricted to bars where the pick's non-divergence filters hold
3. Exposure: BTC held over exactly the in-trade bars, and over the same number of bars at
   random places of the same year (and anywhere in 2017-2026), sized like the trade.
4b. Split by the BTC move over the 60 bars after entry (rise/fall).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import Setup, simulate  # noqa: E402
from pinetest.indicators import ema, macd  # noqa: E402

OUT = Path(__file__).parent
TEST = list(range(2017, 2027))
PICKS = {2017: 680, 2018: 441, 2019: 441, 2020: 441, 2021: 441, 2022: 441, 2023: 441, 2024: 194, 2025: 1566, 2026: 441}
rng = np.random.default_rng(424242)

cache = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
cfgs = cache["configs"]
df = load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
yrs = df.index.year.to_numpy()

# ---------------------------------------------------------------- regenerate real trades
rows, verify = [], {}
for i in sorted(set(PICKS.values())):
    cfg = cfgs[i]
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                             buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"],
                             trend=cfg["trend"], extreme=cfg["extreme"],
                             sides=(1, -1) if cfg["sides"] == "both" else (1,))
    sb = {s.bar: s for s in setups}
    trades = simulate(df, setups, COST)
    for y in [y for y in TEST if PICKS[y] == i]:
        ty = [t for t in trades if yrs[t.bar] == y]
        mine = np.array([round(t.r, 4) for t in ty])
        ref = np.array(cache["results"][i][y])
        verify[y] = {"pick": i, "n": len(ty), "n_cache": len(ref),
                     "max_abs_diff": float(np.abs(mine - ref).max()) if len(ref) and len(mine) == len(ref) else None}
        for t in ty:
            s = sb[t.bar]
            rows.append({"year": y, "pick": i, "bar": t.bar, "date": str(df.index[t.bar]), "side": t.side,
                         "fill_bar": t.fill_bar, "entry": t.entry, "stop": s.stop, "target": s.target,
                         "exit_bar": t.exit_bar, "exit": t.exit, "reason": t.reason, "r": t.r,
                         "rr": cfg["rr"], "max_hold": cfg["max_hold"]})
tr = pd.DataFrame(rows).sort_values("bar").reset_index(drop=True)
tr["risk_pct"] = (tr.entry - tr.stop) / tr.entry
tr["bars_held"] = tr.exit_bar - tr.fill_bar + 1
tr.to_csv(OUT / "v2_trades.csv", index=False)
print("verify", verify, flush=True)
assert (tr.side == 1).all() and (tr.rr == 4).all() and (tr.max_hold == 60).all()
RR, MH = 4.0, 60
ex = (tr.year != 2017).to_numpy()
real_all, real_ex = float(tr.r.mean()), float(tr.r[ex].mean())
print("real", len(tr), round(real_all, 4), "ex2017", int(ex.sum()), round(real_ex, 4), flush=True)
print(tr.groupby("year").agg(n=("r", "size"), avg_r=("r", "mean"), risk_pct_med=("risk_pct", "median"),
                             held=("bars_held", "mean"), tgt=("reason", lambda s: (s == "target").mean())).round(3), flush=True)


# ---------------------------------------------------------------- my long-trade simulator
def sim_long(bars, risk_pct):
    """Independent long market trades, entry o[bar+1], same exit logic as events._exit.
    bars and risk_pct broadcast to the same shape. Returns R, exit_bar."""
    bars = np.asarray(bars)
    fill = bars + 1
    entry = o[fill]
    risk = entry * risk_pct
    stop, target = entry - risk, entry + RR * risk
    shape = np.broadcast(fill, risk).shape
    px = np.full(shape, np.nan)
    eb = np.full(shape, -1)
    done = np.zeros(shape, bool)
    last = np.minimum(fill + MH - 1, n - 1)
    for k in range(MH):
        j = fill + k
        live = ~done & (j <= last)
        jj = np.minimum(j, n - 1)
        for cond, val in ((o[jj] <= stop, o[jj]), (o[jj] >= target, o[jj]), (l[jj] <= stop, stop), (h[jj] >= target, target)):
            hit = live & cond
            px = np.where(hit, val, px)
            eb = np.where(hit, jj, eb)
            done |= hit
            live &= ~hit
    px = np.where(done, px, c[last])
    eb = np.where(done, eb, last)
    return (px - entry - COST / 100 * (entry + px)) / risk, eb


# verify my simulator: (a) the real trades at their real bars, (b) 600 random trades vs events.simulate
r_chk, eb_chk = sim_long(tr.bar.to_numpy(), tr.risk_pct.to_numpy())
sim_check = {"real_trades_max_abs_dR": float(np.abs(r_chk - tr.r).max()), "real_exit_bar_mismatch": int((eb_chk != tr.exit_bar).sum())}
rb = rng.integers(np.flatnonzero(yrs >= 2017)[0], n - 2, 600)
rp = rng.uniform(0.01, 0.12, 600)
mine_r, _ = sim_long(rb, rp)
ref_r = []
for b, p in zip(rb, rp):
    e = o[b + 1]
    t = simulate(df, [Setup(int(b), 1, "market", e * (1 - p), e * (1 + RR * p), max_hold=MH)], COST)
    ref_r.append(t[0].r)
sim_check["random_600_max_abs_dR_vs_events_simulate"] = float(np.abs(mine_r - np.array(ref_r)).max())
print("sim check", sim_check, flush=True)


def rank(real, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return {"real": round(float(real), 3), "null_mean": round(float(null.mean()), 3), "null_median": round(float(np.median(null)), 3),
            "null_p05": round(float(np.quantile(null, .05)), 3), "null_p95": round(float(np.quantile(null, .95)), 3),
            "pct_null_below": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4), "draws": int(len(null))}


out = {"verify_vs_cache": verify, "sim_check": sim_check,
       "real": {"all": {"n": len(tr), "avg_r": round(real_all, 3)}, "ex2017": {"n": int(ex.sum()), "avg_r": round(real_ex, 3)}}}

# ---------------------------------------------------------------- filters for P3
e9, e21, e200 = (ema(df["close"], k).to_numpy() for k in (9, 21, 200))
ml, ms, _ = (x.to_numpy() for x in macd(df["close"]))


def filt(cfg):
    m = np.ones(n, bool)
    if cfg["confirm"] == "ema_9_21":
        m &= e9 > e21
    if cfg["confirm"] == "macd_cross":
        m &= ml > ms
    if cfg["confirm"] == "ema50":
        raise NotImplementedError
    if cfg["trend"]:
        m &= c > e200
    return m


# ---------------------------------------------------------------- P1 / P3: independent draws
D = 20000
idx_year = {y: np.flatnonzero(tr.year.to_numpy() == y) for y in TEST}


def placebo_indep(restrict):
    R = np.full((D, len(tr)), np.nan)
    F60 = np.full((D, len(tr)), np.nan)
    per = {}
    for y in TEST:
        ii = idx_year[y]
        k = len(ii)
        if k == 0:
            continue
        cand = np.flatnonzero((yrs == y) & (np.arange(n) <= n - 2))
        if restrict:
            cand = cand[filt(cfgs[PICKS[y]])[cand]]
        bars = np.stack([rng.choice(cand, size=k, replace=False) for _ in range(D)])
        rp = np.broadcast_to(tr.risk_pct.to_numpy()[ii], (D, k))
        r, _ = sim_long(bars, rp)
        R[:, ii] = r
        f = bars + 1
        F60[:, ii] = c[np.minimum(f + MH - 1, n - 1)] / o[f] - 1
        per[y] = {"n": k, "cand_bars": int(len(cand)), **rank(tr.r.to_numpy()[ii].mean(), r.mean(1))}
    return {"pooled_all": rank(real_all, R.mean(1)), "pooled_ex2017": rank(real_ex, R[:, ex].mean(1)), "per_year": per}, R, F60


res_u, R_u, F_u = placebo_indep(False)
out["P1_uniform"] = res_u
print("P1", res_u["pooled_all"], res_u["pooled_ex2017"], flush=True)
res_f, _, _ = placebo_indep(True)
out["P3_filtered"] = res_f
print("P3", res_f["pooled_all"], res_f["pooled_ex2017"], flush=True)
for y in TEST:
    if y in res_u["per_year"]:
        print(y, "uniform", res_u["per_year"][y]["real"], res_u["per_year"][y]["null_mean"], res_u["per_year"][y]["pct_null_below"],
              "| filtered", res_f["per_year"][y]["null_mean"], res_f["per_year"][y]["pct_null_below"], flush=True)

# ---------------------------------------------------------------- P2: non-overlapping draws
# Precompute for each year the outcome of every candidate bar for each real trade's risk.
D2 = 4000
pre = {}
for y in TEST:
    ii = idx_year[y]
    if len(ii) == 0:
        continue
    cand = np.flatnonzero((yrs == y) & (np.arange(n) <= n - 2))
    rr_, eb_ = sim_long(cand[:, None], tr.risk_pct.to_numpy()[ii][None, :])
    pre[y] = (cand, rr_, eb_)
R2 = np.full((D2, len(tr)), np.nan)
for d in range(D2):
    for y, (cand, rr_, eb_) in pre.items():
        ii = idx_year[y]
        busy = []          # (signal bar, exit bar) intervals already placed
        for col in rng.permutation(len(ii)):
            for _ in range(1000):
                p = int(rng.integers(len(cand)))
                b, e = cand[p], eb_[p, col]
                if all(e < b0 or b > e0 for b0, e0 in busy):
                    break
            busy.append((b, e))
            R2[d, ii[col]] = rr_[p, col]
out["P2_nonoverlap"] = {"pooled_all": rank(real_all, np.nanmean(R2, 1)), "pooled_ex2017": rank(real_ex, np.nanmean(R2[:, ex], 1))}
print("P2", out["P2_nonoverlap"], flush=True)

# ---------------------------------------------------------------- 3. exposure benchmarks
fill, eb, rp = tr.fill_bar.to_numpy(), tr.exit_bar.to_numpy(), tr.risk_pct.to_numpy()
entry = o[fill]
dur = eb - fill + 1
hold_R = (c[eb] - entry - COST / 100 * (entry + c[eb])) / (rp * entry)
hold_ret = c[eb] / entry - 1
strat_ret = tr.exit.to_numpy() / entry - 1          # unlevered move captured by the trade (before cost)
DR = 20000
RH_y = np.empty((DR, len(tr)))
RH_any = np.empty((DR, len(tr)))
oos_bars = np.flatnonzero(yrs >= 2017)
for j in range(len(tr)):
    d_ = dur[j]
    cy = np.flatnonzero((yrs == tr.year[j]) & (np.arange(n) + d_ - 1 <= n - 1))
    ca = oos_bars[oos_bars + d_ - 1 <= n - 1]
    for arr, cc in ((RH_y, cy), (RH_any, ca)):
        s = rng.choice(cc, DR)
        arr[:, j] = (c[s + d_ - 1] - o[s] - COST / 100 * (o[s] + c[s + d_ - 1])) / (rp[j] * o[s])
mu = pd.Series(np.r_[0.0, np.diff(np.log(c))]).groupby(yrs).mean()
drift_R = (np.exp(mu.loc[tr.year].to_numpy() * dur) - 1) / rp
eq = lambda a: round(100 * (float(np.prod(1 + 0.01 * np.asarray(a))) - 1), 1)
out["exposure"] = {
    "bars_in_trade": int(dur.sum()), "pct_of_2017_2026_bars": round(100 * dur.sum() / len(oos_bars), 2),
    "strategy": {"all": round(real_all, 3), "ex2017": round(real_ex, 3), "eq_pct_1pct": eq(tr.r), "eq_pct_1pct_ex2017": eq(tr.r[ex])},
    "btc_same_bars_R": {"all": round(float(hold_R.mean()), 3), "ex2017": round(float(hold_R[ex].mean()), 3),
                        "eq_pct_1pct": eq(hold_R), "eq_pct_1pct_ex2017": eq(hold_R[ex])},
    "unlevered_compound_pct": {"btc_same_bars": round(100 * (float(np.prod(1 + hold_ret)) - 1), 1),
                               "strategy_gross": round(100 * (float(np.prod(1 + strat_ret)) - 1), 1),
                               "btc_same_bars_ex2017": round(100 * (float(np.prod(1 + hold_ret[ex])) - 1), 1),
                               "strategy_gross_ex2017": round(100 * (float(np.prod(1 + strat_ret[ex])) - 1), 1)},
    "btc_random_same_len_same_year": {"all": rank(real_all, RH_y.mean(1)), "ex2017": rank(real_ex, RH_y[:, ex].mean(1))},
    "btc_random_same_len_any_2017_2026": {"all": rank(real_all, RH_any.mean(1)), "ex2017": rank(real_ex, RH_any[:, ex].mean(1))},
    "drift_implied_R": {"all": round(float(drift_R.mean()), 3), "ex2017": round(float(drift_R[ex].mean()), 3)},
    "per_year": {y: {"n": int(len(idx_year[y])), "strategy": round(float(tr.r.to_numpy()[idx_year[y]].mean()), 3),
                     "btc_same_bars": round(float(hold_R[idx_year[y]].mean()), 3),
                     "btc_random_same_len_mean": round(float(RH_y[:, idx_year[y]].mean()), 3),
                     "drift_implied": round(float(drift_R[idx_year[y]].mean()), 3)} for y in TEST if len(idx_year[y])},
}
print(json.dumps(out["exposure"], indent=0), flush=True)

# ---------------------------------------------------------------- 4b. split by next-60-bar BTC move
last60 = np.minimum(fill + MH - 1, n - 1)
up = c[last60] / entry - 1 > 0
split = {}
for lab, m in (("all", np.ones(len(tr), bool)), ("ex2017", ex)):
    Rm, Fm = R_u[:, m], F_u[:, m]
    with np.errstate(all="ignore"):
        pu = np.nanmean(np.where(Fm > 0, Rm, np.nan), 1)
        pdn = np.nanmean(np.where(Fm <= 0, Rm, np.nan), 1)
    w = float(up[m].mean())
    split[lab] = {"real_up": {"n": int((up & m).sum()), "avg_r": round(float(tr.r[up & m].mean()), 3)},
                  "real_down": {"n": int((~up & m).sum()), "avg_r": round(float(tr.r[~up & m].mean()), 3)},
                  "placebo_up_avg_r": rank(float(tr.r[up & m].mean()), pu),
                  "placebo_down_avg_r": rank(float(tr.r[~up & m].mean()), pdn),
                  "real_share_up": round(w, 3), "placebo_share_up": rank(w, (Fm > 0).mean(1)),
                  "placebo_reweighted_to_real_up_share": rank(float(tr.r[m].mean()), w * pu + (1 - w) * pdn)}
out["fwd60_split"] = split
print(json.dumps(split, indent=0), flush=True)
(OUT / "v2_placebo.json").write_text(json.dumps(out, indent=1, default=str))
