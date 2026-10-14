"""Checks 2-4 on the real OOS trades of the expanding/avg-R walk-forward picks.

2. Timing placebo: per year, the same number of long trades at random 4h signal bars of that
   year (entry next open, as real), each with a real trade's risk as % of entry, rr and max_hold.
   Variant 'filtered': random bars restricted to bars where the pick's own non-divergence filters
   (EMA9>EMA21 / MACD>signal / close>EMA200) hold.
3. Exposure benchmarks: BTC held over exactly the strategy's in-trade bars (open of fill bar ->
   close of exit bar), and over the same number of bars placed at random in the same year; both
   sized like the trade (R = BTC move / trade risk, same costs).
4b. Split by the BTC move over the 60 bars after entry (up/down), real vs placebo within group.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lib import COST, TEST, cache, load, sim_long_independent
from pinetest.indicators import ema, macd

OUT = Path(__file__).parent
D = 20000
rng = np.random.default_rng(20261015)
df = load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
yrs = df.index.year.to_numpy()
tr = pd.read_csv(OUT / "trades_expanding_avg_r.csv")
cfgs = cache()["configs"]
assert (tr.rr == 4).all() and (tr.max_hold == 60).all()
real_all, real_ex = tr.r.mean(), tr[tr.year != 2017].r.mean()

# filters of each pick that do not depend on the divergence itself
e9, e21, e200 = (ema(df["close"], k).to_numpy() for k in (9, 21, 200))
line, sig, _ = (x.to_numpy() for x in macd(df["close"]))
def filt_mask(cfg):
    m = np.ones(n, bool)
    if cfg["confirm"] == "ema_9_21": m &= e9 > e21
    if cfg["confirm"] == "macd_cross": m &= line > sig
    if cfg["trend"]: m &= c > e200
    return m

def pct_rank(real, null):
    null = np.asarray(null)
    return {"real": round(float(real), 3), "null_mean": round(float(null.mean()), 3), "null_median": round(float(np.median(null)), 3),
            "null_p05": round(float(np.quantile(null, 0.05)), 3), "null_p95": round(float(np.quantile(null, 0.95)), 3),
            "pct_null_below_real": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4)}

out = {"D": D, "real": {"all": {"n": len(tr), "avg_r": round(real_all, 3)}, "ex2017": {"n": int((tr.year != 2017).sum()), "avg_r": round(real_ex, 3)}}}

# ---------------- 2. timing placebo ----------------
def placebo(restrict):
    R = np.empty((D, len(tr))); FWD = np.empty((D, len(tr)))
    col = 0
    per_year = {}
    for y in TEST:
        sub = tr[tr.year == y]
        k = len(sub)
        cand = np.flatnonzero((yrs == y) & (np.arange(n) + 1 <= n - 1))
        if restrict:
            cand = cand[filt_mask(cfgs[int(sub.pick.iloc[0])])[cand]]
        bars = np.stack([rng.choice(cand, size=k, replace=False) for _ in range(D)])
        rp = np.broadcast_to(sub.risk_pct.to_numpy(), (D, k))
        r, _, fill = sim_long_independent(o, h, l, c, bars, rp, 4.0, 60)
        R[:, col:col + k] = r
        FWD[:, col:col + k] = c[np.minimum(fill + 59, n - 1)] / o[fill] - 1
        per_year[y] = {"n": k, "n_candidate_bars": int(len(cand)), **pct_rank(sub.r.mean(), r.mean(1))}
        col += k
    ex = (tr.year != 2017).to_numpy()
    res = {"pooled_all": pct_rank(real_all, R.mean(1)), "pooled_ex2017": pct_rank(real_ex, R[:, ex].mean(1)),
           "per_year": per_year}
    return res, R, FWD

for name, restrict in (("uniform", False), ("filtered", True)):
    res, R, FWD = placebo(restrict)
    out[f"placebo_{name}"] = res
    if name == "uniform":
        R_u, FWD_u = R, FWD
    print(name, json.dumps({k: res[k] for k in ("pooled_all", "pooled_ex2017")}),
          {y: (v["real"], v["null_mean"], v["pct_null_below_real"]) for y, v in res["per_year"].items()}, flush=True)

# ---------------- 3. exposure benchmarks ----------------
fill, eb, rp = tr.fill_bar.to_numpy(), tr.exit_bar.to_numpy(), tr.risk_pct.to_numpy()
entry = o[fill]
risk = entry * rp
hold_same = (c[eb] - entry - COST / 100 * (entry + c[eb])) / risk
d_bars = eb - fill + 1
ex = (tr.year != 2017).to_numpy()
# random placement of the same holding lengths within the same year
RH = np.empty((D, len(tr)))
for idx, (y, d) in enumerate(zip(tr.year, d_bars)):
    cand = np.flatnonzero((yrs == y) & (np.arange(n) + d - 1 <= n - 1))
    s = rng.choice(cand, size=D)
    RH[:, idx] = (c[s + d - 1] - o[s] - COST / 100 * (o[s] + c[s + d - 1])) / (rp[idx] * o[s])
# full 60-bar hold from the real entry (no stop / target)
last60 = np.minimum(fill + 59, n - 1)
hold60 = (c[last60] - entry - COST / 100 * (entry + c[last60])) / risk
# drift-implied R: the year's mean 4h log return times bars held, in units of risk
mu = pd.Series(np.r_[0, np.diff(np.log(c))], index=df.index).groupby(yrs).mean()
drift_r = (np.exp(mu.loc[tr.year].to_numpy() * d_bars) - 1) / rp
eq = lambda a: round(100 * (np.prod(1 + 0.01 * np.asarray(a)) - 1), 1)
total_bars_test = int(((yrs >= 2017)).sum())
out["exposure"] = {
    "bars_in_trade": int(d_bars.sum()), "share_of_2017_2026_bars_pct": round(100 * d_bars.sum() / total_bars_test, 2),
    "strategy": {"all": round(real_all, 3), "ex2017": round(real_ex, 3), "eq_pct_1pct": eq(tr.r)},
    "btc_same_bars": {"all": round(float(hold_same.mean()), 3), "ex2017": round(float(hold_same[ex].mean()), 3), "eq_pct_1pct": eq(hold_same),
                      "unlevered_btc_pct_over_in_trade_bars": round(100 * (np.exp(np.log(c[eb] / entry).sum()) - 1), 1),
                      "unlevered_strategy_pct": round(100 * (np.prod(tr.exit / tr.entry * (1 - COST / 100) / (1 + COST / 100))) - 100, 1)},
    "btc_same_count_random_bars": {"all": pct_rank(real_all, RH.mean(1)), "ex2017": pct_rank(real_ex, RH[:, ex].mean(1))},
    "btc_60bar_hold_from_real_entry": {"all": round(float(hold60.mean()), 3), "ex2017": round(float(hold60[ex].mean()), 3)},
    "drift_implied_r": {"all": round(float(drift_r.mean()), 3), "ex2017": round(float(drift_r[ex].mean()), 3)},
    "per_year": {int(y): {"n": int((tr.year == y).sum()), "strategy": round(float(tr.r[tr.year == y].mean()), 3),
                          "btc_same_bars": round(float(hold_same[tr.year == y].mean()), 3),
                          "btc_random_same_len_mean": round(float(RH[:, (tr.year == y).to_numpy()].mean()), 3),
                          "drift_implied": round(float(drift_r[(tr.year == y).to_numpy()].mean()), 3)} for y in TEST},
}
print(json.dumps(out["exposure"], indent=0), flush=True)

# ---------------- 4b. split by the next 60 bars' BTC direction ----------------
fwd_real = c[last60] / entry - 1
up = fwd_real > 0
Ru, Fu = R_u.ravel(), FWD_u.ravel()
yr_rep = np.broadcast_to(tr.year.to_numpy(), R_u.shape).ravel()
def grp(mask_real, mask_pl):
    return {"real_n": int(mask_real.sum()), "real_avg_r": round(float(tr.r[mask_real].mean()), 3) if mask_real.any() else None,
            "placebo_avg_r": round(float(Ru[mask_pl].mean()), 3)}
split = {}
for lab, ex_ in (("all", np.ones(len(tr), bool)), ("ex2017", ex)):
    plmask = np.ones_like(Ru, bool) if lab == "all" else (yr_rep != 2017)
    split[lab] = {"up": grp(up & ex_, (Fu > 0) & plmask), "down": grp(~up & ex_, (Fu <= 0) & plmask),
                  "real_share_up": round(float(up[ex_].mean()), 3),
                  "placebo_share_up": round(float((Fu[plmask] > 0).mean()), 3)}
    # stratified: placebo mean re-weighted to the real up/down mix, per draw
    Rm = R_u[:, ex_]; Fm = FWD_u[:, ex_]
    with np.errstate(all="ignore"):
        pu = np.nanmean(np.where(Fm > 0, Rm, np.nan), 1); pd_ = np.nanmean(np.where(Fm <= 0, Rm, np.nan), 1)
    w = up[ex_].mean()
    strat = w * pu + (1 - w) * pd_
    split[lab]["placebo_reweighted_to_real_mix"] = pct_rank(tr.r[ex_].mean(), strat[np.isfinite(strat)])
    # timing skill as share of entries followed by a 60-bar rise
    share_null = (Fm > 0).mean(1)
    split[lab]["share_up_rank"] = pct_rank(up[ex_].mean(), share_null)
out["fwd60_split"] = split
print(json.dumps(split, indent=0))
(OUT / "c3_placebo.json").write_text(json.dumps(out, indent=1, default=str))
