"""Session-2 check 4 continued: timing placebo on the detrended (per-year drift removed) series.
(i)  the real raw-signal trades (same signal bars, same risk % of entry, rr 4, hold 60) with
     outcomes measured on detrended prices, vs random bars of the same year on detrended prices;
(ii) the picks' signals regenerated on the detrended series, vs the same detrended placebo."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
sys.path.insert(0, str(HERE))
from run import COST, load  # noqa: E402
from v2_detrend import PICKS, TEST, detrend  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

rng = np.random.default_rng(31337)
cfgs = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())["configs"]
dd = detrend(load("4h"))
o, h, l, c = (dd[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
yrs = dd.index.year.to_numpy()
RR, MH = 4.0, 60


def sim_long(bars, risk_pct):
    fill = np.asarray(bars) + 1
    entry = o[fill]
    risk = entry * risk_pct
    stop, target = entry - risk, entry + RR * risk
    shape = np.broadcast(fill, risk).shape
    px = np.full(shape, np.nan)
    done = np.zeros(shape, bool)
    last = np.minimum(fill + MH - 1, n - 1)
    for k in range(MH):
        j = fill + k
        live = ~done & (j <= last)
        jj = np.minimum(j, n - 1)
        for cond, val in ((o[jj] <= stop, o[jj]), (o[jj] >= target, o[jj]), (l[jj] <= stop, stop), (h[jj] >= target, target)):
            hit = live & cond
            px = np.where(hit, val, px)
            done |= hit
            live &= ~hit
    px = np.where(done, px, c[last])
    return (px - entry - COST / 100 * (entry + px)) / risk


def rank(real, null):
    null = np.asarray(null)
    return {"real": round(float(real), 3), "null_mean": round(float(null.mean()), 3),
            "null_p05": round(float(np.quantile(null, .05)), 3), "null_p95": round(float(np.quantile(null, .95)), 3),
            "pct_null_below": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4)}


def placebo(t, D=20000):
    """t: DataFrame with year, risk_pct, r. Returns pooled ranks with/without 2017."""
    R = np.zeros((D, len(t)))
    for y in sorted(t.year.unique()):
        ii = np.flatnonzero(t.year.to_numpy() == y)
        cand = np.flatnonzero((yrs == y) & (np.arange(n) <= n - 2))
        bars = np.stack([rng.choice(cand, size=len(ii), replace=False) for _ in range(D)])
        R[:, ii] = sim_long(bars, t.risk_pct.to_numpy()[ii][None, :])
    ex = (t.year != 2017).to_numpy()
    return {"all": {"n": int(len(t)), **rank(t.r.mean(), R.mean(1))},
            "ex2017": {"n": int(ex.sum()), **rank(t.r[ex].mean(), R[:, ex].mean(1))}}


out = {}
# (i) raw signals, detrended outcomes
raw_tr = pd.read_csv(HERE / "v2_trades.csv")
raw_tr["r_detr"] = sim_long(raw_tr.bar.to_numpy(), raw_tr.risk_pct.to_numpy())
t1 = pd.DataFrame({"year": raw_tr.year, "risk_pct": raw_tr.risk_pct, "r": raw_tr.r_detr})
out["i_raw_signals_detrended_outcomes"] = placebo(t1)
out["i_raw_signals_detrended_outcomes"]["per_year_avg_r"] = t1.groupby("year").r.mean().round(3).to_dict()
print("(i)", json.dumps(out["i_raw_signals_detrended_outcomes"]), flush=True)

# (ii) signals regenerated on the detrended series
rows = []
memo = {}
for y in TEST:
    i = PICKS[y]
    if i not in memo:
        cfg = cfgs[i]
        s = rsi_divergences(dd, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                            buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"],
                            trend=cfg["trend"], extreme=cfg["extreme"], sides=(1,))
        sb = {x.bar: x for x in s}
        memo[i] = [(x, sb[x.bar]) for x in simulate(dd, s, COST)]
    for x, s in memo[i]:
        if yrs[x.bar] == y:
            rows.append({"year": y, "bar": x.bar, "r": x.r, "risk_pct": (x.entry - s.stop) / x.entry})
t2 = pd.DataFrame(rows)
t2.to_csv(HERE / "v2_trades_detrended_picks.csv", index=False)
out["ii_detrended_signals"] = placebo(t2)
print("(ii)", json.dumps(out["ii_detrended_signals"]), flush=True)
(HERE / "v2_detrend_placebo.json").write_text(json.dumps(out, indent=1, default=str))
