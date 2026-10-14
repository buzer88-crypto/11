"""Session-2 extra placebo: random long entries with a structure-based stop like the strategy's.
Per year, the same number of trades as the pick at random bars of that year where the pick's
non-divergence filters hold; stop = lowest low of the last 2*lb+1 bars (the pick's pivot span)
minus buffer_atr x ATR; target 4R; max_hold 60; entry next open. Independent trades."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
from run import COST, load  # noqa: E402

from pinetest.indicators import ema, macd  # noqa: E402
from pinetest.smc import atr  # noqa: E402

rng = np.random.default_rng(8080)
cfgs = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())["configs"]
PICKS = {2017: 680, 2018: 441, 2019: 441, 2020: 441, 2021: 441, 2022: 441, 2023: 441, 2024: 194, 2025: 1566, 2026: 441}
df = load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
yrs = df.index.year.to_numpy()
a = atr(df)
e9, e21, e200 = (ema(df["close"], k).to_numpy() for k in (9, 21, 200))
ml, ms, _ = (x.to_numpy() for x in macd(df["close"]))
RR, MH = 4.0, 60
tr = pd.read_csv(HERE / "v2_trades.csv")


def sim_long(bars, stop):
    fill = np.asarray(bars) + 1
    entry = o[fill]
    risk = entry - stop
    target = entry + RR * risk
    px = np.full(fill.shape, np.nan)
    done = np.zeros(fill.shape, bool)
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


D = 20000
R = np.zeros((D, len(tr)))
per = {}
for y in sorted(tr.year.unique()):
    cfg = cfgs[PICKS[y]]
    ii = np.flatnonzero(tr.year.to_numpy() == y)
    span = 2 * cfg["lb"] + 1
    cand = np.flatnonzero((yrs == y) & (np.arange(n) <= n - 2) & (np.arange(n) >= span))
    m = np.ones(len(cand), bool)
    if cfg["confirm"] == "ema_9_21":
        m &= e9[cand] > e21[cand]
    if cfg["confirm"] == "macd_cross":
        m &= ml[cand] > ms[cand]
    if cfg["trend"]:
        m &= c[cand] > e200[cand]
    cand = cand[m]
    lowest = np.array([l[b - span + 1:b + 1].min() for b in cand]) - cfg["buffer_atr"] * a[cand]
    ok = o[cand + 1] - lowest > 0
    cand, lowest = cand[ok], lowest[ok]
    pick = np.stack([rng.choice(len(cand), size=len(ii), replace=False) for _ in range(D)])
    R[:, ii] = sim_long(cand[pick], lowest[pick])
    rp = (o[cand + 1] - lowest) / o[cand + 1]
    real = tr.r.to_numpy()[ii].mean()
    per[int(y)] = {"n": int(len(ii)), "real": round(float(real), 3), "null_mean": round(float(R[:, ii].mean()), 3),
                   "pct_null_below": round(100 * float((R[:, ii].mean(1) < real).mean()), 1),
                   "placebo_median_risk_pct": round(float(np.median(rp)), 4),
                   "real_median_risk_pct": round(float(np.median(tr.risk_pct.to_numpy()[ii])), 4)}
ex = (tr.year != 2017).to_numpy()


def rank(real, null):
    return {"real": round(float(real), 3), "null_mean": round(float(null.mean()), 3),
            "pct_null_below": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4)}


out = {"all": rank(tr.r.mean(), R.mean(1)), "ex2017": rank(tr.r[ex].mean(), R[:, ex].mean(1)), "per_year": per}
print(json.dumps(out, indent=1))
(HERE / "v2_swing_placebo.json").write_text(json.dumps(out, indent=1))
