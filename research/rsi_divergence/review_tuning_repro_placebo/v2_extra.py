"""Session-2 extras: (1) config 441 (the dominant pick) traded as a fixed config, timing placebo
for 2017-2026 and for its selection years 2014-2016; (2) expanding/avg-R walk-forward with other
minimum-trade thresholds, from the cache; (3) per-year placebo excluding 2017 and 2018 both
(2018 was the worst year)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
sys.path.insert(0, str(Path(__file__).parent))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

OUT = Path(__file__).parent
rng = np.random.default_rng(777)
cache = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
cfgs, res, YEARS = cache["configs"], cache["results"], cache["years"]
TEST = list(range(2017, 2027))
df = load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
yrs = df.index.year.to_numpy()


def sim_long(bars, risk_pct, rr, mh):
    fill = np.asarray(bars) + 1
    entry = o[fill]
    risk = entry * risk_pct
    stop, target = entry - risk, entry + rr * risk
    shape = np.broadcast(fill, risk).shape
    px = np.full(shape, np.nan)
    done = np.zeros(shape, bool)
    last = np.minimum(fill + mh - 1, n - 1)
    for k in range(mh):
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
            "pct_null_below": round(100 * float((null < real).mean()), 1),
            "p_one_sided": round(float((1 + (null >= real).sum()) / (1 + len(null))), 4)}


out = {}
# (1) config 441 fixed
cfg = cfgs[441]
setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                         buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"], trend=cfg["trend"],
                         extreme=cfg["extreme"], sides=(1,))
sb = {s.bar: s for s in setups}
trs = simulate(df, setups, COST)
t = pd.DataFrame([{"bar": x.bar, "year": int(yrs[x.bar]), "r": x.r, "risk_pct": (x.entry - sb[x.bar].stop) / x.entry} for x in trs])
D = 10000
for label, ys in (("2017_2026", TEST), ("2018_2026", TEST[1:]), ("2014_2016_selection_years", [2014, 2015, 2016])):
    sub = t[t.year.isin(ys)]
    null = np.zeros(D)
    for y in ys:
        sy = sub[sub.year == y]
        if len(sy) == 0:
            continue
        cand = np.flatnonzero((yrs == y) & (np.arange(n) <= n - 2))
        bars = np.stack([rng.choice(cand, size=len(sy), replace=False) for _ in range(D)])
        null += sim_long(bars, sy.risk_pct.to_numpy()[None, :], 4.0, 60).sum(1)
    null /= len(sub)
    out[f"config441_{label}"] = {"n": int(len(sub)), **rank(sub.r.mean(), null)}
    print(label, out[f"config441_{label}"], flush=True)

# (2) min-trade thresholds for expanding avg-R, from the cache
S = np.array([[sum(r[y]) for y in YEARS] for r in res])
C = np.array([[len(r[y]) for y in YEARS] for r in res])
out["min_trades_sensitivity"] = {}
for mt in (10, 20, 30, 40, 50, 60):
    allr, ex = [], []
    picks = {}
    for y in TEST:
        cols = [j for j, x in enumerate(YEARS) if x < y]
        s, cnt = S[:, cols].sum(1), C[:, cols].sum(1)
        with np.errstate(all="ignore"):
            m = np.where(cnt >= mt, np.round(s / cnt, 3), -np.inf)
        p = int(np.argmax(m))
        picks[y] = p
        allr += res[p][y]
        if y != 2017:
            ex += res[p][y]
    out["min_trades_sensitivity"][mt] = {"n": len(allr), "avg_r": round(float(np.mean(allr)), 3), "n_ex2017": len(ex),
                                         "avg_r_ex2017": round(float(np.mean(ex)), 3), "picks": picks}
    print("min_trades", mt, out["min_trades_sensitivity"][mt], flush=True)
(OUT / "v2_extra.json").write_text(json.dumps(out, indent=1))
