"""Extra attribution checks: is the strategy better than simply holding less BTC (exposure- or
vol-matched constant weight)? Year-by-year vs those, and 2022 whipsaw detail."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.crypto_data import load_panel
from pinetest.portfolio import TEST, TRAIN, backtest, buy_and_hold

PRE = ("2014-01-01", "2017-12-31")
px = load_panel()["PriceUSD"]
p = px["btc"]
vol60 = p.pct_change(fill_method=None).rolling(60).std() * np.sqrt(365)


def frame(s):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w["btc"] = s.where(p.notna(), 0.0).fillna(0.0)
    return w


on = (p > p.rolling(50).mean()).astype(float)
w = frame(on * (0.40 / vol60).clip(upper=1.0))
res = backtest(w, px)
bh = backtest(buy_and_hold(px, "btc"), px)
out = {}
for name, per in (("pre", PRE), ("train", TRAIN), ("test", TEST)):
    sm, bm = res.metrics(*per), bh.metrics(*per)
    c_exp = round(sm["exposure"], 3)
    c_vol = round(sm["vol_pct"] / bm["vol_pct"], 3)
    row = {"strategy": {k: round(sm[k], 3) for k in ("sharpe", "total_pct", "max_dd_pct", "vol_pct")}}
    for lab, c in (("const_exposure_matched", c_exp), ("const_vol_matched", c_vol)):
        r = backtest(frame(pd.Series(c, index=p.index)), px).metrics(*per)
        row[lab] = {"weight": c} | {k: round(r[k], 3) for k in ("sharpe", "total_pct", "max_dd_pct", "vol_pct")}
    out[name] = row

# Year by year vs constant 0.43 (TEST mean exposure) and 0.40 (full-period mean exposure 0.39)
c43 = backtest(frame(pd.Series(0.43, index=p.index)), px)
yy = {}
for y in range(2014, 2027):
    s = res.returns.loc[str(y)]
    c = c43.returns.loc[str(y)]
    yy[str(y)] = {"strategy_pct": round(100 * ((1 + s).prod() - 1), 1),
                  "const43_pct": round(100 * ((1 + c).prod() - 1), 1),
                  "strategy_sharpe": round(float(s.mean() / s.std() * np.sqrt(365)), 2),
                  "const43_sharpe": round(float(c.mean() / c.std() * np.sqrt(365)), 2),
                  "mean_exposure": round(float(res.gross_exposure.loc[str(y)].mean()), 2)}
out["yearly_vs_const43"] = yy
out["years_strategy_return_beats_const43"] = sum(v["strategy_pct"] > v["const43_pct"] for v in yy.values())

# 2022 whipsaw detail: holding spells in 2022 and their compounded P&L
h = res.held["btc"].loc["2022"]
r = res.returns.loc["2022"]
inpos = (h > 1e-9).astype(int)
g = (inpos != inpos.shift()).cumsum()
spells = []
for _, idx in inpos.groupby(g).groups.items():
    if inpos.loc[idx[0]] == 1:
        spells.append({"start": str(idx[0].date()), "days": len(idx),
                       "pnl_pct": round(100 * ((1 + r.loc[idx]).prod() - 1), 2),
                       "avg_w": round(float(h.loc[idx].mean()), 2)})
out["2022_spells"] = spells
out["2022_n_spells"] = len(spells)
out["2022_n_losing_spells"] = sum(s["pnl_pct"] < 0 for s in spells)
out["2022_mean_exposure"] = round(float(res.gross_exposure.loc["2022"].mean()), 3)

# Short spells (<=5 days) contribution in TEST
ht = res.held["btc"].loc[TEST[0]:]
rt = res.returns.loc[TEST[0]:]
ip = (ht > 1e-9).astype(int)
gg = (ip != ip.shift()).cumsum()
short_pnl, long_pnl = 0.0, 0.0
for _, idx in ip.groupby(gg).groups.items():
    if ip.loc[idx[0]] == 1:
        lr = float(np.log1p(rt.loc[idx]).sum())
        if len(idx) <= 5:
            short_pnl += lr
        else:
            long_pnl += lr
out["test_logret_from_spells_le5d"] = round(short_pnl, 3)
out["test_logret_from_spells_gt5d"] = round(long_pnl, 3)

Path(__file__).with_name("extras_results.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
