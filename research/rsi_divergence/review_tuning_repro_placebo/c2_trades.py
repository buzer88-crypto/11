"""Check 2a: regenerate the expanding/avg-R picks' trades and verify them against the cache.
Writes trades_expanding_avg_r.csv (one row per OOS trade)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lib import COST, TEST, YEARS, by_year, cache, load, sim_long_independent, trades_for

OUT = Path(__file__).parent
PICKS = {2017: 680, 2018: 441, 2019: 441, 2020: 441, 2021: 441, 2022: 441, 2023: 441, 2024: 194, 2025: 1566, 2026: 441}
d = cache()
df = load("4h")
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
rows, check = [], {}
memo = {}
for y in TEST:
    i = PICKS[y]
    if i not in memo:
        memo[i] = by_year(df, trades_for(df, d["configs"][i]))
    ts = memo[i][y]
    mine = [round(t.r, 4) for t in ts]
    check[y] = {"pick": i, "n_regen": len(mine), "n_cache": len(d["results"][i][y]),
                "max_abs_diff": float(np.max(np.abs(np.array(mine) - np.array(d["results"][i][y])))) if mine else 0.0}
    cfg = d["configs"][i]
    for t in ts:
        assert t.side == 1
        rows.append({"year": y, "pick": i, "bar": t.bar, "date": str(df.index[t.bar]), "fill_bar": t.fill_bar,
                     "entry": t.entry, "exit_bar": t.exit_bar, "exit": t.exit, "reason": t.reason, "r": t.r,
                     "rr": cfg["rr"], "max_hold": cfg["max_hold"]})
tr = pd.DataFrame(rows)
# cross-check via the setups directly
from pinetest.divergence import rsi_divergences
stops = {}
for i in set(PICKS.values()):
    cfg = d["configs"][i]
    for s in rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"], rr=cfg["rr"],
                             buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"], confirm=cfg["confirm"],
                             trend=cfg["trend"], extreme=cfg["extreme"], sides=(1,)):
        stops[(i, s.bar)] = s.stop
tr["stop"] = [stops[(p, b)] for p, b in zip(tr.pick, tr.bar)]
tr["risk"] = tr.entry - tr.stop
tr["risk_pct"] = tr.risk / tr.entry
tr["bars_held"] = tr.exit_bar - tr.fill_bar + 1
# sanity: my vectorised simulator reproduces each real trade's R when placed at its real bar
r2, eb2, _ = sim_long_independent(o, h, l, c, tr.bar.to_numpy(), tr.risk_pct.to_numpy(), tr.rr.to_numpy(), 60)
tr["r_vec"] = r2
print("vectorised sim max |dR| vs real:", float(np.max(np.abs(r2 - tr.r))), "exit bar mismatches:", int((eb2 != tr.exit_bar).sum()))
tr.to_csv(OUT / "trades_expanding_avg_r.csv", index=False)
print(json.dumps(check, indent=0))
print(tr.groupby("year").agg(n=("r", "size"), avg_r=("r", "mean"), med_risk_pct=("risk_pct", "median"),
                             bars=("bars_held", "mean"), tgt=("reason", lambda s: (s == "target").mean())).round(3))
print(tr.reason.value_counts())
print("overall", tr.r.mean(), "ex2017", tr[tr.year != 2017].r.mean())
(OUT / "c2_trades_check.json").write_text(json.dumps({"check": check, "vec_sim_max_abs_dR": float(np.max(np.abs(r2 - tr.r)))}, indent=1))
