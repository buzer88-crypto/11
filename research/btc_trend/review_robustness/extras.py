"""Extras: 2022 anatomy, static-43% by year, bootstrap of Sharpe diff in sub-periods."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import BH, PRE, STRAT, bt, static_w, stats  # noqa

END = "2026-05-23"
s, b = STRAT.returns, BH.returns
st = bt(static_w(0.43)).returns
print("year  strat  bh  static43   (total %, maxDD %)")
for y in range(2014, 2027):
    a, z = f"{y}-01-01", min(f"{y}-12-31", END)
    print(y, stats(s.loc[a:z])["total_pct"], stats(b.loc[a:z])["total_pct"], stats(st.loc[a:z])["total_pct"],
          "| dd", stats(s.loc[a:z])["max_dd_pct"], stats(b.loc[a:z])["max_dd_pct"], stats(st.loc[a:z])["max_dd_pct"])

# 2022 anatomy: entries/exits and P&L by holding spell
h = STRAT.held["btc"].loc["2021-11-01":"2022-12-31"]
on = (h > 0).astype(int)
spell = (on.diff().fillna(0) != 0).cumsum()
print("\n2022 holding spells (held>0):")
for k, g in h.groupby(spell):
    if g.iloc[0] > 0:
        r = s.loc[g.index]
        print(g.index[0].date(), g.index[-1].date(), len(g), "days, mean w %.2f" % g.mean(), "ret %.1f%%" % (100 * ((1 + r).prod() - 1)))
print("turnover/yr 2022:", round(STRAT.turnover.loc["2022"].sum(), 2), " cost drag 2022 (10bps):", round(STRAT.turnover.loc["2022"].sum() * 10 / 1e4 * 100, 2), "%")


def boot(x, y, B=2000, seed=11, L=30):
    rng = np.random.default_rng(seed)
    n = len(x)
    sh = lambda v: v.mean() / v.std() * np.sqrt(365)
    out = []
    for _ in range(B):
        idx = np.empty(n, dtype=int); i = 0
        while i < n:
            j = rng.integers(n); k = min(rng.geometric(1 / L), n - i)
            idx[i:i + k] = (j + np.arange(k)) % n; i += k
        out.append(sh(x[idx]) - sh(y[idx]))
    out = np.array(out)
    return round(float(sh(x) - sh(y)), 3), round(float((out <= 0).mean()), 3), [round(float(q), 3) for q in np.quantile(out, [.05, .95])]

print("\npaired stationary bootstrap (mean block 30d) Sharpe diff strat - B&H: point, p(diff<=0), 90% CI")
for lab, (a, z) in {"TEST": ("2022-01-01", END), "TEST ex2022": ("2023-01-01", END), "PRE": PRE,
                    "TRAIN": ("2018-01-01", "2021-12-31")}.items():
    print(lab, boot(s.loc[a:z].to_numpy(), b.loc[a:z].to_numpy()))
