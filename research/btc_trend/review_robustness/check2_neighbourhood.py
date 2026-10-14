"""Check 2: SMA 30..250 (step 10) x vol target 20..80% (+ no vol target), daily rebalance."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import BH, PRE, TEST, TRAIN, bt, static_w, trend_w  # noqa

END = "2026-05-23"
PER = {"train": TRAIN, "test": ("2022-01-01", END), "pre": PRE, "test_ex22": ("2023-01-01", END)}
NS = list(range(30, 260, 10))
VTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, None]
bh = {p: BH.metrics(*v) for p, v in PER.items()}
static_cache = {}


def static_dd(frac, per):
    key = (round(frac, 2), per)
    if key not in static_cache:
        static_cache[key] = bt(static_w(round(frac, 2))).metrics(*PER[per])
    return static_cache[key]


rows = []
for n in NS:
    for vt in VTS:
        r = bt(trend_w(n, vt))
        row = {"n": n, "vt": "full" if vt is None else int(vt * 100)}
        for p, per in PER.items():
            m = r.metrics(*per)
            row[f"{p}_sh"] = m["sharpe"]
            row[f"{p}_dd"] = m["max_dd_pct"]
            row[f"{p}_tot"] = m["total_pct"]
            row[f"{p}_exp"] = m["exposure"]
            if p in ("test", "pre"):
                st = static_dd(m["exposure"], p)
                row[f"{p}_static_dd"] = st["max_dd_pct"]
                row[f"{p}_static_sh"] = st["sharpe"]
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_name("neighbourhood.csv"), index=False)

out = {"bh": {p: {"sharpe": round(v["sharpe"], 3), "max_dd": round(v["max_dd_pct"], 1)} for p, v in bh.items()}}
for p in ("train", "test", "pre", "test_ex22"):
    piv = df.pivot(index="n", columns="vt", values=f"{p}_sh").round(2)
    piv = piv[[20, 30, 40, 50, 60, 70, 80, "full"]]
    print(f"\n=== {p} Sharpe (B&H {bh[p]['sharpe']:.2f}) ===")
    print(piv.to_string())
    out[f"{p}_beats_bh_share"] = round(float((df[f"{p}_sh"] > bh[p]["sharpe"]).mean()), 3)
    out[f"{p}_sharpe_quantiles"] = [round(float(x), 3) for x in df[f"{p}_sh"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])]
# rank of chosen config
ch = df[(df.n == 50) & (df.vt == 40)].iloc[0]
for p in ("train", "test", "pre"):
    out[f"chosen_{p}_rank"] = f"{int((df[f'{p}_sh'] > ch[f'{p}_sh']).sum()) + 1}/{len(df)}"
# local neighbourhood of chosen (n 30..70, vt 30..50)
nb = df[df.n.between(30, 70) & df.vt.isin([30, 40, 50])]
out["local_nbhd_test_sh"] = {"min": round(nb.test_sh.min(), 3), "median": round(nb.test_sh.median(), 3), "max": round(nb.test_sh.max(), 3)}
# train-best in bigger grid
tb = df.loc[df.train_sh.idxmax()]
out["train_best_big_grid"] = {k: (tb[k] if not isinstance(tb[k], (np.floating, float)) else round(float(tb[k]), 3))
                              for k in ("n", "vt", "train_sh", "test_sh", "pre_sh", "test_dd", "pre_dd")}
# by-n summary (averaged over vol targets) and by vt
out["test_sh_by_n_median_over_vt"] = df.groupby("n").test_sh.median().round(3).to_dict()
out["test_sh_by_vt_median_over_n"] = {str(k): v for k, v in df.groupby("vt").test_sh.median().round(3).to_dict().items()}
# drawdown: config vs static allocation at the same mean exposure
for p in ("test", "pre"):
    out[f"{p}_dd_better_than_bh_share"] = round(float((df[f"{p}_dd"] > bh[p]["max_dd_pct"]).mean()), 3)
    out[f"{p}_dd_better_than_static_same_exposure_share"] = round(float((df[f"{p}_dd"] > df[f"{p}_static_dd"]).mean()), 3)
    out[f"{p}_sharpe_better_than_static_same_exposure_share"] = round(float((df[f"{p}_sh"] > df[f"{p}_static_sh"]).mean()), 3)
    out[f"{p}_median_dd_minus_static_dd"] = round(float((df[f"{p}_dd"] - df[f"{p}_static_dd"]).median()), 2)
print("\n=== TEST max DD (B&H %.1f) ===" % bh["test"]["max_dd_pct"])
print(df.pivot(index="n", columns="vt", values="test_dd").round(0)[[20, 30, 40, 50, 60, 70, 80, "full"]].to_string())
print("\n=== TEST max DD minus static-at-same-exposure max DD (positive = strategy better) ===")
print((df.pivot(index="n", columns="vt", values="test_dd") - df.pivot(index="n", columns="vt", values="test_static_dd"))
      .round(0)[[20, 30, 40, 50, 60, 70, 80, "full"]].to_string())
Path(__file__).with_name("results_2.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
