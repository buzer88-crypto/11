"""No-lookahead check: weights built from a panel truncated at a cutoff date must equal the
full-panel weights on every rebalance row up to a week before the cutoff, for every grid config.

Run:  cd /home/user/11 && python3 research/onchain/check_lookahead.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import run  # noqa: E402
from pinetest.crypto_data import load_panel  # noqa: E402

panel = load_panel()
full = run.Data(panel)
bad, n = 0, 0
for cut in (pd.Timestamp("2019-06-15"), pd.Timestamp("2022-06-15")):
    trunc = run.Data({k: v.loc[:cut] for k, v in panel.items()})
    for i, (hyp, p, build) in enumerate(run.grid()):
        wf, _ = build(full)
        wt, _ = build(trunc)
        idx = wt.index[wt.index <= cut - pd.Timedelta(days=8)]
        a, c = wf.loc[idx], wt.loc[idx]
        rows = a.notna().any(axis=1) & c.notna().any(axis=1)
        diff = (a[rows].fillna(0) - c[rows].fillna(0)).abs().to_numpy().max()
        n += 1
        if diff > 1e-9:
            bad += 1
            print("MISMATCH", cut.date(), i, hyp, p, diff)
print(f"checked {n} (config, cutoff) pairs, mismatches: {bad}")
