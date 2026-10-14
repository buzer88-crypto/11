"""Verify trades_meta.pkl (from an earlier, interrupted run) without regenerating signals:
(1) per-config per-year R lists equal tune_cache.pkl exactly; (2) the vectorised market simulator
reproduces every trade's R and exit bar from (signal bar, side, risk %, rr, max_hold); (3) trades
never overlap within a config (one at a time); (4) fill = signal + 1.
PYTHONPATH=. python .../c0_verify.py"""
import json
import pickle

import numpy as np

from wf_lib import OUT, YEARS, load_cache, load_df, sim_market

cache = load_cache()
meta = pickle.loads((OUT / "trades_meta.pkl").read_bytes())
assert meta["configs"] == cache["configs"]
df = load_df("4h")
yrs = df.index.year.to_numpy()
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
mism = 0
overlap = 0
fillbad = 0
rows = []
for k, (tr, by) in enumerate(zip(meta["trades"], cache["results"])):
    got = {int(y): [] for y in YEARS}
    for row in tr:
        y = int(yrs[int(row[0])])
        if y in got:
            got[y].append(round(row[5], 4))
    if got != {int(y): list(v) for y, v in by.items()}:
        mism += 1
    if len(tr):
        overlap += int((tr[1:, 1] <= tr[:-1, 2]).sum())
        fillbad += int((tr[:, 1] != tr[:, 0] + 1).sum())
        cfg = meta["configs"][k]
        rows.append(np.column_stack([tr, np.full(len(tr), cfg["rr"]), np.full(len(tr), cfg["max_hold"])]))
A = np.vstack(rows)
eb, r = sim_market(o, h, l, c, A[:, 1].astype(int), A[:, 3], A[:, 4], A[:, 6], A[:, 7].astype(int))
d = np.abs(r - A[:, 5])
out = {"n_configs": len(meta["configs"]), "n_trades": int(len(A)), "configs_with_R_by_year_mismatch_vs_cache": mism,
       "overlapping_trades_within_config": overlap, "fill_not_signal_plus_1": fillbad,
       "sim_max_abs_R_diff": float(d.max()), "sim_n_R_diff_gt_1e-6": int((d > 1e-6).sum()),
       "sim_n_exit_bar_diff": int((eb != A[:, 2]).sum())}
print(json.dumps(out, indent=1))
(OUT / "c0_verify.json").write_text(json.dumps(out, indent=1))
