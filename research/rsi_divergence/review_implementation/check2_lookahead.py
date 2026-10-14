"""Check 2: lookahead. Recompute setups (all four types, lb 3 and 5, every filter) and the
chosen-config trades on data truncated at several cut points and compare with the full-data run.
Also confirm every trade fills at the open of the bar after its signal bar."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.divergence import rsi_divergences
from pinetest.events import simulate

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("study_run", HERE.parent / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
df = run.load("4h")
CUTS = ["2015-03-17 08:00", "2017-12-14 20:00", "2019-06-26 12:00", "2021-04-14 04:00",
        "2022-11-08 16:00", "2024-03-05 00:00", "2025-10-06 12:00", "2026-08-01 04:00"]
out = {"cuts": {}}


def key(s):
    return (s.bar, s.side, round(s.stop, 9), round(s.target, 9), s.entry, s.max_hold)


total_cmp = total_bad = 0
for cut in CUTS:
    k = df.index.searchsorted(pd.Timestamp(cut), side="right")      # bars [0, k)
    tr = df.iloc[:k]
    bad = cmp_ = 0
    for kind in ("regular", "hidden"):
        for lb in (3, 5):
            for filt in ("none", "extreme", "trend"):
                kw = dict(kind=kind, left=lb, right=lb, rr=3.0, extreme=filt == "extreme", trend=filt == "trend")
                full = {key(s) for s in rsi_divergences(df, **kw) if s.bar <= k - 2}
                part = {key(s) for s in rsi_divergences(tr, **kw)}
                cmp_ += len(full | part)
                bad += len(full ^ part)
    # chosen config trades: those finished before the cut must be identical
    kw = dict(kind="hidden", left=3, right=3, rr=3.0, trend=True)
    t_full = [t for t in simulate(df, rsi_divergences(df, **kw), 0.1) if t.exit_bar < k - 1]
    t_part = [t for t in simulate(tr, rsi_divergences(tr, **kw), 0.1) if t.exit_bar < k - 1]
    trade_bad = int(t_full != t_part)
    out["cuts"][cut] = {"bars_kept": int(k), "setups_compared": cmp_, "setup_differences": bad,
                        "trades_closed_before_cut": len(t_full), "trade_lists_differ": trade_bad}
    total_cmp += cmp_
    total_bad += bad + trade_bad
    print(cut, out["cuts"][cut])

# Entry timing on every trade of the chosen config.
setups = run.setups_for(df, "hidden", 3, 3.0, "trend")
trades = simulate(df, setups, 0.1)
o = df["open"].to_numpy()
out["entries_checked"] = len(trades)
out["entries_not_at_next_open"] = sum(not (t.fill_bar == t.bar + 1 and t.entry == o[t.bar + 1]) for t in trades)
# Signal bar = RSI pivot bar + right: the stop must sit beyond the pivot bar's low/high (bar - 3).
h, l = df["high"].to_numpy(), df["low"].to_numpy()
from pinetest.smc import atr  # noqa: E402
a = atr(df)
out["stops_not_at_pivot_extreme"] = sum(
    abs(s.stop - ((l[s.bar - 3] - 0.1 * a[s.bar]) if s.side > 0 else (h[s.bar - 3] + 0.1 * a[s.bar]))) > 1e-6
    for s in setups)
out["total_setup_or_trade_differences"] = total_bad
print({k: v for k, v in out.items() if k != "cuts"})
(HERE / "check2_lookahead.json").write_text(json.dumps(out, indent=1, default=int))
