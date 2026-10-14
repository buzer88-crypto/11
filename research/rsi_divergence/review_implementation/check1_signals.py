"""Check 1: compare the independent Pine emulation with pinetest.divergence.rsi_divergences()
on BTC 4h, left=right in {5, 3}, all four divergence types, and explain every disagreement."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tv_reimpl import pine_rsi, tv_divergences  # noqa: E402

from pinetest.divergence import pivots, rsi_divergences  # noqa: E402  (only for the comparison)
from pinetest.indicators import rsi as study_rsi  # noqa: E402

ROOT = HERE.parents[2]
df = pd.read_csv(ROOT / "data" / "BTCUSD_4h.csv", parse_dates=["date"]).set_index("date").loc["2013-01-01":]
o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
n = len(c)
res = {}

# RSI agreement
my_rsi = pine_rsi(c)
st_rsi = study_rsi(df["close"], 14).to_numpy()
both = ~np.isnan(my_rsi) & ~np.isnan(st_rsi)
res["rsi"] = {"nan_pattern_equal": bool((np.isnan(my_rsi) == np.isnan(st_rsi)).all()),
              "max_abs_diff": float(np.abs(my_rsi[both] - st_rsi[both]).max())}
print("RSI", res["rsi"])

TYPES = {"regular_bull": ("regular", 1), "hidden_bull": ("hidden", 1),
         "regular_bear": ("regular", -1), "hidden_bear": ("hidden", -1)}
period_lo = df.index.searchsorted(pd.Timestamp("2014-01-01"))


def study_signals(kind, side, lb, buffer_atr):
    ss = rsi_divergences(df, kind=kind, left=lb, right=lb, rr=1.0, sides=(side,), buffer_atr=buffer_atr)
    return {s.bar for s in ss}


def explain(bar, tvrec, kind, side, lb, my_osc, ties_piv):
    """Classify why the study and the emulation disagree on this signal bar."""
    is_low = side > 0
    price = l if is_low else h
    p = bar - lb
    # pivots by both conventions
    st_piv = np.flatnonzero(pivots(st_rsi, lb, lb, is_low))
    tv_piv = ties_piv
    in_st, in_tv = p in set(st_piv), p in set(tv_piv)
    if in_st != in_tv:
        return "pivot_tie(centre)"
    # previous pivot differs?
    prev_st = st_piv[st_piv < p][-1] if (st_piv < p).any() else None
    prev_tv = tv_piv[tv_piv < p][-1] if (tv_piv < p).any() else None
    if prev_st != prev_tv:
        return "pivot_tie(previous pivot differs)"
    d = p - prev_st
    if (6 <= d <= 61) != (5 <= d <= 60):
        return f"range_offbyone(dist={d})"
    if price[p] == price[prev_st]:
        return "equal_price_extreme"
    if my_osc[p] == my_osc[prev_st] or st_rsi[p] == st_rsi[prev_st]:
        return "equal_rsi"
    return "UNEXPLAINED"


summary = {}
for lb in (5, 3):
    for ties in ("left_strict", "strict", "right_strict"):
        tv, my_osc, state = tv_divergences(o, h, l, c, lbL=lb, lbR=lb, ties=ties)
        for tname, (kind, side) in TYPES.items():
            key = f"lb{lb}_{ties}_{tname}"
            tvbars = {r[0] for r in tv[tname] if r[0] + 1 < n}
            st_raw = study_signals(kind, side, lb, 1e9)      # no 'open already beyond stop' filter
            st_def = study_signals(kind, side, lb, 0.1)      # as used in the study
            inter = tvbars & st_raw
            union = tvbars | st_raw
            piv = np.flatnonzero(state["pl" if side > 0 else "ph"]) - lb
            reasons = {}
            details = []
            for b in sorted(union - inter):
                why = explain(b, None, kind, side, lb, my_osc, piv)
                who = "tv_only" if b in tvbars else "study_only"
                reasons[f"{who}:{why}"] = reasons.get(f"{who}:{why}", 0) + 1
                details.append({"bar": int(b), "date": str(df.index[b]), "who": who, "why": why})
            summary[key] = {
                "tv_n": len(tvbars), "study_n_raw": len(st_raw), "both": len(inter),
                "jaccard_pct": round(100 * len(inter) / max(len(union), 1), 2),
                "tv_only": len(tvbars - st_raw), "study_only": len(st_raw - tvbars),
                "reasons": reasons,
                "study_n_after_risk_filter": len(st_def),
                "dropped_by_open_beyond_stop": len(st_raw - st_def),
                "since2014_tv_n": sum(b >= period_lo for b in tvbars),
                "since2014_mismatch": sum(b >= period_lo for b in union - inter),
                "details": details,
            }
            print(key, {k: v for k, v in summary[key].items() if k != "details"})
res["signals"] = summary
(HERE / "check1_signals.json").write_text(json.dumps(res, indent=1, default=str))
