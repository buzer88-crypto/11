"""Checks 1, 3, 4 (episode dependence, outlier days, start dates, exposure-matched static)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import BH, PRE, PX, STRAT, STRAT_W, TEST, TRAIN, bt, met, static_w, stats, trend_w, vt_only_w  # noqa

OUT = {}
s, b = STRAT.returns, BH.returns
END = "2026-05-23"  # last day with a BTC price

# ---------------- Check 1: dependence on 2022 ----------------
c1 = {}
for lab, (a, z) in {"test_full": ("2022-01-01", END), "test_ex2022": ("2023-01-01", END),
                    "test_2022_only": ("2022-01-01", "2022-12-31"),
                    "2023_2024": ("2023-01-01", "2024-12-31"), "2025_2026": ("2025-01-01", END)}.items():
    c1[lab] = {"strat": met(STRAT, a, z), "bh": met(BH, a, z)}
# share of TEST excess log-growth that comes from 2022
lg = lambda r: float(np.log1p(r).sum())
ex_total = lg(s.loc["2022":END]) - lg(b.loc["2022":END])
ex_2022 = lg(s.loc["2022"]) - lg(b.loc["2022"])
c1["excess_log_growth"] = {"test_total": round(ex_total, 3), "from_2022": round(ex_2022, 3),
                           "from_2023_on": round(ex_total - ex_2022, 3)}
# calendar-year Sharpe diffs 2014-2026
years = {}
for y in range(2014, 2027):
    rs, rb = s.loc[str(y)].loc[:END], b.loc[str(y)].loc[:END]
    ss, sb = stats(rs), stats(rb)
    years[y] = {"strat_sh": ss["sharpe"], "bh_sh": sb["sharpe"], "diff": round(ss["sharpe"] - sb["sharpe"], 2),
                "strat_ret": ss["total_pct"], "bh_ret": sb["total_pct"],
                "strat_dd": ss["max_dd_pct"], "bh_dd": sb["max_dd_pct"]}
c1["calendar_years"] = years
c1["calendar_years_strat_wins_sharpe"] = f"{sum(v['diff'] > 0 for v in years.values())}/{len(years)}"
c1["calendar_years_strat_wins_return"] = f"{sum(v['strat_ret'] > v['bh_ret'] for v in years.values())}/{len(years)}"


# rolling 365-day Sharpe difference, windows fully inside 2014-01-01..END
def roll_sh(r):
    return r.rolling(365).mean() / r.rolling(365).std() * np.sqrt(365)


ss_, bs_ = roll_sh(s.loc["2014-01-01":END]), roll_sh(b.loc["2014-01-01":END])
d = (ss_ - bs_).dropna()
rs1 = (1 + s.loc["2014-01-01":END]).rolling(365).apply(np.prod, raw=True) - 1
rb1 = (1 + b.loc["2014-01-01":END]).rolling(365).apply(np.prod, raw=True) - 1
rd = (rs1 - rb1).dropna()
bh_up = (rb1 > 0).reindex(d.index)
roll = {"n_windows": int(len(d)), "share_strat_sharpe_wins": round(float((d > 0).mean()), 3),
        "median_diff": round(float(d.median()), 3), "mean_diff": round(float(d.mean()), 3),
        "p10_diff": round(float(d.quantile(0.1)), 3), "p90_diff": round(float(d.quantile(0.9)), 3),
        "share_strat_return_wins": round(float((rd > 0).mean()), 3),
        "share_wins_when_bh_1y_up": round(float((d[bh_up] > 0).mean()), 3),
        "share_wins_when_bh_1y_down": round(float((d[~bh_up] > 0).mean()), 3),
        "share_bh_1y_up": round(float(bh_up.mean()), 3)}
for lab, (a, z) in {"pre_windows(end<=2017)": ("2014", "2017"), "train_windows(end 2018-21)": ("2018", "2021"),
                    "test_windows(end>=2022-12-31)": ("2022-12-31", END)}.items():
    dd = d.loc[a:z]
    roll[lab] = {"n": int(len(dd)), "share_wins": round(float((dd > 0).mean()), 3), "median_diff": round(float(dd.median()), 3)}
# windows ending in TEST that do not touch 2022 at all
dd = d.loc["2023-12-31":END]
roll["windows_starting_2023_or_later"] = {"n": int(len(dd)), "share_wins": round(float((dd > 0).mean()), 3),
                                          "median_diff": round(float(dd.median()), 3)}
c1["rolling_1y_sharpe_diff"] = roll
OUT["check1"] = c1

# ---------------- Check 3a: drop best/worst days ----------------
def drop_days(r, nbest=0, nworst=0):
    r = r.copy()
    order = r.sort_values()
    drop = list(order.index[:nworst]) + (list(order.index[-nbest:]) if nbest else [])
    return r.drop(drop)

c3 = {}
for lab, (a, z) in {"test": ("2022-01-01", END), "pre": PRE, "test_ex2022": ("2023-01-01", END)}.items():
    rs, rb = s.loc[a:z], b.loc[a:z]
    c3[lab] = {
        "strat_raw": stats(rs), "bh_raw": stats(rb),
        "strat_drop5best5worst": stats(drop_days(rs, 5, 5)), "bh_drop5best5worst": stats(drop_days(rb, 5, 5)),
        "strat_drop5best": stats(drop_days(rs, 5, 0)), "bh_drop5best": stats(drop_days(rb, 5, 0)),
        "strat_drop10best": stats(drop_days(rs, 10, 0)),
        "strat_drop5worst": stats(drop_days(rs, 0, 5)),
        # same calendar days removed from both
        "bh_on_strat_days_removed": stats(rb.drop(drop_days(rs, 5, 5).index.symmetric_difference(rs.index))),
    }
    top = rs.sort_values()
    c3[lab]["strat_5best_days"] = {str(k.date()): round(100 * v, 2) for k, v in top.iloc[-5:].items()}
    c3[lab]["strat_5worst_days"] = {str(k.date()): round(100 * v, 2) for k, v in top.iloc[:5].items()}
OUT["check3_days"] = c3

# ---------------- Check 3b: start month ----------------
def from_start(start, w=None, base=None):
    w = (STRAT_W if w is None else w).copy()
    w.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)] = 0.0
    return bt(w).metrics(start, END)

rows = []
starts = [f"2022-{m:02d}-01" for m in range(1, 13)] + [f"2021-{m:02d}-01" for m in range(1, 13)] + \
         [f"2023-{m:02d}-01" for m in range(1, 13)]
bhw = pd.DataFrame({"btc": PX["btc"].notna().astype(float)})
for st in starts:
    ms, mb = from_start(st), from_start(st, bhw)
    rows.append({"start": st, "strat_sh": round(ms["sharpe"], 3), "bh_sh": round(mb["sharpe"], 3),
                 "strat_tot": round(ms["total_pct"], 1), "bh_tot": round(mb["total_pct"], 1),
                 "strat_dd": round(ms["max_dd_pct"], 1), "bh_dd": round(mb["max_dd_pct"], 1)})
sm = pd.DataFrame(rows).set_index("start")
OUT["check3_start_months"] = sm.to_dict(orient="index")
for yr in ("2021", "2022", "2023"):
    sub = sm[sm.index.str.startswith(yr)]
    OUT[f"check3_start_summary_{yr}"] = {
        "strat_sharpe_wins": f"{int((sub.strat_sh > sub.bh_sh).sum())}/12",
        "strat_return_wins": f"{int((sub.strat_tot > sub.bh_tot).sum())}/12",
        "dd_better": f"{int((sub.strat_dd > sub.bh_dd).sum())}/12",
        "median_sharpe_diff": round(float((sub.strat_sh - sub.bh_sh).median()), 3),
        "min_sharpe_diff": round(float((sub.strat_sh - sub.bh_sh).min()), 3)}

# ---------------- Check 4: static allocation at matched exposure / vol ----------------
c4 = {}
for lab, (a, z) in {"train": TRAIN, "test": ("2022-01-01", END), "pre": PRE, "test_ex2022": ("2023-01-01", END)}.items():
    ms = met(STRAT, a, z)
    vol_frac = ms["vol_pct"] / met(BH, a, z)["vol_pct"]
    row = {"strat": ms, "bh": met(BH, a, z)}
    for f, nm in ((round(ms["exposure"], 2), "static_at_mean_exposure_W"), (round(vol_frac, 2), "static_at_matched_vol_W"),
                  (0.43, "static_0.43_W")):
        row[nm] = {"frac": f, **met(bt(static_w(f)), a, z)}
    row["vt40_always_on_D(no trend filter)"] = met(bt(vt_only_w(0.40)), a, z)
    row["sma50_full_size_D(no vt)"] = met(bt(trend_w(50, None)), a, z)
    # regression of strategy on B&H returns
    rs, rb = s.loc[a:z], b.loc[a:z]
    beta = float(np.cov(rs, rb)[0, 1] / rb.var())
    resid = rs - beta * rb
    row["regression_on_bh"] = {"beta": round(beta, 3), "alpha_ann_pct": round(float(resid.mean() * 365 * 100), 2),
                               "alpha_t": round(float(resid.mean() / resid.std() * np.sqrt(len(resid))), 2)}
    c4[lab] = row
OUT["check4"] = c4

# ---------------- extra: paired stationary block bootstrap of TEST Sharpe diff ----------------
rng = np.random.default_rng(7)
rs, rb = s.loc["2022-01-01":END].to_numpy(), b.loc["2022-01-01":END].to_numpy()
st0 = bt(static_w(0.43)).returns.loc["2022-01-01":END].to_numpy()
n = len(rs)
diffs, diffs_st = [], []
for _ in range(2000):
    idx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        j = rng.integers(n)
        L = rng.geometric(1 / 30)
        k = min(L, n - i)
        idx[i:i + k] = (j + np.arange(k)) % n
        i += k
    sh = lambda x: x.mean() / x.std() * np.sqrt(365)
    diffs.append(sh(rs[idx]) - sh(rb[idx]))
    diffs_st.append(sh(rs[idx]) - sh(st0[idx]))
diffs, diffs_st = np.array(diffs), np.array(diffs_st)
OUT["bootstrap_test_sharpe_diff_vs_bh"] = {"point": round(float(rs.mean() / rs.std() * np.sqrt(365) - rb.mean() / rb.std() * np.sqrt(365)), 3),
                                           "p_le_0": round(float((diffs <= 0).mean()), 3),
                                           "ci90": [round(float(np.quantile(diffs, .05)), 3), round(float(np.quantile(diffs, .95)), 3)]}
OUT["bootstrap_test_sharpe_diff_vs_static43"] = {"p_le_0": round(float((diffs_st <= 0).mean()), 3),
                                                 "ci90": [round(float(np.quantile(diffs_st, .05)), 3), round(float(np.quantile(diffs_st, .95)), 3)]}

Path(__file__).with_name("results_1_3_4.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps(OUT, indent=1, default=float))
