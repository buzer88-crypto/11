"""Adversarial review (lens: practicality and attribution) of the BTC long/flat trend claim.

Chosen config: SMA50 trend filter, sized to 40% vol (60d realized, max weight 1), daily rebalance.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pinetest.crypto_data import load_panel
from pinetest.portfolio import TEST, TRAIN, backtest, buy_and_hold, rebalance_on

PRE = ("2014-01-01", "2017-12-31")
TEST_EX22 = ("2023-01-01", None)
FULL = ("2014-01-01", None)
PERIODS = {"pre": PRE, "train": TRAIN, "test": TEST, "test_ex2022": TEST_EX22, "full_2014_2026": FULL}
OUT = Path(__file__).parent

px = load_panel()["PriceUSD"]
p = px["btc"]
ret = p.pct_change(fill_method=None)
vol60 = ret.rolling(60).std() * np.sqrt(365)


def frame(series):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w["btc"] = series.where(p.notna(), 0.0).fillna(0.0)
    return w


def trend(n=50, kind="sma"):
    if kind == "sma":
        return (p > p.rolling(n).mean()).astype(float)
    return (p / p.shift(n) - 1 > 0).astype(float)


vt = (0.40 / vol60).clip(upper=1.0)


def met(res, period):
    x = res.metrics(*period)
    return {k: round(x[k], 3) for k in ("sharpe", "cagr_pct", "total_pct", "vol_pct", "max_dd_pct",
                                         "exposure", "turnover_per_year")}


def allp(res):
    return {k: met(res, v) for k, v in PERIODS.items()}


def yearly(res, years):
    return {str(y): round(100 * ((1 + res.returns.loc[str(y)]).prod() - 1), 1) for y in years}


def band(w, thr=0.10):
    """No-trade band: emit a new target only when it differs from the last traded target by more
    than `thr`, or when it is an entry from 0 / exit to 0 (trend switch). Other rows -> NaN (hold)."""
    s = w["btc"].to_numpy()
    out = np.full(len(s), np.nan)
    last = 0.0
    for i, x in enumerate(s):
        if (x == 0.0 and last != 0.0) or (x > 0 and last == 0.0) or abs(x - last) > thr:
            out[i] = x
            last = x
    o = pd.DataFrame(np.nan, index=w.index, columns=w.columns)
    rows = ~np.isnan(out)
    o.loc[rows, :] = 0.0
    o.loc[rows, "btc"] = out[rows]
    return o


def trade_stats(res, sig_on, period):
    """Trades = days with nonzero traded notional (>1e-9). Spells = contiguous days with BTC held > 0."""
    a, b = period
    t = res.turnover.loc[a:b]
    years = len(t) / 365
    held = res.held["btc"].loc[a:b]
    on = (held > 1e-9).astype(int)
    starts = ((on.diff() == 1)).sum() + int(on.iloc[0] == 1)
    # spell lengths
    grp = (on != on.shift()).cumsum()
    spells = on.groupby(grp).agg(["first", "size"])
    on_spells = spells[spells["first"] == 1]["size"]
    off_spells = spells[spells["first"] == 0]["size"]
    tiny = ((t > 1e-9) & (t < 0.05)).sum()
    return {
        "trade_days_per_year": round(float((t > 1e-9).sum() / years), 1),
        "tiny_trades_lt5pct_per_year": round(float(tiny / years), 1),
        "entries_per_year": round(float(starts / years), 2),
        "avg_holding_days": round(float(on_spells.mean()), 1) if len(on_spells) else None,
        "median_holding_days": float(on_spells.median()) if len(on_spells) else None,
        "avg_flat_days": round(float(off_spells.mean()), 1) if len(off_spells) else None,
        "n_holding_spells_1to5_days": int((on_spells <= 5).sum()),
        "n_holding_spells": int(len(on_spells)),
        "turnover_per_year": round(float(t.sum() / years), 2),
        "cost_drag_pct_per_year_at_10bps": round(float(t.sum() / years * 0.001 * 100), 2),
    }


def dd_info(res, period):
    r = res.returns.loc[period[0]:period[1]]
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]]
    rec_date = str(rec.index[0].date()) if len(rec) else "not recovered"
    # longest underwater spell
    uw = (dd < 0).astype(int)
    g = (uw != uw.shift()).cumsum()
    spells = uw.groupby(g).agg(["first", "size"])
    longest = int(spells[spells["first"] == 1]["size"].max()) if (spells["first"] == 1).any() else 0
    return {"max_dd_pct": round(100 * float(dd.min()), 1), "peak": str(peak.date()),
            "trough": str(trough.date()), "recovered": rec_date, "longest_underwater_days": longest}


def boot_sharpe_diff(r1, r2, n=2000, block=60, seed=7):
    """Circular block bootstrap of Sharpe(r1) - Sharpe(r2), paired."""
    rng = np.random.default_rng(seed)
    a, b = r1.to_numpy(), r2.to_numpy()
    T = len(a)
    k = int(np.ceil(T / block))
    diffs = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        x, y = a[idx], b[idx]
        diffs[i] = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(365)
    return {"point": round(float((a.mean() / a.std() - b.mean() / b.std()) * np.sqrt(365)), 3),
            "ci90": [round(float(np.quantile(diffs, 0.05)), 3), round(float(np.quantile(diffs, 0.95)), 3)],
            "p_le_0": round(float((diffs <= 0).mean()), 3)}


R = {}
bh = backtest(buy_and_hold(px, "btc"), px)
w_both = frame(trend(50) * vt)
w_trend = frame(trend(50))
w_vt = frame(vt)
res_both = backtest(w_both, px)

# ---- 1. Decomposition ----------------------------------------------------------------------
mean_exp_test = res_both.metrics(*TEST)["exposure"]
const = {}
for c in (0.43, 0.5):
    const[f"const_{c}_daily"] = backtest(frame(pd.Series(c, index=p.index)), px)
dec = {
    "buy_hold": allp(bh),
    "trend_only_sma50_full_daily": allp(backtest(w_trend, px)),
    "voltarget_only_40_always_in_daily": allp(backtest(w_vt, px)),
    "both_chosen": allp(res_both),
    **{k: allp(v) for k, v in const.items()},
}
R["1_decomposition"] = dec
R["1_mean_exposure_test_chosen"] = round(mean_exp_test, 3)

# Sharpe-difference significance (TEST), chosen vs B&H and components vs B&H
tb, te = TEST
R["1_bootstrap_sharpe_diff_test_vs_bh"] = {
    "both": boot_sharpe_diff(res_both.returns.loc[tb:te], bh.returns.loc[tb:te]),
    "trend_only": boot_sharpe_diff(backtest(w_trend, px).returns.loc[tb:te], bh.returns.loc[tb:te]),
    "vt_only": boot_sharpe_diff(backtest(w_vt, px).returns.loc[tb:te], bh.returns.loc[tb:te]),
}
R["1_bootstrap_sharpe_diff_test_ex2022_vs_bh"] = {
    "both": boot_sharpe_diff(res_both.returns.loc["2023-01-01":], bh.returns.loc["2023-01-01":]),
}

# ---- 2. Trading reality --------------------------------------------------------------------
sig = trend(50)
R["2_trade_stats"] = {
    "chosen_daily": {k: trade_stats(res_both, sig, v) for k, v in (("pre", PRE), ("train", TRAIN), ("test", TEST))},
}
R["2_signal_flips_per_year_sma50"] = {
    k: round(float((sig.loc[v[0]:v[1]].diff().abs() > 0).sum() / (len(sig.loc[v[0]:v[1]]) / 365)), 1)
    for k, v in (("pre", PRE), ("train", TRAIN), ("test", TEST))
}
costs = {}
for c in (0, 10, 30, 50, 100):
    r = backtest(w_both, px, cost_bps=c)
    costs[str(c)] = {k: met(r, v)["sharpe"] for k, v in (("pre", PRE), ("train", TRAIN), ("test", TEST))} | \
        {"test_total_pct": met(r, TEST)["total_pct"]}
R["2_cost_sensitivity_chosen_sharpe"] = costs
R["2_cost_sensitivity_bh_sharpe_test"] = {str(c): met(backtest(buy_and_hold(px, "btc"), px, cost_bps=c), TEST)["sharpe"]
                                         for c in (10, 100)}
lags = {}
for L in (0, 1, 2, 3):
    r = backtest(w_both, px, lag=L)
    lags[str(L)] = {k: met(r, v)["sharpe"] for k, v in (("pre", PRE), ("train", TRAIN), ("test", TEST))} | \
        {"test_max_dd": met(r, TEST)["max_dd_pct"]}
R["2_lag_sensitivity_chosen_sharpe"] = lags

w_band = band(w_both, 0.10)
res_band = backtest(w_band, px)
R["2_no_trade_band_10pp"] = {
    "metrics": allp(res_band),
    "trade_stats": {k: trade_stats(res_band, sig, v) for k, v in (("pre", PRE), ("train", TRAIN), ("test", TEST))},
    "cost_sharpe": {str(c): {k: met(backtest(w_band, px, cost_bps=c), v)["sharpe"]
                             for k, v in (("pre", PRE), ("test", TEST))} for c in (10, 30, 50, 100)},
}
w_band_w = band(rebalance_on(w_both, "W-SUN").ffill(), 0.10)
# weekly + band: only evaluate the band on weekly decision days
wk = rebalance_on(w_both, "W-SUN")
mask = wk["btc"].notna()
wb = band(w_both.loc[mask].reindex(w_both.index).ffill().fillna(0.0), 0.10)
wb.loc[~mask.to_numpy()] = np.nan
res_wb = backtest(wb, px)
R["2_weekly_plus_band_10pp"] = {"metrics": allp(res_wb),
                               "trade_stats_test": trade_stats(res_wb, sig, TEST)}

# ---- 3. Simpler rules ----------------------------------------------------------------------
simple = {
    "sma200_full_daily": frame(trend(200)),
    "sma50_full_weekly": rebalance_on(frame(trend(50)), "W-SUN"),
    "sma200_full_weekly": rebalance_on(frame(trend(200)), "W-SUN"),
    "sma100_full_daily": frame(trend(100)),
    "sma150_full_daily": frame(trend(150)),
}
R["3_simpler_rules"] = {}
for k, w in simple.items():
    r = backtest(w, px)
    R["3_simpler_rules"][k] = {"metrics": allp(r), "trade_stats_test": trade_stats(r, sig, TEST),
                               "test_sharpe_cost50": met(backtest(w, px, cost_bps=50), TEST)["sharpe"],
                               "test_sharpe_lag3": met(backtest(w, px, lag=3), TEST)["sharpe"],
                               "yearly": yearly(r, range(2014, 2027))}

# ---- 4. Calendar years and lived drawdowns -------------------------------------------------
yrs = range(2014, 2027)
ys, yb = yearly(res_both, yrs), yearly(bh, yrs)
cal = {}
for y in yrs:
    rs = res_both.returns.loc[str(y)]
    rb = bh.returns.loc[str(y)]
    cal[str(y)] = {"strategy_pct": ys[str(y)], "bh_pct": yb[str(y)],
                   "better_return": ys[str(y)] > yb[str(y)],
                   "strategy_sharpe": round(float(rs.mean() / rs.std() * np.sqrt(365)), 2),
                   "bh_sharpe": round(float(rb.mean() / rb.std() * np.sqrt(365)), 2),
                   "strategy_intra_year_dd": round(100 * float(((1 + rs).cumprod() / (1 + rs).cumprod().cummax() - 1).min()), 1),
                   "bh_intra_year_dd": round(100 * float(((1 + rb).cumprod() / (1 + rb).cumprod().cummax() - 1).min()), 1)}
R["4_calendar_years"] = cal
R["4_years_strategy_return_beats_bh"] = sum(v["better_return"] for v in cal.values())
R["4_lived_drawdowns"] = {
    "chosen_full_2014_2026": dd_info(res_both, FULL),
    "chosen_test": dd_info(res_both, TEST),
    "bh_full_2014_2026": dd_info(bh, FULL),
    "bh_test": dd_info(bh, TEST),
    "chosen_band": dd_info(res_band, FULL),
    "sma200_full_daily": dd_info(backtest(simple["sma200_full_daily"], px), FULL),
}
eqs = res_both.equity(*FULL).iloc[-1]
eqb = bh.equity(*FULL).iloc[-1]
R["4_terminal_wealth_multiple_2014_2026"] = {"chosen": round(float(eqs), 1), "bh": round(float(eqb), 1)}

(OUT / "review_results.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
