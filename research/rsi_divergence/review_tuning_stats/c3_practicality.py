"""Check 3: the dominant walk-forward pick traded on its own over 2014-2026 (signal bars 2014-01-01 ..
2026-10-14), against BTC buy and hold and a daily SMA50 long/flat filter over the same span.
Risk 1% of equity per trade, compounded; 0.1% per side (as in the study).
Also: BTC held as a static (never rebalanced) fraction of capital sized to the strategy's drawdown.
Output: c3_practicality.json.  PYTHONPATH=. python .../c3_practicality.py"""
import json
import pickle

import numpy as np
import pandas as pd

from wf_lib import COST, DOMINANT, OUT, load_df, sim_market

START = "2014-01-01"


def max_dd(eq):
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    k = int(np.argmin(dd))
    return float(dd.min()), k, int(np.argmax(eq[:k + 1])) if k > 0 else 0


def cagr(total_mult, years):
    return total_mult ** (1 / years) - 1


def main():
    df = load_df("4h")
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = df.index
    i0 = int(idx.searchsorted(pd.Timestamp(START)))
    n = len(c)
    span_years = (idx[-1] - idx[i0]).total_seconds() / (365.25 * 86400)

    meta = pickle.loads((OUT / "trades_meta.pkl").read_bytes())
    k = meta["configs"].index(DOMINANT)
    tr = meta["trades"][k]
    tr = tr[tr[:, 0] >= i0]
    sig, fill, ex, side, riskp, r = (tr[:, j] for j in range(6))
    sig, fill, ex = sig.astype(int), fill.astype(int), ex.astype(int)
    yrs = idx.year.to_numpy()

    # equity, closed-trade and mark-to-market on 4h closes
    eq_closed = [1.0]
    mtm = np.ones(n - i0)
    eq = 1.0
    last = i0
    for a, b, e_risk, rr_ in zip(fill, ex, riskp, r):
        mtm[last - i0:a - i0] = eq
        entry = o[a]
        units = 0.01 * eq / (e_risk * entry)
        seg = eq + units * (c[a:b + 1] - entry) - COST * entry * units
        mtm[a - i0:b + 1 - i0] = seg
        eq = eq * (1 + 0.01 * rr_)
        mtm[b - i0] = eq
        eq_closed.append(eq)
        last = b + 1
    mtm[last - i0:] = eq
    dd_closed, _, _ = max_dd(eq_closed)
    dd_mtm, kdd, kpk = max_dd(mtm)

    losing = r < 0
    streak = best = 0
    for x in losing:
        streak = streak + 1 if x else 0
        best = max(best, streak)
    bars_in = int((ex - fill + 1).sum())
    notional = 0.01 / riskp
    per_year = {}
    for y in range(2014, 2027):
        m = yrs[sig] == y
        rr_y = r[m]
        btc_y = c[yrs == y]
        per_year[y] = {"n": int(m.sum()), "avg_r": round(float(rr_y.mean()), 3) if m.any() else None,
                       "return_pct_1pct_risk": round(100 * float(np.prod(1 + 0.01 * rr_y) - 1), 1),
                       "btc_pct": round(100 * float(btc_y[-1] / o[np.flatnonzero(yrs == y)[0]] - 1), 1)}

    total_mult = eq
    strat = {
        "config": DOMINANT, "n_trades": int(len(r)), "trades_per_year": round(len(r) / span_years, 1),
        "trades_per_year_min_max": [min(v["n"] for v in per_year.values()), max(v["n"] for v in per_year.values())],
        "win_rate_pct": round(100 * float((r > 0).mean()), 1),
        "hit_4R_target_pct": round(100 * float((r > 3.5).mean()), 1),
        "avg_r": round(float(r.mean()), 3), "median_r": round(float(np.median(r)), 3),
        "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2),
        "avg_r_2014_2016": round(float(r[yrs[sig] <= 2016].mean()), 3), "n_2014_2016": int((yrs[sig] <= 2016).sum()),
        "avg_r_2017_2026": round(float(r[yrs[sig] >= 2017].mean()), 3), "n_2017_2026": int((yrs[sig] >= 2017).sum()),
        "avg_r_2018_2026": round(float(r[yrs[sig] >= 2018].mean()), 3), "n_2018_2026": int((yrs[sig] >= 2018).sum()),
        "longest_losing_streak": int(best),
        "total_return_pct_1pct_risk": round(100 * (total_mult - 1), 1),
        "cagr_pct_1pct_risk": round(100 * cagr(total_mult, span_years), 2),
        "max_dd_closed_trade_pct": round(100 * dd_closed, 1),
        "max_dd_mark_to_market_pct": round(100 * dd_mtm, 1),
        "max_dd_mtm_peak_to_trough": [str(idx[i0 + kpk]), str(idx[i0 + kdd])],
        "time_in_market_pct": round(100 * bars_in / (n - i0), 1),
        "avg_bars_held": round(float((ex - fill + 1).mean()), 1),
        "notional_pct_of_equity": {"median": round(100 * float(np.median(notional)), 0), "mean": round(100 * float(notional.mean()), 0),
                                   "max": round(100 * float(notional.max()), 0), "share_trades_over_100pct": round(float((notional > 1).mean()), 3)},
        "risk_pct_of_price": {"median": round(100 * float(np.median(riskp)), 2), "p10": round(100 * float(np.quantile(riskp, 0.1)), 2),
                              "p90": round(100 * float(np.quantile(riskp, 0.9)), 2)},
        "avg_exposure_pct_of_equity_over_span": round(100 * float((notional * (ex - fill + 1)).sum() / (n - i0)), 1),
        "per_year": per_year,
    }

    # Same trades at higher risk per trade
    lev = {}
    for rk in (0.02, 0.03, 0.05):
        e = np.cumprod(1 + rk * r)
        lev[f"{int(rk*100)}pct"] = {"total_return_pct": round(100 * (e[-1] - 1), 1),
                                    "max_dd_closed_trade_pct": round(100 * max_dd(np.r_[1, e])[0], 1),
                                    "median_notional_pct": round(100 * float(np.median(rk / riskp)), 0)}
    strat["other_risk_levels"] = lev

    # Placebo for this config: same trades (side, risk %, rr, hold, count per year), random signal bars in the same year
    rng = np.random.default_rng(99)
    pl = []
    ystart = {y: int(np.searchsorted(yrs, y)) for y in range(2014, 2027)}
    yend = {y: min(int(np.searchsorted(yrs, y + 1)), n - 2) for y in range(2014, 2027)}
    ty = yrs[sig]
    lo = np.array([ystart[y] for y in ty]); hi = np.array([yend[y] for y in ty])
    for _ in range(5000):
        b = lo + (rng.random(len(ty)) * (hi - lo)).astype(int)
        _, rp = sim_market(o, h, l, c, b + 1, side, riskp, np.full(len(b), 4.0), np.full(len(b), 60))
        pl.append((rp.mean(), rp[ty >= 2017].mean(), rp[ty >= 2018].mean()))
    pl = np.array(pl)
    strat["random_timing_placebo"] = {
        "note": "independent trades (overlap allowed), 5000 draws",
        "placebo_avg_r_2014_2026_mean": round(float(pl[:, 0].mean()), 3),
        "P_placebo_ge_real_2014_2026": round(float((pl[:, 0] >= r.mean()).mean()), 4),
        "placebo_avg_r_2017_2026_mean": round(float(pl[:, 1].mean()), 3),
        "P_placebo_ge_real_2017_2026": round(float((pl[:, 1] >= r[ty >= 2017].mean()).mean()), 4),
        "placebo_avg_r_2018_2026_mean": round(float(pl[:, 2].mean()), 3),
        "P_placebo_ge_real_2018_2026": round(float((pl[:, 2] >= r[ty >= 2018].mean()).mean()), 4)}

    # Benchmarks on daily bars, same span
    d = load_df("1d").loc[START:]
    dc, do = d["close"].to_numpy(float), d["open"].to_numpy(float)
    bh = dc / do[0]
    bh_dd = max_dd(bh)[0]
    dy = (d.index[-1] - d.index[0]).total_seconds() / (365.25 * 86400)
    full = load_df("1d")
    sma = full["close"].rolling(50).mean().loc[START:].to_numpy()
    ret = np.r_[dc[0] / do[0] - 1, dc[1:] / dc[:-1] - 1]
    pos = np.r_[0, (dc > sma)[:-1].astype(float)]                 # decided at yesterday's close
    sw = np.abs(np.diff(np.r_[0, pos]))
    sma_eq = np.cumprod(1 + pos * ret - sw * COST)
    bench = {
        "btc_buy_hold": {"total_return_pct": round(100 * (bh[-1] - 1), 1), "cagr_pct": round(100 * cagr(bh[-1], dy), 1),
                         "max_dd_pct": round(100 * bh_dd, 1), "time_in_market_pct": 100.0},
        "btc_sma50_long_flat": {"total_return_pct": round(100 * (sma_eq[-1] - 1), 1), "cagr_pct": round(100 * cagr(sma_eq[-1], dy), 1),
                                "max_dd_pct": round(100 * max_dd(sma_eq)[0], 1), "time_in_market_pct": round(100 * float(pos.mean()), 1),
                                "switches_per_year": round(float(sw.sum() / dy), 1)},
    }
    # static BTC sleeve (rest in cash, never rebalanced) with the same mark-to-market max drawdown as the strategy
    target = abs(dd_mtm)
    lo_f, hi_f = 0.0, 1.0
    for _ in range(60):
        f = (lo_f + hi_f) / 2
        e = (1 - f) + f * bh
        if abs(max_dd(e)[0]) > target:
            hi_f = f
        else:
            lo_f = f
    f = lo_f
    e = (1 - f) + f * bh
    bench["btc_static_sleeve_same_dd"] = {"btc_fraction_pct": round(100 * f, 1), "total_return_pct": round(100 * (e[-1] - 1), 1),
                                          "cagr_pct": round(100 * cagr(e[-1], dy), 2), "max_dd_pct": round(100 * max_dd(e)[0], 1)}
    # daily-rebalanced constant fraction with the same max drawdown
    lo_f, hi_f = 0.0, 1.0
    for _ in range(60):
        f = (lo_f + hi_f) / 2
        e = np.cumprod(1 + f * ret)
        if abs(max_dd(e)[0]) > target:
            hi_f = f
        else:
            lo_f = f
    f = lo_f
    e = np.cumprod(1 + f * ret)
    bench["btc_rebalanced_fraction_same_dd"] = {"btc_fraction_pct": round(100 * f, 1), "total_return_pct": round(100 * (e[-1] - 1), 1),
                                                "cagr_pct": round(100 * cagr(e[-1], dy), 2), "max_dd_pct": round(100 * max_dd(e)[0], 1)}
    lo_f, hi_f = 0.0, 1.0
    for _ in range(60):
        f = (lo_f + hi_f) / 2
        e = np.cumprod(1 + f * (pos * ret - sw * COST))
        if abs(max_dd(e)[0]) > target:
            hi_f = f
        else:
            lo_f = f
    f = lo_f
    e = np.cumprod(1 + f * (pos * ret - sw * COST))
    bench["sma50_fraction_same_dd"] = {"fraction_pct": round(100 * f, 1), "total_return_pct": round(100 * (e[-1] - 1), 1),
                                       "cagr_pct": round(100 * cagr(e[-1], dy), 2), "max_dd_pct": round(100 * max_dd(e)[0], 1)}
    out = {"span": [str(idx[i0]), str(idx[-1])], "span_years": round(span_years, 2), "strategy": strat, "benchmarks": bench}
    (OUT / "c3_practicality.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
