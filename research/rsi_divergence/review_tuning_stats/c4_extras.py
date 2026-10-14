"""Check 4 (extras for checks 1 and 3).
A. Walk-forward picks per year (4 variants) and how often the dominant config is the pick.
B. Real cross-section 2014-2026: where the dominant config ranks among the 2400, max t over configs.
C. The actual OOS sequence (expanding / avg R): equity at 1% risk, drawdown, losing streak, time in
   market, against BTC buy-and-hold and SMA50 long/flat over 2017-2026 (also at matched drawdown).
D. Placebo universes (same construction as c1 D, plus an all-long version where every placebo trade is
   long): max full-sample t over the 2400 placebo configs, hindsight-best config, and the 4 reported
   walk-forward variants for the all-long version.
PYTHONPATH=. python .../c4_extras.py [n_draws]   (2 processes)"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import c1_multiple_testing as c1
from wf_lib import (COST, DOMINANT, OUT, REPORTED, TEST_YEARS, YEARS, _tstat, arrays_from_results, load_cache,
                    load_df, oos_stats, sim_market, walk_forward)


def max_dd(eq):
    eq = np.asarray(eq, float)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def xsec(N, S, Q, cols, min_n):
    n, s, q = N[:, cols].sum(1), S[:, cols].sum(1), Q[:, cols].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(n > 0, s / n, -np.inf)
    t = _tstat(n, s, q)
    ok = n >= min_n
    return n, avg, t, ok


def placebo(args):
    seed, all_long = args
    G = c1._G
    o, h, l, c = G["o"], G["h"], G["l"], G["c"]
    T, ty = G["T"], G["ty"]
    n = len(c)
    rng = np.random.default_rng(seed)
    lo = np.array([G["ystart"][int(y)] for y in ty])
    hi = np.array([G["yend"][int(y)] for y in ty])
    bar = lo + (rng.random(len(T)) * (hi - lo)).astype(int)
    cid = T[:, 0].astype(int)
    order = np.lexsort((bar, cid))
    cid, bar = cid[order], bar[order]
    side = np.ones(len(bar)) if all_long else T[order, 2]
    risk, rr, hold = T[order, 3], T[order, 4], T[order, 5].astype(int)
    alive = np.ones(len(bar), bool)
    exit_bar, r = sim_market(o, h, l, c, bar + 1, side, risk, rr, hold)
    same = np.r_[False, cid[1:] == cid[:-1]]
    gid = np.cumsum(~same) - 1
    for _ in range(400):
        prev_exit = np.full(len(bar), -1)
        tmp = np.where(alive, exit_bar, -1)
        cm = np.maximum.accumulate(tmp + gid * (n + 10)) - gid * (n + 10)
        prev_exit[1:] = np.where(same[1:], cm[:-1], -1)
        conflict = alive & (bar <= prev_exit)
        if not conflict.any():
            break
        nb = prev_exit[conflict] + 1
        bar[conflict] = nb
        dead = conflict.copy()
        dead[conflict] = nb + 1 >= n - 1
        alive[dead] = False
        redo = conflict & ~dead
        if redo.any():
            eb, rr_ = sim_market(o, h, l, c, bar[redo] + 1, side[redo], risk[redo], rr[redo], hold[redo])
            exit_bar[redo], r[redo] = eb, rr_
    else:
        raise RuntimeError("no convergence")
    yi = G["yrs"][np.minimum(bar, n - 1)] - YEARS[0]
    ok = alive & (yi >= 0) & (yi < len(YEARS))
    C, Y = G["nconf"], len(YEARS)
    N, S, Q = np.zeros((C, Y)), np.zeros((C, Y)), np.zeros((C, Y))
    np.add.at(N, (cid[ok], yi[ok]), 1)
    np.add.at(S, (cid[ok], yi[ok]), r[ok])
    np.add.at(Q, (cid[ok], yi[ok]), r[ok] ** 2)
    out = {}
    for name, cols, mn in (("full_2014_2026", np.arange(13), 100), ("oos_2017_2026", np.arange(3, 13), 100)):
        nn, avg, t, okc = xsec(N, S, Q, cols, mn)
        k = int(np.argmax(np.where(okc, avg, -np.inf)))
        out[name] = {"max_t": float(t[okc].max()), "best_avg_r": float(avg[k]), "best_avg_r_t": float(t[k]),
                     "median_avg_r": float(np.median(avg[okc]))}
    rep = [oos_stats(walk_forward(N, S, Q, w, m)[1]) for w, m in REPORTED]
    picks, _ = walk_forward(N, S, Q, 0, "avg_r")
    ex = [(N[p, yi], S[p, yi], Q[p, yi]) for p, yi in zip(picks, range(3, 13)) if yi != 3 and p >= 0]
    out["rep"] = [(x["n"], x["avg_r"], x["t"]) for x in rep]
    out["exp_avg_r_ex_first"] = oos_stats(tuple(np.sum(ex, 0)))
    return out


def main():
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cache = load_cache()
    configs, results = cache["configs"], cache["results"]
    N, S, Q = arrays_from_results(results)
    dom = configs.index(DOMINANT)
    out = {"dominant_index": dom}

    # A. picks
    A = {}
    for w, m in REPORTED:
        picks, _ = walk_forward(N, S, Q, w, m)
        A[f"window_{w or 'expanding'}_{m}"] = {
            "picks": {int(y): int(p) for y, p in zip(TEST_YEARS, picks)},
            "years_dominant_picked": [int(y) for y, p in zip(TEST_YEARS, picks) if p == dom],
            "n_distinct": int(len(set(picks.tolist())))}
    out["A_picks"] = A

    # B. real cross-section
    B = {}
    for name, cols, mn in (("full_2014_2026", np.arange(13), 50), ("oos_2017_2026", np.arange(3, 13), 50),
                           ("2018_2026", np.arange(4, 13), 50)):
        nn, avg, t, ok = xsec(N, S, Q, cols, mn)
        rank_avg = int((avg[ok] > avg[dom]).sum()) + 1
        rank_t = int((t[ok] > t[dom]).sum()) + 1
        B[name] = {"n_configs_min_trades": int(ok.sum()), "dominant": {"n": int(nn[dom]), "avg_r": round(float(avg[dom]), 3),
                   "t": round(float(t[dom]), 2), "rank_by_avg_r": rank_avg, "rank_by_t": rank_t},
                   "max_t": round(float(t[ok].max()), 2), "n_t_ge_3": int((t[ok] >= 3).sum()),
                   "median_avg_r": round(float(np.median(avg[ok])), 3), "share_avg_r_pos": round(float((avg[ok] > 0).mean()), 3),
                   "long_only_median_avg_r": round(float(np.median(avg[ok & np.array([c['sides'] == 'long' for c in configs])])), 3)}
    # corner check: is the dominant at the edge of the space in each numeric dimension?
    B["dominant_at_edge_of_space"] = {"rsi_len 21 (max of 7..21)": True, "lb 8 (max of 2..8)": True, "rr 4 (max of 1..4)": True,
                                      "buffer_atr 0 (min of 0..1)": True, "max_hold 60 (max of 12..60)": True}
    out["B_cross_section"] = B

    # C. actual OOS sequence, expanding / avg R
    meta = pickle.loads((OUT / "trades_meta.pkl").read_bytes())
    df = load_df("4h")
    yrs = df.index.year.to_numpy()
    picks = A["window_expanding_avg_r"]["picks"]
    rows = []
    for y, p in picks.items():
        tr = meta["trades"][p]
        rows.append(tr[yrs[tr[:, 0].astype(int)] == y])
    tr = np.vstack(rows)
    tr = tr[np.argsort(tr[:, 0])]
    overl = int((tr[1:, 1] <= tr[:-1, 2]).sum())
    r = tr[:, 5]
    eq = np.r_[1, np.cumprod(1 + 0.01 * r)]
    cumr = np.r_[0, np.cumsum(r)]
    ddr = float((cumr - np.maximum.accumulate(cumr)).min())
    streak = best = 0
    for x in r < 0:
        streak = streak + 1 if x else 0
        best = max(best, streak)
    i0 = int(df.index.searchsorted(pd.Timestamp("2017-01-01")))
    tim = float((tr[:, 2] - tr[:, 1] + 1).sum() / (len(df) - i0))
    # underwater: longest stretch (calendar days) between equity highs, closed-trade
    exit_t = df.index[tr[:, 2].astype(int)]
    peak_t, longest = df.index[i0], pd.Timedelta(0)
    hi_eq = 1.0
    for e, t in zip(eq[1:], exit_t):
        if e >= hi_eq:
            longest = max(longest, t - peak_t)
            hi_eq, peak_t = e, t
    longest = max(longest, df.index[-1] - peak_t)
    span = (df.index[-1] - df.index[i0]).total_seconds() / (365.25 * 86400)
    d = load_df("1d")
    full_sma = d["close"].rolling(50).mean()
    d2 = d.loc["2017-01-01":]
    dc, do = d2["close"].to_numpy(float), d2["open"].to_numpy(float)
    sma = full_sma.loc["2017-01-01":].to_numpy()
    ret = np.r_[dc[0] / do[0] - 1, dc[1:] / dc[:-1] - 1]
    pos = np.r_[0, (dc > sma)[:-1].astype(float)]
    sw = np.abs(np.diff(np.r_[0, pos]))
    bh = dc / do[0]
    smaq = np.cumprod(1 + pos * ret - sw * COST)
    target = abs(max_dd(eq))

    def matched(stream):
        lo_f, hi_f = 0.0, 1.0
        for _ in range(60):
            f = (lo_f + hi_f) / 2
            if abs(max_dd(np.cumprod(1 + f * stream))) > target:
                hi_f = f
            else:
                lo_f = f
        e = np.cumprod(1 + lo_f * stream)
        return {"fraction_pct": round(100 * lo_f, 1), "total_return_pct": round(100 * (e[-1] - 1), 1),
                "cagr_pct": round(100 * (e[-1] ** (1 / span) - 1), 2), "max_dd_pct": round(100 * max_dd(e), 1)}

    out["C_oos_sequence_expanding_avg_r"] = {
        "n": int(len(r)), "overlapping_trades": overl, "trades_per_year": round(len(r) / span, 1),
        "win_rate_pct": round(100 * float((r > 0).mean()), 1), "longest_losing_streak": int(best),
        "total_return_pct_1pct_risk": round(100 * (eq[-1] - 1), 1),
        "cagr_pct_1pct_risk": round(100 * (eq[-1] ** (1 / span) - 1), 2),
        "max_dd_closed_pct_1pct_risk": round(100 * max_dd(eq), 1), "max_dd_R": round(ddr, 2),
        "longest_without_new_equity_high_days": int(longest.days), "time_in_market_pct": round(100 * tim, 1),
        "ex2017_total_return_pct": round(100 * (float(np.prod(1 + 0.01 * r[yrs[tr[:, 0].astype(int)] >= 2018])) - 1), 1),
        "btc_buy_hold_2017_2026": {"total_return_pct": round(100 * (bh[-1] - 1), 1), "max_dd_pct": round(100 * max_dd(bh), 1)},
        "btc_sma50_long_flat_2017_2026": {"total_return_pct": round(100 * (smaq[-1] - 1), 1), "max_dd_pct": round(100 * max_dd(smaq), 1),
                                          "time_in_market_pct": round(100 * float(pos.mean()), 1)},
        "btc_rebalanced_fraction_same_dd": matched(ret), "sma50_fraction_same_dd": matched(pos * ret - sw * COST)}
    print(json.dumps({k: out[k] for k in ("A_picks", "B_cross_section", "C_oos_sequence_expanding_avg_r")}, indent=1), flush=True)
    (OUT / "c4_extras.json").write_text(json.dumps(out, indent=1))

    # D. placebo universes
    if n_draws:
        with ProcessPoolExecutor(2, initializer=c1._init_placebo) as ex:
            res_keep = list(ex.map(placebo, [(20_000 + i, False) for i in range(n_draws)], chunksize=4))
            res_long = list(ex.map(placebo, [(30_000 + i, True) for i in range(n_draws)], chunksize=4))
        nn, avg, t, ok = xsec(N, S, Q, np.arange(13), 100)
        k = int(np.argmax(np.where(ok, avg, -np.inf)))
        real = {"full_max_t": round(float(t[ok].max()), 2), "full_best_avg_r": round(float(avg[k]), 3),
                "dominant_full_t": round(float(t[dom]), 2)}
        nn2, avg2, t2, ok2 = xsec(N, S, Q, np.arange(3, 13), 100)
        real["oos_window_max_t"] = round(float(t2[ok2].max()), 2)
        D = {"n_draws": n_draws, "real": real}
        for name, res in (("keep_side", res_keep), ("all_long", res_long)):
            mt = np.array([x["full_2014_2026"]["max_t"] for x in res])
            mt2 = np.array([x["oos_2017_2026"]["max_t"] for x in res])
            ba = np.array([x["full_2014_2026"]["best_avg_r"] for x in res])
            rep_t = np.array([[z[2] for z in x["rep"]] for x in res])
            rep_a = np.array([[z[1] for z in x["rep"]] for x in res])
            exf = np.array([x["exp_avg_r_ex_first"]["avg_r"] for x in res])
            D[name] = {
                "full_max_t_over_configs_quantiles": {q: round(float(np.quantile(mt, q)), 2) for q in (0.05, 0.5, 0.95)},
                "P_full_max_t_ge_real_max": round(float((mt >= real["full_max_t"]).mean()), 4),
                "P_full_max_t_ge_dominant_t": round(float((mt >= real["dominant_full_t"]).mean()), 4),
                "oos_window_max_t_quantiles": {q: round(float(np.quantile(mt2, q)), 2) for q in (0.05, 0.5, 0.95)},
                "P_oos_window_max_t_ge_real": round(float((mt2 >= real["oos_window_max_t"]).mean()), 4),
                "full_best_avg_r_quantiles": {q: round(float(np.quantile(ba, q)), 3) for q in (0.05, 0.5, 0.95)},
                "P_full_best_avg_r_ge_real": round(float((ba >= real["full_best_avg_r"]).mean()), 4),
                "median_full_avg_r_over_configs": round(float(np.median([x["full_2014_2026"]["median_avg_r"] for x in res])), 3),
                "wf_expanding_avg_r_t_quantiles": {q: round(float(np.quantile(rep_t[:, 2], q)), 2) for q in (0.5, 0.9, 0.95, 0.99)},
                "wf_expanding_avg_r_avg_quantiles": {q: round(float(np.quantile(rep_a[:, 2], q)), 3) for q in (0.5, 0.9, 0.95)},
                "P_wf_expanding_avg_r_t_ge_2.38": round(float((rep_t[:, 2] >= 2.38).mean()), 4),
                "P_wf_expanding_avg_r_avg_ge_0.581": round(float((rep_a[:, 2] >= 0.581).mean()), 4),
                "P_best_of_4_t_ge_2.38": round(float((rep_t.max(1) >= 2.38).mean()), 4),
                "best_of_4_t_quantiles": {q: round(float(np.quantile(rep_t.max(1), q)), 2) for q in (0.5, 0.9, 0.95, 0.99)},
                "P_wf_expanding_ex_first_year_avg_ge_0.237": round(float((exf >= 0.237).mean()), 4),
                "wf_expanding_ex_first_year_avg_quantiles": {q: round(float(np.quantile(exf, q)), 3) for q in (0.5, 0.9, 0.95)},
            }
        out["D_placebo_extras"] = D
        print(json.dumps(D, indent=1), flush=True)
        (OUT / "c4_extras.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
