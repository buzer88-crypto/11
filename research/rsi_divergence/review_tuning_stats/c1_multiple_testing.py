"""Check 1: researcher degrees of freedom and the chance of a best-of-4 walk-forward t >= 2.38.

A. Reproduce the 4 reported walk-forward variants from tune_cache.pkl with a vectorised walk-forward.
B. Garden of forking paths on the real data: 60 nearby walk-forward variants
   (window 2/3/4/5/expanding x metric avg_r/t/total_r x min_trades 10/20/30/50).
C. Null 1, year-label shuffle: permute the 13 calendar-year columns (one permutation for all 2400
   configs, so the cross-section and each year's market are kept, only the time order is broken),
   rerun the 4 reported variants and the 60-variant family; record max t.
D. Null 2, placebo universe: every one of the 2400 configs is replaced by random-timed trades: each
   real trade keeps its side, its risk as % of entry, its config's rr and max_hold and its calendar
   year, but the signal bar is drawn uniformly within that year; one trade at a time (a trade that
   would start inside the previous one is pushed to the bar after its exit). Same 4 variants + family.
Output: c1_multiple_testing.json.  PYTHONPATH=. python .../c1_multiple_testing.py [n_placebo] [n_perm]
"""
import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from wf_lib import (OUT, REPORTED, TEST_YEARS, YEARS, arrays_from_results, load_cache, load_df,
                    oos_stats, sim_market, walk_forward)

TARGET_T = 2.38
FAMILY = [(w, m, mt) for w in (2, 3, 4, 5, 0) for m in ("avg_r", "t", "total_r") for mt in (10, 20, 30, 50)]


def run_variants(N, S, Q):
    rep = [oos_stats(walk_forward(N, S, Q, w, m)[1]) for w, m in REPORTED]
    fam = [oos_stats(walk_forward(N, S, Q, w, m, mt)[1]) for w, m, mt in FAMILY]
    return rep, fam


def p_norm_one_sided(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


# ---------------- placebo universe ----------------
_G = {}


def _init_placebo():
    df = load_df("4h")
    _G["o"], _G["h"], _G["l"], _G["c"] = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    yrs = df.index.year.to_numpy()
    _G["yrs"] = yrs
    meta = pickle.loads((OUT / "trades_meta.pkl").read_bytes())
    cfgs = meta["configs"]
    rows = []
    for ci, tr in enumerate(meta["trades"]):
        if len(tr) == 0:
            continue
        cid = np.full(len(tr), ci)
        rows.append(np.column_stack([cid, tr[:, 0], tr[:, 3], tr[:, 4],
                                     np.full(len(tr), cfgs[ci]["rr"]), np.full(len(tr), cfgs[ci]["max_hold"])]))
    allt = np.vstack(rows)
    ty = yrs[allt[:, 1].astype(int)]
    keep = (ty >= YEARS[0]) & (ty <= YEARS[-1])
    _G["T"] = allt[keep]
    _G["ty"] = ty[keep]
    n = len(_G["c"])
    _G["ystart"] = {int(y): int(np.searchsorted(yrs, y)) for y in YEARS}
    _G["yend"] = {int(y): min(int(np.searchsorted(yrs, y + 1)), n - 2) for y in YEARS}   # exclusive; bar+1 < n
    _G["nconf"] = len(cfgs)


def placebo_universe(seed):
    o, h, l, c = _G["o"], _G["h"], _G["l"], _G["c"]
    T, ty = _G["T"], _G["ty"]
    n = len(c)
    rng = np.random.default_rng(seed)
    lo = np.array([_G["ystart"][int(y)] for y in ty])
    hi = np.array([_G["yend"][int(y)] for y in ty])
    bar = lo + (rng.random(len(T)) * (hi - lo)).astype(int)
    cid = T[:, 0].astype(int)
    order = np.lexsort((bar, cid))
    cid, bar = cid[order], bar[order]
    side, risk, rr, hold = T[order, 2], T[order, 3], T[order, 4], T[order, 5].astype(int)
    alive = np.ones(len(bar), bool)
    exit_bar, r = sim_market(o, h, l, c, bar + 1, side, risk, rr, hold)
    same = np.r_[False, cid[1:] == cid[:-1]]
    for _ in range(200):
        # previous alive trade's exit within the same config
        prev_exit = np.full(len(bar), -1)
        # forward-fill exit of previous alive trade per config
        last_exit = np.where(alive, exit_bar, -1)
        # running max of exits within config (exits are increasing along alive trades)
        grp_start = ~same
        tmp = last_exit.copy()
        # cumulative max within groups
        idx = np.arange(len(tmp))
        gid = np.cumsum(grp_start) - 1
        cm = np.maximum.accumulate(np.where(True, tmp + (gid * (n + 10)), 0))
        cm = cm - gid * (n + 10)
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
    yi = _G["yrs"][np.minimum(bar, n - 1)] - YEARS[0]
    ok = alive & (yi >= 0) & (yi < len(YEARS))
    C, Y = _G["nconf"], len(YEARS)
    N, S, Q = np.zeros((C, Y)), np.zeros((C, Y)), np.zeros((C, Y))
    np.add.at(N, (cid[ok], yi[ok]), 1)
    np.add.at(S, (cid[ok], yi[ok]), r[ok])
    np.add.at(Q, (cid[ok], yi[ok]), r[ok] ** 2)
    rep, fam = run_variants(N, S, Q)
    allr = r[ok]
    longs = side[ok] > 0
    return {"rep": [(x["n"], x["avg_r"], x["t"]) for x in rep], "fam": [(x["n"], x["avg_r"], x["t"]) for x in fam],
            "universe_avg_r": float(allr.mean()), "universe_long_avg_r": float(allr[longs].mean())}


def verify_sim():
    """The vectorised simulator must reproduce the real trades' R from their real bars."""
    _init_placebo()
    T = _G["T"]
    o, h, l, c = _G["o"], _G["h"], _G["l"], _G["c"]
    meta = pickle.loads((OUT / "trades_meta.pkl").read_bytes())
    real_r = np.concatenate([tr[:, 5] for tr in meta["trades"] if len(tr)])
    real_bar = np.concatenate([tr[:, 0] for tr in meta["trades"] if len(tr)])
    yrs = _G["yrs"]
    ty = yrs[real_bar.astype(int)]
    keep = (ty >= YEARS[0]) & (ty <= YEARS[-1])
    eb, r = sim_market(o, h, l, c, T[:, 1].astype(int) + 1, T[:, 2], T[:, 3], T[:, 4], T[:, 5].astype(int))
    d = np.abs(r - real_r[keep])
    return {"n_trades": int(len(r)), "max_abs_diff_r": float(d.max()), "n_diff_gt_1e-6": int((d > 1e-6).sum())}


def summarize_null(rows_rep, rows_fam, real_rep, real_fam):
    rep_t = np.array([[x[2] for x in row] for row in rows_rep])
    fam_t = np.array([[x[2] for x in row] for row in rows_fam])
    rep_avg = np.array([[x[1] for x in row] for row in rows_rep])
    max4 = rep_t.max(1)
    real_max_fam = max(x["t"] for x in real_fam)
    return {
        "n_draws": int(len(rep_t)),
        "per_variant_null_median_t": {f"{w or 'exp'}_{m}": round(float(np.median(rep_t[:, k])), 2) for k, (w, m) in enumerate(REPORTED)},
        "per_variant_null_median_avg_r": {f"{w or 'exp'}_{m}": round(float(np.median(rep_avg[:, k])), 3) for k, (w, m) in enumerate(REPORTED)},
        "P_expanding_avg_r_t_ge_2.38": round(float((rep_t[:, 2] >= TARGET_T).mean()), 4),
        "P_expanding_avg_r_avgR_ge_0.581": round(float((rep_avg[:, 2] >= 0.581).mean()), 4),
        "P_best_of_4_t_ge_2.38": round(float((max4 >= TARGET_T).mean()), 4),
        "best_of_4_t_quantiles": {p: round(float(np.quantile(max4, p)), 2) for p in (0.5, 0.9, 0.95, 0.99)},
        "family60_max_t_quantiles": {p: round(float(np.quantile(fam_t.max(1), p)), 2) for p in (0.5, 0.9, 0.95, 0.99)},
        "P_family60_max_t_ge_real_family_max": round(float((fam_t.max(1) >= real_max_fam).mean()), 4),
        "P_family60_max_t_ge_2.38": round(float((fam_t.max(1) >= TARGET_T).mean()), 4),
        "family60_null_median_of_variant_medians_t": round(float(np.median(np.median(fam_t, 1))), 2),
    }


def main():
    n_placebo = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    cache = load_cache()
    configs, results = cache["configs"], cache["results"]
    N, S, Q = arrays_from_results(results)
    tr = json.loads((OUT.parent / "tune_results.json").read_text())
    out = {}

    # A. reproduce
    repro = {}
    for w, m in REPORTED:
        picks, nsq = walk_forward(N, S, Q, w, m)
        key = f"window_{w or 'expanding'}_{m}"
        st = oos_stats(nsq)
        cfg_match = all(configs[p] == tr["walk_forward"][key]["per_year"][str(y)]["config"] for p, y in zip(picks, TEST_YEARS))
        ex = [(N[p, yi], S[p, yi], Q[p, yi]) for p, yi in zip(picks, range(3, 13)) if yi != 3]
        ex = oos_stats(tuple(np.sum(ex, 0)))
        repro[key] = {"n": st["n"], "avg_r": round(st["avg_r"], 3), "t": round(st["t"], 2), "picks_match_reported": cfg_match,
                      "ex2017": {k: round(v, 3) for k, v in ex.items()},
                      "one_sided_p_normal": round(p_norm_one_sided(st["t"]), 4)}
    out["A_reproduce"] = repro
    p1 = p_norm_one_sided(2.38)
    out["A_naive_adjustment"] = {
        "p_one_sided_t2.38_normal": round(p1, 4),
        "sidak_4_variants": round(1 - (1 - p1) ** 4, 4),
        "note": "Sidak treats the 4 variants as independent; they share configs/years, so this is a rough upper bound "
                "on the adjustment; the simulated nulls below handle the dependence."}

    # B. garden of forking paths on real data
    real_rep, real_fam = run_variants(N, S, Q)
    fam_rows = []
    for (w, m, mt), st in zip(FAMILY, real_fam):
        fam_rows.append({"window": w or "expanding", "metric": m, "min_trades": mt, "n": st["n"],
                         "avg_r": round(st["avg_r"], 3), "t": round(st["t"], 2)})
    ts = np.array([x["t"] for x in real_fam])
    avgs = np.array([x["avg_r"] for x in real_fam])
    # same family without 2017 (test years 2018..2026)
    fam_ex = []
    for w, m, mt in FAMILY:
        picks, _ = walk_forward(N, S, Q, w, m, mt)
        ex = [(N[p, yi], S[p, yi], Q[p, yi]) for p, yi in zip(picks, range(3, 13)) if yi != 3 and p >= 0]
        fam_ex.append(oos_stats(tuple(np.sum(ex, 0))))
    out["B_family60_real"] = {
        "n_variants": len(FAMILY), "median_t": round(float(np.median(ts)), 2), "max_t": round(float(ts.max()), 2),
        "min_t": round(float(ts.min()), 2), "n_t_ge_2.38": int((ts >= 2.38).sum()), "n_t_ge_1.96": int((ts >= 1.96).sum()),
        "n_avg_r_positive": int((avgs > 0).sum()), "median_avg_r": round(float(np.median(avgs)), 3),
        "ex2017_median_avg_r": round(float(np.median([x["avg_r"] for x in fam_ex])), 3),
        "ex2017_median_t": round(float(np.median([x["t"] for x in fam_ex])), 2),
        "ex2017_max_t": round(float(max(x["t"] for x in fam_ex)), 2),
        "ex2017_n_t_ge_1.96": int(sum(x["t"] >= 1.96 for x in fam_ex)),
        "rows": fam_rows}

    # C. year-label shuffle
    t0 = time.time()
    rng = np.random.default_rng(777)
    rows_rep, rows_fam = [], []
    for _ in range(n_perm):
        perm = rng.permutation(len(YEARS))
        rep, fam = run_variants(N[:, perm], S[:, perm], Q[:, perm])
        rows_rep.append([(x["n"], x["avg_r"], x["t"]) for x in rep])
        rows_fam.append([(x["n"], x["avg_r"], x["t"]) for x in fam])
    out["C_year_shuffle_null"] = summarize_null(rows_rep, rows_fam, real_rep, real_fam)
    out["C_year_shuffle_null"]["secs"] = round(time.time() - t0, 1)
    (OUT / "c1_multiple_testing.json").write_text(json.dumps(out, indent=1))
    print("C done", json.dumps(out["C_year_shuffle_null"], indent=1), flush=True)

    # D. placebo universe
    out["D_sim_check"] = verify_sim()
    print("sim check", out["D_sim_check"], flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(2, initializer=_init_placebo) as ex:
        res = list(ex.map(placebo_universe, range(10_000, 10_000 + n_placebo), chunksize=4))
    out["D_placebo_universe_null"] = summarize_null([x["rep"] for x in res], [x["fam"] for x in res], real_rep, real_fam)
    ua = np.array([x["universe_avg_r"] for x in res])
    ul = np.array([x["universe_long_avg_r"] for x in res])
    real_all = np.concatenate([np.concatenate([np.asarray(v, float) for v in by.values()]) for by in results])
    out["D_placebo_universe_null"]["placebo_all_trades_avg_r_mean"] = round(float(ua.mean()), 3)
    out["D_placebo_universe_null"]["placebo_long_trades_avg_r_mean"] = round(float(ul.mean()), 3)
    out["D_placebo_universe_null"]["real_all_trades_avg_r"] = round(float(real_all.mean()), 3)
    out["D_placebo_universe_null"]["secs"] = round(time.time() - t0, 1)
    (OUT / "c1_multiple_testing.json").write_text(json.dumps(out, indent=1))
    print("D done", json.dumps(out["D_placebo_universe_null"], indent=1), flush=True)


if __name__ == "__main__":
    main()
