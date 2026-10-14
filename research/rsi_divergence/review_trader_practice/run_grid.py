"""Pre-registered trader-rule grid (see PREREG.txt). Writes grid_rows.csv and results_grid.json."""
import itertools, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import tp_lib as T

OUT = Path(__file__).parent
SEED = 20261014
COSTS = (0.1, 0.05)
DET = ["rsi_pivstop", "rsi_swingstop", "price_pivot"]
LBS = [3, 5]
ENTRIES = ["next_open", "confirm"]
P = list(T.PERIODS)


def main():
    t0 = time.time()
    D = T.Data(T.load("4h"))
    rows, trades_of = [], {}
    for det, lb, setup, entry, ex in itertools.product(DET, LBS, T.SETUPS, ENTRIES, T.EXITS):
        cands = T.candidates(D, det, lb, setup)
        tr = T.simulate(D, cands, entry, ex, det, lb)
        key = (det, lb, setup, entry, ex)
        trades_of[key] = tr
        per = T.by_period(D, tr)
        row = dict(zip(["detector", "lb", "setup", "entry", "exit"], key))
        for cost in COSTS:
            for p in P:
                s = T.stats([t.r(cost) for t in per[p]])
                for k, v in s.items():
                    row[f"{p}_{k}_c{cost}"] = v
        rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "grid_rows.csv", index=False)
    print("grid done", len(g), round(time.time() - t0, 1), "s")
    res = {"n_configs": len(g)}

    # --- grid summaries
    summ = {}
    for cost in COSTS:
        c = f"c{cost}"
        pos = {p: g[f"{p}_avg_r_{c}"].gt(0) & g[f"{p}_n_{c}"].gt(0) for p in P}
        all3 = pos["pre"] & pos["train"] & pos["test"]
        summ[c] = {
            **{f"{p}_median_avg_r": round(float(g[f"{p}_avg_r_{c}"].median()), 3) for p in P},
            **{f"{p}_n_positive": int(pos[p].sum()) for p in P},
            "n_positive_all_three": int(all3.sum()),
            "all_three_configs": g.loc[all3, ["detector", "lb", "setup", "entry", "exit", f"pre_avg_r_{c}",
                                              f"train_avg_r_{c}", f"test_avg_r_{c}", f"test_n_{c}", f"test_t_{c}"]].to_dict("records"),
            "spearman_train_test": round(float(g[f"train_avg_r_{c}"].rank().corr(g[f"test_avg_r_{c}"].rank())), 3),
            "spearman_pre_test": round(float(g[f"pre_avg_r_{c}"].rank().corr(g[f"test_avg_r_{c}"].rank())), 3),
            "test_avg_r_quantiles": {q: round(float(g[f"test_avg_r_{c}"].quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "marginal_median_avg_r": {
                dim: {str(v): {p: round(float(sub[f"{p}_avg_r_{c}"].median()), 3) for p in P}
                      for v, sub in g.groupby(dim)}
                for dim in ("detector", "lb", "setup", "entry", "exit")},
        }
    res["grid_summary"] = summ

    # --- selection on TRAIN
    rng = np.random.default_rng(SEED)
    picks = {}
    for cost in COSTS:
        c = f"c{cost}"
        el = g[g[f"train_n_{c}"] >= 30]
        for crit in ("avg_r", "sharpe"):
            r = el.loc[el[f"train_{crit}_{c}"].idxmax()]
            picks[f"{crit}_c{cost}"] = tuple(r[["detector", "lb", "setup", "entry", "exit"]])
    res["picks"] = {k: list(map(lambda x: x if isinstance(x, str) else int(x), v)) for k, v in picks.items()}
    # top-10 on TRAIN by avg R (0.1%) and how they do later
    el = g[g["train_n_c0.1"] >= 30].sort_values("train_avg_r_c0.1", ascending=False).head(10)
    res["top10_train_avg_r"] = el[["detector", "lb", "setup", "entry", "exit", "pre_avg_r_c0.1", "train_avg_r_c0.1",
                                   "train_n_c0.1", "test_avg_r_c0.1", "test_n_c0.1", "test_t_c0.1"]].to_dict("records")

    detail = {}
    for key in dict.fromkeys(picks.values()):
        det, lb, setup, entry, ex = key
        tr = trades_of[key]
        per = T.by_period(D, tr)
        d = {"config": list(map(str, key))}
        for cost in COSTS:
            d[f"cost_{cost}"] = {p: T.stats([t.r(cost) for t in per[p]]) for p in P}
        for side, nm in ((1, "long"), (-1, "short")):
            d[nm] = {p: T.stats([t.r(0.1) for t in per[p] if t.side == side]) for p in P}
        d["test_years"] = {str(y): T.stats([t.r(0.1) for t in per["test"] if D.idx[t.bar].year == y])
                           for y in sorted({D.idx[t.bar].year for t in per["test"]})}
        d["exit_reasons_test"] = pd.Series([t.reason for t in per["test"]]).value_counts().to_dict()
        d["median_bars_held_test"] = float(np.median([t.exit_bar - t.fill + 1 for t in per["test"]])) if per["test"] else None
        yrs = (D.idx[-1] - pd.Timestamp("2022-01-01")).days / 365.25
        st = T.stats([t.r(0.1) for t in per["test"]])
        d["test_pct_per_year_at_1pct_risk"] = round(st["avg_r"] * st["n"] / yrs, 1)
        pl = {}
        for p in P:
            real = per[p]
            if not real:
                continue
            vals = {c: [] for c in COSTS}
            for _ in range(300):
                ptr = T.placebo(D, real, ex, det, lb, rng)
                for c in COSTS:
                    vals[c].append(T.stats([t.r(c) for t in ptr])["avg_r"])
            pl[p] = {}
            for c in COSTS:
                v = np.array(vals[c])
                rr = T.stats([t.r(c) for t in real])["avg_r"]
                pl[p][f"c{c}"] = {"real_avg_r": rr, "pct_placebo_below_real": round(100 * float((v < rr).mean()), 1),
                                  "placebo_median": round(float(np.median(v)), 3),
                                  "placebo_p5_p95": [round(float(np.quantile(v, .05)), 3), round(float(np.quantile(v, .95)), 3)]}
        d["placebo_random_bars_same_side"] = pl
        detail["|".join(map(str, key))] = d
    res["pick_detail"] = detail
    print("picks done", round(time.time() - t0, 1), "s")

    # --- placebo version of the whole grid: count configs positive in all three periods
    reps = 20
    counts = {c: [] for c in COSTS}
    test_meds = {c: [] for c in COSTS}
    for rep in range(reps):
        cnt = {c: np.ones(len(rows), bool) for c in COSTS}
        tm = {c: [] for c in COSTS}
        for k_i, key in enumerate(trades_of):
            det, lb, setup, entry, ex = key
            per = T.by_period(D, trades_of[key])
            for p in P:
                ptr = T.placebo(D, per[p], ex, det, lb, rng)
                for c in COSTS:
                    s = T.stats([t.r(c) for t in ptr])
                    ok = s["n"] > 0 and s["avg_r"] > 0
                    cnt[c][k_i] &= ok
                    if p == "test":
                        tm[c].append(s["avg_r"])
        for c in COSTS:
            counts[c].append(int(cnt[c].sum()))
            test_meds[c].append(float(np.median(tm[c])))
        print("placebo grid rep", rep, {c: counts[c][-1] for c in COSTS}, round(time.time() - t0, 1), "s", flush=True)
    res["placebo_grid"] = {f"c{c}": {"reps": reps, "all_three_counts": counts[c],
                                     "all_three_median": float(np.median(counts[c])),
                                     "all_three_p95": float(np.quantile(counts[c], .95)),
                                     "test_median_avg_r_median": round(float(np.median(test_meds[c])), 3),
                                     "test_median_avg_r_range": [round(min(test_meds[c]), 3), round(max(test_meds[c]), 3)]}
                           for c in COSTS}

    (OUT / "results_grid.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("pick_detail",)}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
