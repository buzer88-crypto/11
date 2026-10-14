"""Check 3: neighbourhood of the dominant pick on 4h.

One-at-a-time changes (every grid value of tune.SPACE plus some off-grid values, and asymmetric
pivots), two 2-D slices (rsi_len x lb, rr x max_hold), and a placebo: long trades at random 4h
bars of the same year with the dominant config's risk sizes, rr 4, max hold 60, same costs.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUTD = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim import YEARS, df_for, evaluate, row  # noqa: E402

from pinetest.events import Setup, simulate  # noqa: E402

DOM = {"kind": "hidden", "rsi_len": 21, "lb": 8, "rr": 4.0, "buffer_atr": 0.0, "max_hold": 60,
       "confirm": "ema_9_21", "trend": False, "sides": "long", "extreme": False}
ONE_AT_A_TIME = {
    "kind": ["regular"],
    "rsi_len": [7, 9, 14, 18, 25, 28],
    "lb": [2, 3, 5, 6, 7, 9, 10, 12],
    "rr": [1.0, 1.5, 2.0, 3.0, 3.5, 5.0, 6.0],
    "buffer_atr": [0.1, 0.25, 0.5, 1.0],
    "max_hold": [12, 30, 45, 90, 120],
    "confirm": ["none", "macd_cross", "ema50", "ema20_reclaim", "macd_hist", "macd_div"],
    "trend": [True], "sides": ["both"], "extreme": [True],
    "range_upper": [40, 80], "left": [5, 6, 10], "right": [5, 6, 10],
}
OFF_GRID = {"rsi_len": {18, 25, 28}, "lb": {6, 7, 9, 10, 12}, "rr": {3.5, 5.0, 6.0}, "buffer_atr": {0.1},
            "max_hold": {45, 90, 120}, "confirm": {"macd_hist", "macd_div"}, "range_upper": {40, 80},
            "left": {5, 6, 10}, "right": {5, 6, 10}}


def _eval(cfg):
    return evaluate(cfg, "4h")


def placebo(risk_pct, n_draws=300, seed=11):
    """Long market entries at random bars of the same calendar year as each real trade, with that
    trade's risk as % of entry, rr 4, max hold 60; one at a time as in simulate. Avg R per draw."""
    df = df_for("4h")
    o = df["open"].to_numpy()
    years = df.index.year.to_numpy()
    rng = np.random.default_rng(seed)
    year_bars = {y: np.flatnonzero(years == y) for y in YEARS}
    out = {"all": [], "2017_2026": [], "2018_2026": []}
    for _ in range(n_draws):
        setups = []
        for y, rp in risk_pct:
            bars = year_bars[y]
            b = int(bars[rng.integers(len(bars) - 1)])
            if b + 1 >= len(o):
                continue
            ref = o[b + 1]
            setups.append(Setup(b, 1, "market", ref * (1 - rp), ref * (1 + 4.0 * rp), max_hold=60))
        tr = simulate(df, setups, 0.1)
        rs = {y: [] for y in YEARS}
        for t in tr:
            rs[int(years[t.bar])].append(t.r)
        for k, ys in (("all", range(2014, 2027)), ("2017_2026", range(2017, 2027)), ("2018_2026", range(2018, 2027))):
            v = [x for y in ys for x in rs[y]]
            out[k].append(float(np.mean(v)) if v else 0.0)
    return {k: np.array(v) for k, v in out.items()}


def main():
    jobs = [("dominant", None, None, DOM)]
    for k, vals in ONE_AT_A_TIME.items():
        for v in vals:
            jobs.append(("one", k, v, {**DOM, k: v}))
    for rl in (7, 9, 14, 18, 21, 25):
        for lb in (2, 3, 5, 6, 8, 10):
            jobs.append(("rsi_x_lb", "rsi_len,lb", (rl, lb), {**DOM, "rsi_len": rl, "lb": lb}))
    for rr in (1.0, 2.0, 3.0, 4.0, 5.0):
        for mh in (12, 30, 60, 90):
            jobs.append(("rr_x_hold", "rr,max_hold", (rr, mh), {**DOM, "rr": rr, "max_hold": mh}))
    import pickle
    cache = OUTD / "check3_evals.pkl"
    if cache.exists():
        res = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor(1) as ex:
            res = list(ex.map(_eval, [j[3] for j in jobs], chunksize=4))
        cache.write_bytes(pickle.dumps(res))
    rows = []
    for (grp, k, v, cfg), by in zip(jobs, res):
        r = row(by)
        rows.append({"group": grp, "param": k, "value": v, "off_grid": bool(k in OFF_GRID and v in OFF_GRID[k]), **r})
        p = r["per_year"]
        print(grp, k, v, "| 14-16", r["2014_2016"]["n"], r["2014_2016"]["avg_r"], "| 17-26", r["2017_2026"]["n"],
              r["2017_2026"]["avg_r"], "| 18-26", r["2018_2026"]["n"], r["2018_2026"]["avg_r"], r["2018_2026"]["t"],
              "|", " ".join(f"{y % 100}:{p[y]['avg_r']}" for y in YEARS), flush=True)

    # Placebo for the dominant config's trades (stops recomputed from its setups).
    from pinetest.divergence import rsi_divergences
    df = df_for("4h")
    setups = rsi_divergences(df, kind="hidden", rsi_len=21, left=8, right=8, rr=4.0, buffer_atr=0.0, max_hold=60,
                             confirm="ema_9_21", trend=False, extreme=False, sides=(1,))
    stop_by_bar = {s.bar: s.stop for s in setups}
    by, trades = evaluate(DOM, "4h", with_trades=True)
    years = df.index.year.to_numpy()
    risk_pct = [(int(years[t.bar]), (t.entry - stop_by_bar[t.bar]) / t.entry) for t in trades
                if int(years[t.bar]) in YEARS]
    pl = placebo(risk_pct)
    dom_row = rows[0]
    plac = {}
    for k, real_key in (("all", "2014_2026"), ("2017_2026", "2017_2026"), ("2018_2026", "2018_2026")):
        v = pl[k]
        plac[k] = {"placebo_median_avg_r": round(float(np.median(v)), 3), "placebo_p95": round(float(np.quantile(v, .95)), 3),
                   "real_avg_r": dom_row[real_key]["avg_r"],
                   "pct_placebo_below_real": round(100 * float(np.mean(v < dom_row[real_key]["avg_r"])), 1)}
    print("placebo:", json.dumps(plac))

    one_all = [r for r in rows if r["group"] == "one"]
    one = [r for r in one_all if r["2018_2026"]["avg_r"] is not None]
    one_grid = [r for r in one if not r["off_grid"]]
    summ = {
        "dominant": {k: dom_row[k] for k in ("2014_2016", "2017_2026", "2018_2026")},
        "one_at_a_time_n": len(one_all),
        "one_at_a_time_no_trades": [f"{r['param']}={r['value']}" for r in one_all if r["2018_2026"]["avg_r"] is None],
        "one_at_a_time_2018_2026_avg_r": {"median": round(float(np.median([r["2018_2026"]["avg_r"] for r in one])), 3),
                                          "share_positive_pct": round(100 * float(np.mean([r["2018_2026"]["avg_r"] > 0 for r in one])), 1),
                                          "share_ge_half_dominant_pct": round(100 * float(np.mean([r["2018_2026"]["avg_r"] >= dom_row["2018_2026"]["avg_r"] / 2 for r in one])), 1)},
        "one_at_a_time_grid_only_2018_2026_avg_r_median": round(float(np.median([r["2018_2026"]["avg_r"] for r in one_grid])), 3),
        "one_at_a_time_2017_2026_avg_r_median": round(float(np.median([r["2017_2026"]["avg_r"] for r in one])), 3),
        "one_at_a_time_2018_2026_signal_params": {f"{r['param']}={r['value']}": r["2018_2026"]["avg_r"] for r in one
                                                  if r["param"] in ("kind", "rsi_len", "lb", "confirm", "sides", "left", "right")},
        "one_at_a_time_2018_2026_signal_params_median": round(float(np.median([r["2018_2026"]["avg_r"] for r in one
                                                  if r["param"] in ("kind", "rsi_len", "lb", "confirm", "sides", "left", "right")])), 3),
        "one_at_a_time_2018_2026_management_params_median": round(float(np.median([r["2018_2026"]["avg_r"] for r in one
                                                  if r["param"] in ("rr", "buffer_atr", "max_hold", "trend", "range_upper")])), 3),
        "placebo_random_long_rr4_hold60": plac,
    }
    for g in ("rsi_x_lb", "rr_x_hold"):
        v = [r["2018_2026"]["avg_r"] for r in rows if r["group"] == g]
        summ[f"{g}_2018_2026"] = {"median": round(float(np.median(v)), 3), "share_positive_pct": round(100 * float(np.mean(np.array(v) > 0)), 1)}
    (OUTD / "check3_neighbourhood.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
