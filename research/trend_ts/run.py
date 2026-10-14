"""Time-series trend following (absolute momentum) on the crypto panel.

Run:  cd /home/user/11 && python3 research/trend_ts/run.py

Hypotheses (all signals use data <= t only; the harness trades row t at close t+lag):
  (a) LF   per-asset long/flat on the point-in-time top-20 universe.
  (b) LS   per-asset long/short on the top-20 universe (shorts pay the harness short fee).
  (c) BTC  BTC-only trend, long/flat and long/short; extra out-of-sample 2014-2017.
  (d) VT   volatility-targeted long/flat (multi-asset and BTC), 40% annual target, gross <= 1.

Signals: "ret"  = past L-day return > 0  (up) / < 0 (down)
         "ma"   = close > L-day simple moving average (up) / < (down)

Weighting (multi-asset): "ew" equal, "iv" inverse 60-day volatility.
Sizing (LF only): "slot" = each universe member owns a fixed slice (1/N or its inverse-vol share)
                  and that slice sits in cash when the asset is not in an uptrend;
                  "full" = the whole book is spread over the assets that are on (cash only if none).

The grid is fixed up front and run in full.  The chosen config is the best TRAIN Sharpe
(2018-2021) over the whole grid; TEST (2022-) is only reported, never used for selection.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `pinetest`

from pinetest.crypto_data import load_panel, universe
from pinetest.portfolio import (DAYS_PER_YEAR, TEST, TRAIN, backtest, buy_and_hold,
                                circular_shift, permute_assets, rebalance_on)

OUT_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/claude-0/research-trend_ts")

SEED = 20260524
LOOKBACKS = (20, 50, 100, 200)
SIGNALS = ("ret", "ma")
REBALS = ("W", "D")             # W = weekly (Sunday close), D = every day
WEIGHTINGS = ("ew", "iv")
SIZINGS = ("slot", "full")
VOL_WIN = 60                    # days, for inverse-vol weights and the vol target estimate
VOL_TARGET = 0.40               # annualised
BTC_BENCH_TEST_SHARPE = 0.48    # BTC buy&hold TEST Sharpe (data/benchmarks.json: 0.479)
N_PLACEBO_CHOSEN = 500
N_PLACEBO_INFO = 200            # for the best config of each hypothesis (information only)
EXTRA_OOS = ("2014-01-01", "2017-12-31")
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)


# --------------------------------------------------------------------------------------------
# Signals and weights (every function maps data <= t to row t)
# --------------------------------------------------------------------------------------------

def trend_sign(px: pd.DataFrame, kind: str, lookback: int) -> pd.DataFrame:
    """+1 uptrend, -1 downtrend, 0 unknown (not enough history) or exactly flat."""
    if kind == "ret":
        x = px / px.shift(lookback) - 1
    elif kind == "ma":
        x = px / px.rolling(lookback, min_periods=lookback).mean() - 1
    else:
        raise ValueError(kind)
    return np.sign(x).fillna(0.0)


def rolling_vol(rets: pd.DataFrame, win: int = VOL_WIN) -> pd.DataFrame:
    """Annualised trailing volatility using returns up to and including t."""
    return rets.rolling(win, min_periods=win // 2).std() * np.sqrt(DAYS_PER_YEAR)


def base_weights(members: pd.DataFrame, vol: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Raw (unnormalised) per-asset budget for universe members."""
    if weighting == "ew":
        b = members.astype(float)
    elif weighting == "iv":
        b = (1.0 / vol).where(members & vol.gt(0)).fillna(0.0)
    else:
        raise ValueError(weighting)
    return b


def normalise_rows(x: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    return x.div(denom.replace(0, np.nan), axis=0).fillna(0.0)


def multi_long_flat(sign, members, vol, weighting, sizing):
    b = base_weights(members, vol, weighting)
    on = b * (sign > 0)
    denom = b.sum(axis=1) if sizing == "slot" else on.sum(axis=1)
    return normalise_rows(on, denom)


def multi_long_short(sign, members, vol, weighting):
    b = base_weights(members, vol, weighting)
    return normalise_rows(b * sign, b.sum(axis=1))      # gross = share of budget with a known sign


def btc_weights(sign, px, direction):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    s = sign["btc"].where(px["btc"].notna(), 0.0)
    w["btc"] = s.clip(lower=0.0) if direction == "lf" else s
    return w


def ex_ante_vol(w: pd.DataFrame, rets: pd.DataFrame, win: int = VOL_WIN) -> np.ndarray:
    """Annualised vol of today's weights applied to the last `win` daily returns (<= t)."""
    W = w.to_numpy()
    R = rets.fillna(0.0).to_numpy()
    out = np.full(len(W), np.nan)
    for t in range(win - 1, len(W)):
        nz = W[t] != 0
        if not nz.any():
            continue
        p = R[t - win + 1:t + 1][:, nz] @ W[t, nz]
        out[t] = p.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
    return out


def vol_target(w: pd.DataFrame, rets: pd.DataFrame, target: float = VOL_TARGET) -> pd.DataFrame:
    """Scale each row so ex-ante vol is `target`, never above gross 1."""
    sig = ex_ante_vol(w, rets)
    gross = w.abs().sum(axis=1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.minimum(target / sig, 1.0 / gross)
    k = np.where(np.isfinite(k), k, 0.0)
    return w.mul(k, axis=0)


def apply_rebal(w: pd.DataFrame, rebal: str) -> pd.DataFrame:
    return rebalance_on(w, "W-SUN") if rebal == "W" else w


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------

def build_grid():
    grid = []
    for sig, L, rb, wt, sz in itertools.product(SIGNALS, LOOKBACKS, REBALS, WEIGHTINGS, SIZINGS):
        grid.append(dict(hyp="a_LF", signal=sig, L=L, rebal=rb, weighting=wt, sizing=sz,
                         direction="lf", vol_target=None, asset="top20"))
    for sig, L, rb, wt in itertools.product(SIGNALS, LOOKBACKS, REBALS, WEIGHTINGS):
        grid.append(dict(hyp="b_LS", signal=sig, L=L, rebal=rb, weighting=wt, sizing="slot",
                         direction="ls", vol_target=None, asset="top20"))
    for sig, L, rb, d in itertools.product(SIGNALS, LOOKBACKS, REBALS, ("lf", "ls")):
        grid.append(dict(hyp="c_BTC", signal=sig, L=L, rebal=rb, weighting=None, sizing=None,
                         direction=d, vol_target=None, asset="btc"))
    for sig, L, rb, wt in itertools.product(SIGNALS, LOOKBACKS, REBALS, WEIGHTINGS):
        grid.append(dict(hyp="d_VT", signal=sig, L=L, rebal=rb, weighting=wt, sizing="full",
                         direction="lf", vol_target=VOL_TARGET, asset="top20"))
    for sig, L, rb in itertools.product(SIGNALS, LOOKBACKS, REBALS):
        grid.append(dict(hyp="d_VT", signal=sig, L=L, rebal=rb, weighting=None, sizing=None,
                         direction="lf", vol_target=VOL_TARGET, asset="btc"))
    for i, g in enumerate(grid):
        g["id"] = i
        g["name"] = config_name(g)
    return grid


def config_name(g):
    parts = [g["hyp"], g["asset"], g["direction"], f'{g["signal"]}{g["L"]}', g["rebal"]]
    if g["weighting"]:
        parts.append(g["weighting"])
    if g["sizing"]:
        parts.append(g["sizing"])
    if g["vol_target"]:
        parts.append(f'vt{int(100 * g["vol_target"])}')
    return "|".join(parts)


class Builder:
    def __init__(self, panel):
        self.px = panel["PriceUSD"]
        self.rets = self.px.pct_change(fill_method=None)
        self.members = universe(panel, top_n=20, min_history=180, min_volume=1e6)
        self.vol = rolling_vol(self.rets)
        self._sign = {}

    def sign(self, kind, L):
        if (kind, L) not in self._sign:
            self._sign[(kind, L)] = trend_sign(self.px, kind, L)
        return self._sign[(kind, L)]

    def weights(self, g) -> pd.DataFrame:
        s = self.sign(g["signal"], g["L"])
        if g["asset"] == "btc":
            w = btc_weights(s, self.px, g["direction"])
        elif g["direction"] == "ls":
            w = multi_long_short(s, self.members, self.vol, g["weighting"])
        else:
            w = multi_long_flat(s, self.members, self.vol, g["weighting"], g["sizing"])
        if g["vol_target"]:
            w = vol_target(w, self.rets, g["vol_target"])
        return apply_rebal(w, g["rebal"])


# --------------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------------

KEYS = ("sharpe", "cagr_pct", "total_pct", "max_dd_pct", "vol_pct", "exposure", "turnover_per_year")


def slim(m):
    return {k: round(float(m[k]), 4) for k in KEYS}


def yearly(returns: pd.Series, years=TEST_YEARS):
    out = {}
    for y in years:
        r = returns.loc[f"{y}-01-01":f"{y}-12-31"]
        out[str(y)] = round(100 * float((1 + r).prod() - 1), 2) if len(r) else None
    return out


def test_sharpe(res):
    return float(res.metrics(*TEST)["sharpe"])


PLACEBO_WINDOWS = {"test": TEST, "extra_oos_2014_2017": EXTRA_OOS, "2014_to_end": ("2014-01-01", None)}


def placebo(w, px, members, kind, n, seed, cost_bps=10.0, windows=("test",), span_start=None):
    """Sharpe of `n` placebo portfolios in each window.  `span_start` restricts the timing
    placebo's rotation to dates >= span_start (sensitivity variant, see README)."""
    rng = np.random.default_rng(seed)
    px_span = px
    if span_start is not None:
        px_span = px.copy()
        px_span.loc[px.index < span_start] = np.nan
    out = {k: np.empty(n) for k in windows}
    for i in range(n):
        pw = circular_shift(w, px_span, rng) if kind == "timing" else permute_assets(w, members, rng)
        res = backtest(pw, px, cost_bps=cost_bps)
        for k in windows:
            out[k][i] = res.metrics(*PLACEBO_WINDOWS[k])["sharpe"]
    return out


def pct_below(real, draws):
    return float(100 * np.mean(draws < real))


def placebo_summary(real, draws):
    return dict(real_sharpe=round(real, 4), percentile=round(pct_below(real, draws), 2),
                placebo_median=round(float(np.median(draws)), 4),
                placebo_p95=round(float(np.percentile(draws, 95)), 4))


def placebo_block(b, g, w, n, seed, span_sensitivity=False):
    """Mandatory placebos (TEST window).  BTC-only: timing only, also reported on the
    2014-2017 extra OOS and 2014-end windows.  Multi-asset: timing and selection."""
    is_btc = g["asset"] == "btc"
    windows = tuple(PLACEBO_WINDOWS) if is_btc else ("test",)
    jobs = [("timing", "timing", 10.0, None)]
    if not is_btc:
        jobs.append(("selection", "selection", 10.0, None))
        if g["rebal"] == "D":
            # Re-drawing random assets every day makes the placebo pay large turnover costs,
            # which flatters the real strategy; re-check with zero costs on both sides.
            jobs.append(("selection_cost0", "selection", 0.0, None))
        if span_sensitivity:
            # The harness rotates each asset over its whole priced span, so positions from
            # 2013-2017 (when the universe had 1-12 members and weights were large) land in
            # TEST and the placebo's gross can exceed 1.  Variant: rotate within 2018-end only.
            jobs.append(("timing_span2018", "timing", 10.0, "2018-01-01"))
    out = {}
    for j, (label, kind, cost, span) in enumerate(jobs):
        res = backtest(w, b.px, cost_bps=cost)
        draws = placebo(w, b.px, b.members, kind, n, seed + 1000 * j, cost, windows, span)
        real_test = float(res.metrics(*TEST)["sharpe"])
        d = dict(n=n, kind=kind, cost_bps=cost, span_start=span, **placebo_summary(real_test, draws["test"]))
        for k in windows:
            if k != "test":
                d[k] = placebo_summary(float(res.metrics(*PLACEBO_WINDOWS[k])["sharpe"]), draws[k])
        out[label] = d
    return out


def lookahead_audit(panel, grid, cut="2021-06-15", step=7):
    """Rebuild weights from a panel truncated at `cut`; rows <= cut must be identical."""
    full = Builder(panel)
    trunc = Builder({k: v.loc[:cut] for k, v in panel.items()})
    worst = 0.0
    for g in grid[::step]:
        g2 = dict(g, rebal="D")                  # compare the daily targets before thinning
        d = (full.weights(g2).loc[:cut] - trunc.weights(g2)).abs().max().max()
        worst = max(worst, float(d))
    assert worst < 1e-12, f"weights depend on data after {cut}: max diff {worst}"
    return dict(cut=cut, configs_checked=len(grid[::step]), max_abs_diff=worst)


def placebo_min_pct(pb):
    """Strict: the lowest TEST percentile over every placebo variant that was run."""
    return min(v["percentile"] for v in pb.values())


def describe(g):
    sig = (f'past {g["L"]}-day return > 0' if g["signal"] == "ret"
           else f'close above its {g["L"]}-day moving average')
    rb = "weekly (Sunday close)" if g["rebal"] == "W" else "daily"
    if g["asset"] == "btc":
        what = "BTC only, " + ("long when " if g["direction"] == "lf" else "long/short on sign of: ")
        what += sig
        if g["direction"] == "lf":
            what += ", otherwise cash"
    else:
        what = ("Top-20 universe, long assets whose " + sig +
                (", short the rest" if g["direction"] == "ls" else ", others in cash"))
        what += f', {"equal" if g["weighting"] == "ew" else "inverse-60d-vol"} weights'
        if g["direction"] == "lf" and not g["vol_target"]:
            what += (", fixed 1/N slot per member (gross = share in uptrend)" if g["sizing"] == "slot"
                     else ", fully invested across the 'on' assets")
    if g["vol_target"]:
        what += f', gross scaled to {int(100 * g["vol_target"])}% ex-ante vol (60d), max gross 1'
    return f"{what}; rebalanced {rb}."


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main():
    t0 = time.time()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    b = Builder(panel)
    px = b.px

    # Sanity check against the stored benchmark
    bh = backtest(buy_and_hold(px, "btc"), px)
    bh_m = bh.split_metrics()
    assert abs(bh_m["test"]["sharpe"] - 0.479) < 0.01, bh_m["test"]["sharpe"]
    bench = {"btc_bh": {"train": slim(bh_m["train"]), "test": slim(bh_m["test"]),
                        "extra_oos_2014_2017": slim(bh.metrics(*EXTRA_OOS)),
                        "test_years": yearly(bh.returns)}}

    grid = build_grid()
    audit = lookahead_audit(panel, grid)
    print("lookahead audit:", audit, flush=True)
    rows = []
    for g in grid:
        w = b.weights(g)
        res = backtest(w, px)
        sm = res.split_metrics()
        row = dict(g)
        row["train"] = slim(sm["train"])
        row["test"] = slim(sm["test"])
        row["test_years"] = yearly(res.returns)
        if g["asset"] == "btc":
            row["extra_oos_2014_2017"] = slim(res.metrics(*EXTRA_OOS))
        rows.append(row)
    print(f"grid: {len(rows)} configs in {time.time() - t0:.1f}s")

    df = pd.DataFrame([{"id": r["id"], "name": r["name"], "hyp": r["hyp"],
                        "group": f'{r["hyp"]}|{r["asset"]}|{r["direction"]}',
                        "train_sharpe": r["train"]["sharpe"], "test_sharpe": r["test"]["sharpe"],
                        "test_total": r["test"]["total_pct"], "test_dd": r["test"]["max_dd_pct"]}
                       for r in rows])
    df.to_csv(TMP_DIR / "grid.csv", index=False)

    # ---- selection on TRAIN only -----------------------------------------------------------
    best_id = int(df.loc[df["train_sharpe"].idxmax(), "id"])
    best_per_hyp = {h: int(sub.loc[sub["train_sharpe"].idxmax(), "id"]) for h, sub in df.groupby("hyp")}

    def grid_stats(sub):
        return dict(n=int(len(sub)), median_test_sharpe=round(float(sub["test_sharpe"].median()), 4),
                    mean_test_sharpe=round(float(sub["test_sharpe"].mean()), 4),
                    n_beat_btc_test=int((sub["test_sharpe"] > BTC_BENCH_TEST_SHARPE).sum()),
                    median_train_sharpe=round(float(sub["train_sharpe"].median()), 4),
                    spearman_train_test=round(float(sub["train_sharpe"].rank().corr(sub["test_sharpe"].rank())), 4))

    summary = {"all": grid_stats(df)}
    for h, sub in df.groupby("hyp"):
        summary[h] = grid_stats(sub)
    # finer split (descriptive; the hypotheses above are the pre-registered groups)
    summary["by_group"] = {k: grid_stats(sub) for k, sub in df.groupby("group")}
    summary["by_signal"] = {k: grid_stats(df[df["name"].str.contains(f"|{k}", regex=False)])
                            for k in ("ret", "ma")}
    summary["by_rebal"] = {k: grid_stats(df[df["name"].str.contains(f"|{k}|", regex=False)
                                            | df["name"].str.endswith(f"|{k}")])
                           for k in REBALS}
    summary["by_lookback"] = {str(L): grid_stats(df[df["name"].str.contains(rf"\|(?:ret|ma){L}\|")])
                              for L in LOOKBACKS}
    print(json.dumps(summary, indent=1), flush=True)

    # ---- chosen config: full robustness -----------------------------------------------------
    g = grid[best_id]
    w = b.weights(g)
    res = backtest(w, px)
    sm = res.split_metrics()
    chosen = dict(g)
    chosen["description"] = describe(g)
    chosen["train"] = slim(sm["train"])
    chosen["test"] = slim(sm["test"])
    chosen["full"] = slim(sm["full"])
    chosen["test_years"] = yearly(res.returns)
    chosen["test_sharpe_cost30"] = round(test_sharpe(backtest(w, px, cost_bps=30)), 4)
    chosen["test_sharpe_lag0"] = round(test_sharpe(backtest(w, px, lag=0)), 4)
    chosen["test_sharpe_lag2"] = round(test_sharpe(backtest(w, px, lag=2)), 4)
    if g["asset"] == "btc":
        chosen["extra_oos_2014_2017"] = slim(res.metrics(*EXTRA_OOS))
    tp = time.time()
    chosen["placebo"] = placebo_block(b, g, w, N_PLACEBO_CHOSEN, SEED, span_sensitivity=True)
    print(f"placebo for chosen: {time.time() - tp:.1f}s")
    pct = placebo_min_pct(chosen["placebo"])
    yrs = [v for v in chosen["test_years"].values() if v is not None]
    checks = {
        "test_sharpe_gt_btc": chosen["test"]["sharpe"] > BTC_BENCH_TEST_SHARPE,
        "placebo_pct_ge_95": pct >= 95,
        "cost30_test_sharpe_gt_btc": chosen["test_sharpe_cost30"] > BTC_BENCH_TEST_SHARPE,
        "positive_years_ge_3_of_5": sum(v > 0 for v in yrs) >= 3,
        "grid_median_test_sharpe_gt_0": summary["all"]["median_test_sharpe"] > 0,
    }
    chosen["placebo_min_percentile"] = pct
    chosen["robust_checks"] = checks
    chosen["robust_candidate"] = all(checks.values())

    # ---- best of each hypothesis (TRAIN-chosen within it), information only ----------------
    per_hyp = {}
    for h, gid in best_per_hyp.items():
        gh = grid[gid]
        wh = b.weights(gh)
        rh = backtest(wh, px)
        d = dict(id=gid, name=gh["name"], description=describe(gh),
                 train=slim(rh.metrics(*TRAIN)), test=slim(rh.metrics(*TEST)),
                 test_years=yearly(rh.returns),
                 test_sharpe_cost30=round(test_sharpe(backtest(wh, px, cost_bps=30)), 4),
                 test_sharpe_lag0=round(test_sharpe(backtest(wh, px, lag=0)), 4))
        if gh["asset"] == "btc":
            d["extra_oos_2014_2017"] = slim(rh.metrics(*EXTRA_OOS))
        d["placebo"] = (chosen["placebo"] if gid == best_id
                        else placebo_block(b, gh, wh, N_PLACEBO_INFO, SEED + 7 * gid))
        per_hyp[h] = d
        print(h, d["name"], "train", d["train"]["sharpe"], "test", d["test"]["sharpe"],
              {k: v["percentile"] for k, v in d["placebo"].items()})

    # BTC-only configs: is the 2014-2017 extra out-of-sample consistent with TRAIN choice?
    def btc_stats(sel):
        ex = np.array([r["extra_oos_2014_2017"]["sharpe"] for r in sel])
        ex_dd = np.array([r["extra_oos_2014_2017"]["max_dd_pct"] for r in sel])
        te = np.array([r["test"]["sharpe"] for r in sel])
        return dict(n=len(sel), median_extra_oos_sharpe=round(float(np.median(ex)), 4),
                    median_extra_oos_max_dd_pct=round(float(np.median(ex_dd)), 2),
                    n_beat_btc_bh_extra_oos=int((ex > bench["btc_bh"]["extra_oos_2014_2017"]["sharpe"]).sum()),
                    median_test_sharpe=round(float(np.median(te)), 4),
                    n_beat_btc_bh_test=int((te > BTC_BENCH_TEST_SHARPE).sum()))

    btc_rows = [r for r in rows if r["asset"] == "btc"]
    btc_extra = {"all_btc": btc_stats(btc_rows),
                 "btc_long_flat": btc_stats([r for r in btc_rows if r["direction"] == "lf"]),
                 "btc_long_short": btc_stats([r for r in btc_rows if r["direction"] == "ls"])}

    results = dict(
        meta=dict(seed=SEED, lookbacks=LOOKBACKS, signals=SIGNALS, rebals=REBALS, weightings=WEIGHTINGS,
                  sizings=SIZINGS, vol_window=VOL_WIN, vol_target=VOL_TARGET, cost_bps=10, lag=1,
                  short_fee_annual=0.05, train=TRAIN, test=TEST, extra_oos=EXTRA_OOS,
                  universe="top_n=20, min_history=180, min_volume=1e6",
                  n_configs_tested=len(rows), selection="max TRAIN Sharpe over the whole grid",
                  runtime_s=None),
        benchmarks=bench,
        lookahead_audit=audit,
        grid_summary=summary,
        btc_only_extra_oos_summary=btc_extra,
        chosen=chosen,
        best_per_hypothesis=per_hyp,
        grid=rows,
    )
    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(json.dumps({k: chosen[k] for k in ("name", "description", "train", "test", "test_years",
                                             "test_sharpe_cost30", "test_sharpe_lag0", "placebo",
                                             "robust_checks", "robust_candidate")}, indent=1))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
