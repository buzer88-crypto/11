"""Risk management and regime timing (not prediction) on the crypto panel.

Run:  cd /home/user/11 && python3 research/regime_vol/run.py

Hypotheses (all signals use data <= t only; the harness trades row t at close t+lag):
  (a) VT  volatility targeting: scale BTC, or the top-20 equal-weight basket, to a target annual
          vol {20%, 40%, 60%} from trailing realized vol {20, 60 days}; max gross 1.
          Rebalance weekly (W) or daily (D).
  (b) RF  BTC regime filter: hold BTC, the alt basket (top-20 minus BTC) or the top-20 basket
          only while BTC closes above its N-day MA (N in 50/100/200), otherwise cash. W or D.
  (c) VB  volatility-compression breakout: enter when the asset's 20-day vol (measured the day
          before) is in its lowest x% {10, 20, 30} (expanding, or rolling 365-day percentile)
          and the close breaks above the prior N-day high {20, 50, 100}; exit when the close
          falls below the prior N/2-day low.  BTC alone, or each top-20 member in a 1/N slot.
  (d) DD  drawdown stop: hold until the close is X% {10, 20, 30} below the highest close since
          entry; then cash until the close makes a new N-day high {20, 50, 100}.  BTC, the
          top-20 basket (stop on its own index), or each top-20 member in a 1/N slot.

Daily mode ("D") decides every day but only emits a row (= trades) on days the target changes
plus each Sunday (to re-balance drift).  For vol targeting the target changes every day.

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
                                circular_shift, equal_weight, permute_assets, rebalance_on)

OUT_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/claude-0/research-regime_vol")

SEED = 20261014
BTC_BENCH_TEST_SHARPE = 0.48    # BTC buy&hold TEST Sharpe (data/benchmarks.json: 0.479)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
EXTRA_OOS = ("2014-01-01", "2017-12-31")   # never used for selection; meaningful for BTC only
N_PLACEBO_CHOSEN = 500
N_PLACEBO_INFO = 200

# ---- pre-registered grid ---------------------------------------------------------------------
VT_ASSETS, VT_TARGETS, VT_WINDOWS = ("btc", "top20"), (0.20, 0.40, 0.60), (20, 60)
RF_ASSETS, RF_MAS = ("btc", "alts", "top20"), (50, 100, 200)
REBALS = ("W", "D")
VB_ASSETS, VB_PCTS, VB_HIGHS = ("btc", "top20_each"), (0.10, 0.20, 0.30), (20, 50, 100)
VB_PCT_MODES = ("expanding", "rolling365")
VB_VOL_WIN, VB_PCT_MINP = 20, 365
DD_ASSETS, DD_STOPS, DD_HIGHS = ("btc", "top20", "top20_each"), (0.10, 0.20, 0.30), (20, 50, 100)


def build_grid():
    grid = []
    for a, tgt, win, rb in itertools.product(VT_ASSETS, VT_TARGETS, VT_WINDOWS, REBALS):
        grid.append(dict(hyp="a_VT", asset=a, target=tgt, vol_win=win, rebal=rb))
    for a, n, rb in itertools.product(RF_ASSETS, RF_MAS, REBALS):
        grid.append(dict(hyp="b_RF", asset=a, ma=n, rebal=rb))
    for a, x, n, mode in itertools.product(VB_ASSETS, VB_PCTS, VB_HIGHS, VB_PCT_MODES):
        grid.append(dict(hyp="c_VB", asset=a, pct=x, high=n, pct_mode=mode, rebal="D"))
    for a, x, n in itertools.product(DD_ASSETS, DD_STOPS, DD_HIGHS):
        grid.append(dict(hyp="d_DD", asset=a, stop=x, high=n, rebal="D"))
    for i, g in enumerate(grid):
        g["id"] = i
        g["name"] = config_name(g)
    return grid


def config_name(g):
    h = g["hyp"]
    if h == "a_VT":
        p = f'vt{int(100 * g["target"])}|win{g["vol_win"]}'
    elif h == "b_RF":
        p = f'btc>ma{g["ma"]}'
    elif h == "c_VB":
        p = f'p{int(100 * g["pct"])}|hi{g["high"]}|{g["pct_mode"]}'
    else:
        p = f'stop{int(100 * g["stop"])}|hi{g["high"]}'
    return f'{h}|{g["asset"]}|{p}|{g["rebal"]}'


def describe(g):
    what = {"btc": "BTC", "alts": "the equal-weight alt basket (top-20 members except BTC)",
            "top20": "the equal-weight top-20 basket",
            "top20_each": "each top-20 member separately (fixed 1/N slot, cash when off)"}[g["asset"]]
    rb = ("weekly (Sunday close)" if g["rebal"] == "W"
          else "decided daily, traded on change days plus a weekly drift rebalance")
    h = g["hyp"]
    if h == "a_VT":
        s = (f'Hold {what} scaled to {int(100 * g["target"])}% annual vol using trailing '
             f'{g["vol_win"]}-day realized vol, max gross 1')
    elif h == "b_RF":
        s = f'Hold {what} only while BTC closes above its {g["ma"]}-day MA, otherwise cash'
    elif h == "c_VB":
        s = (f'Long {what} when its 20-day vol (previous day) is in the lowest {int(100 * g["pct"])}% '
             f'({g["pct_mode"]} percentile) and the close breaks the prior {g["high"]}-day high; '
             f'exit below the prior {g["high"] // 2}-day low')
    else:
        s = (f'Hold {what}; go to cash after a {int(100 * g["stop"])}% fall from the highest close '
             f'since entry, re-enter on a new {g["high"]}-day closing high')
    return f"{s}; {rb}."


# --------------------------------------------------------------------------------------------
# Signals (every function maps data <= t to row t)
# --------------------------------------------------------------------------------------------

def ann_vol(r, win):
    return r.rolling(win, min_periods=win).std() * np.sqrt(DAYS_PER_YEAR)


def ew(members: pd.DataFrame) -> pd.DataFrame:
    w = members.astype(float)
    return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def breakout_state(P: pd.DataFrame, pct: pd.DataFrame, x: float, n: int) -> np.ndarray:
    """1 while long.  Entry at t: vol percentile at t-1 <= x and close_t > max(close t-n..t-1).
    Exit at t: close_t < min(close t-n/2..t-1)."""
    m = n // 2
    hi = P.shift(1).rolling(n, min_periods=n).max()
    lo = P.shift(1).rolling(m, min_periods=m).min()
    entry = ((pct.shift(1) <= x) & (P > hi)).to_numpy()
    exit_ = (P < lo).to_numpy()
    valid = P.notna().to_numpy()
    st = np.zeros(P.shape[1], bool)
    out = np.zeros(P.shape, bool)
    for t in range(len(P)):
        nxt = np.where(st, ~exit_[t], entry[t])
        st = np.where(valid[t], nxt, st)
        out[t] = st
    return out


def dd_stop_state(P: pd.DataFrame, stop: float, n: int) -> np.ndarray:
    """1 while held.  Starts held at the first price.  Stop at t if close_t <= peak*(1-stop),
    peak = highest close since entry (incl. t).  Re-enter at t if close_t > max(close t-n..t-1)."""
    hi = P.shift(1).rolling(n, min_periods=n).max().to_numpy()
    p = P.to_numpy()
    st = np.ones(P.shape[1], bool)
    peak = np.full(P.shape[1], -np.inf)
    out = np.zeros(P.shape, bool)
    for t in range(len(P)):
        pt = p[t]
        v = ~np.isnan(pt)
        inn = st & v
        peak[inn] = np.maximum(peak[inn], pt[inn])
        stopped = inn & (pt <= peak * (1 - stop))
        st[stopped] = False
        with np.errstate(invalid="ignore"):
            reent = (~st) & v & ~stopped & (pt > hi[t])
        st[reent] = True
        peak[reent] = pt[reent]
        out[t] = st
    return out


def thin(w: pd.DataFrame, rebal: str) -> pd.DataFrame:
    """W: Sunday rows only.  D: rows where the target changed, plus every Sunday."""
    if rebal == "W":
        return rebalance_on(w, "W-SUN")
    change = w.ne(w.shift(1)).any(axis=1).to_numpy().copy()
    change[0] = True
    keep = change | (w.index.dayofweek == 6)
    out = w.copy()
    out.loc[~keep] = np.nan
    return out


class Builder:
    def __init__(self, panel):
        self.px = panel["PriceUSD"]
        self.rets = self.px.pct_change(fill_method=None)
        self.members = universe(panel, top_n=20, min_history=180, min_volume=1e6)
        self.alts = self.members.copy()
        self.alts["btc"] = False
        self._c = {}

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def mem(self, asset):
        return self.alts if asset == "alts" else self.members

    def base(self, asset) -> pd.DataFrame:
        """Fully invested underlying (target weights before the overlay)."""
        def f():
            if asset == "btc":
                b = pd.DataFrame(0.0, index=self.px.index, columns=self.px.columns)
                b["btc"] = self.px["btc"].notna().astype(float)
                return b
            return ew(self.mem(asset))
        return self.cached(("base", asset), f)

    def index_ret(self, asset) -> pd.Series:
        """Daily return of the equal-weight basket held from close t-1 (members at t-1)."""
        def f():
            if asset == "btc":
                return self.rets["btc"]
            prev = self.mem(asset).shift(1, fill_value=False)
            return self.rets.where(prev).mean(axis=1)
        return self.cached(("iret", asset), f)

    def index_level(self, asset) -> pd.Series:
        def f():
            if asset == "btc":
                return self.px["btc"]
            r = self.index_ret(asset)
            lvl = (1 + r.fillna(0.0)).cumprod()
            return lvl.where(r.notna().cummax())
        return self.cached(("ilvl", asset), f)

    def vol_pct(self, mode):
        def f():
            v = ann_vol(self.rets, VB_VOL_WIN)
            if mode == "expanding":
                return v.expanding(min_periods=VB_PCT_MINP).rank(pct=True)
            return v.rolling(365, min_periods=VB_PCT_MINP).rank(pct=True)
        return self.cached(("vpct", mode), f)

    def overlay(self, g):
        """Returns (base, scale) for single-overlay strategies, or (targets, None) for per-asset."""
        h, a = g["hyp"], g["asset"]
        idx = self.px.index
        if h == "a_VT":
            vol = ann_vol(self.index_ret(a), g["vol_win"])
            scale = (g["target"] / vol).clip(upper=1.0).fillna(0.0)
            return self.base(a), scale
        if h == "b_RF":
            btc = self.px["btc"]
            on = (btc > btc.rolling(g["ma"], min_periods=g["ma"]).mean()).astype(float)
            return self.base(a), on
        if h == "c_VB":
            if a == "btc":
                st = breakout_state(self.px[["btc"]], self.vol_pct(g["pct_mode"])[["btc"]],
                                    g["pct"], g["high"])
                return self.base("btc"), pd.Series(st[:, 0].astype(float), idx)
            st = breakout_state(self.px, self.vol_pct(g["pct_mode"]), g["pct"], g["high"])
            slot = ew(self.members)
            return slot * st, None
        if h == "d_DD":
            if a == "top20_each":
                st = dd_stop_state(self.px, g["stop"], g["high"])
                return ew(self.members) * st, None
            lvl = self.index_level(a).to_frame()
            st = dd_stop_state(lvl, g["stop"], g["high"])[:, 0]
            return self.base(a), pd.Series(st.astype(float), idx)
        raise ValueError(h)

    def targets(self, g) -> pd.DataFrame:
        b, s = self.overlay(g)
        return b if s is None else b.mul(s, axis=0)

    def weights(self, g) -> pd.DataFrame:
        return thin(self.targets(g), g["rebal"])

    def underlying(self, asset) -> pd.DataFrame:
        """Buy & hold benchmark for the asset the overlay is applied to."""
        if asset == "btc":
            return buy_and_hold(self.px, "btc")
        return equal_weight(self.alts if asset == "alts" else self.members, "W-SUN")


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


def scale_shift(base, scale, rebal, rng, span_start=None, min_shift=90):
    """Common-timing placebo for overlay strategies: rotate the scalar overlay (on/off or
    vol-target scale) as one block in time, keep the underlying basket at its real dates."""
    s = scale.to_numpy().copy()
    valid = base.abs().sum(axis=1).to_numpy() > 0
    if span_start is not None:
        valid &= base.index >= span_start
    pos = np.flatnonzero(valid)
    a, b = pos[0], pos[-1] + 1
    k = rng.integers(min_shift, (b - a) - min_shift)
    s[a:b] = np.roll(s[a:b], k)
    return thin(base.mul(pd.Series(s, base.index), axis=0), rebal)


def pct_below(real, draws):
    return float(100 * np.mean(np.asarray(draws) < real))


def placebo_block(bld, g, n, seed):
    """Timing placebo (harness circular_shift) always.  Overlay strategies: also the common
    block shift of the overlay over its whole span and over 2018-end.  Per-asset strategies:
    also circular_shift restricted to 2018-end and the selection placebo (permute_assets)."""
    px = bld.px
    w = bld.weights(g)
    base, scale = bld.overlay(g)
    real = test_sharpe(backtest(w, px))
    jobs = [("timing_circular_shift", "harness")]
    if scale is not None:
        jobs += [("timing_common_shift_full", "common_full"), ("timing_common_shift_2018", "common_2018")]
    else:
        # The harness rotates each asset over its whole priced span, so per-asset positions from
        # 2013-2017 (universe of 1-12 coins, so large 1/N slots) land in TEST.  Variant: rotate
        # within 2018-end only.
        jobs += [("timing_circular_shift_2018", "harness_2018"), ("selection_permute_assets", "selection")]
    px_2018 = px.copy()
    px_2018.loc[px.index < "2018-01-01"] = np.nan
    out = {}
    for j, (label, kind) in enumerate(jobs):
        rng = np.random.default_rng(seed + 1000 * j)
        draws, exps = np.empty(n), np.empty(n)
        for i in range(n):
            if kind == "harness":
                pw = circular_shift(w, px, rng)
            elif kind == "harness_2018":
                pw = circular_shift(w, px_2018, rng)
            elif kind == "selection":
                pw = permute_assets(w, bld.members, rng)
            else:
                pw = scale_shift(base, scale, g["rebal"], rng,
                                 span_start="2018-01-01" if kind == "common_2018" else None)
            res = backtest(pw, px)
            m = res.metrics(*TEST)
            draws[i], exps[i] = m["sharpe"], m["exposure"]
        out[label] = dict(n=n, real_test_sharpe=round(real, 4), percentile=round(pct_below(real, draws), 2),
                          placebo_median=round(float(np.median(draws)), 4),
                          placebo_p95=round(float(np.percentile(draws, 95)), 4),
                          placebo_mean_test_exposure=round(float(exps.mean()), 4))
    return out


def posthoc_btc_timing(bld, grid, n, seed):
    """INFORMATION ONLY, chosen after looking at TEST: every BTC-only regime-filter and
    drawdown-stop config, with the harness timing placebo on TEST and on 2014-2017
    (a period never used for anything else)."""
    px = bld.px
    out = {}
    for g in grid:
        if g["asset"] != "btc" or g["hyp"] not in ("b_RF", "d_DD"):
            continue
        w = bld.weights(g)
        res = backtest(w, px)
        tr, te, ex = res.metrics(*TRAIN), res.metrics(*TEST), res.metrics(*EXTRA_OOS)
        rng = np.random.default_rng(seed + g["id"])
        d_te, d_ex = np.empty(n), np.empty(n)
        for i in range(n):
            pr = backtest(circular_shift(w, px, rng), px)
            d_te[i] = pr.metrics(*TEST)["sharpe"]
            d_ex[i] = pr.metrics(*EXTRA_OOS)["sharpe"]
        out[g["name"]] = dict(
            train_sharpe=round(tr["sharpe"], 4), test_sharpe=round(te["sharpe"], 4),
            test_max_dd_pct=round(te["max_dd_pct"], 2), test_exposure=round(te["exposure"], 3),
            test_placebo_pct=round(pct_below(te["sharpe"], d_te), 2),
            extra_oos_sharpe=round(ex["sharpe"], 4), extra_oos_max_dd_pct=round(ex["max_dd_pct"], 2),
            extra_oos_placebo_pct=round(pct_below(ex["sharpe"], d_ex), 2), n=n)
        print("posthoc", g["name"], out[g["name"]], flush=True)
    return out


def lookahead_audit(panel, grid, cut="2021-06-15"):
    """Rebuild every config's daily targets from a panel truncated at `cut`; rows <= cut must match."""
    full = Builder(panel)
    trunc = Builder({k: v.loc[:cut] for k, v in panel.items()})
    worst = 0.0
    for g in grid:
        d = (full.targets(g).loc[:cut] - trunc.targets(g)).abs().max().max()
        worst = max(worst, float(d))
    assert worst < 1e-12, f"weights depend on data after {cut}: max diff {worst}"
    return dict(cut=cut, configs_checked=len(grid), max_abs_diff=worst)


def grid_stats(sub):
    return dict(
        n=int(len(sub)),
        median_test_sharpe=round(float(sub["test_sharpe"].median()), 4),
        n_beat_btc_test=int((sub["test_sharpe"] > BTC_BENCH_TEST_SHARPE).sum()),
        n_test_sharpe_gt_0=int((sub["test_sharpe"] > 0).sum()),
        median_train_sharpe=round(float(sub["train_sharpe"].median()), 4),
        spearman_train_test=(round(float(sub["train_sharpe"].rank().corr(sub["test_sharpe"].rank())), 4)
                             if len(sub) > 2 else None),
        median_test_max_dd=round(float(sub["test_dd"].median()), 2),
        # key question: better than the same asset bought and held?
        n_sharpe_gt_underlying=int((sub["test_sharpe"] > sub["und_test_sharpe"]).sum()),
        median_sharpe_minus_underlying=round(float((sub["test_sharpe"] - sub["und_test_sharpe"]).median()), 4),
        n_dd_better_than_underlying=int((sub["test_dd"] > sub["und_test_dd"]).sum()),
        n_dd_better_than_static_same_exposure=int((sub["test_dd"] > sub["static_test_dd"]).sum()),
        median_sharpe_minus_static=round(float((sub["test_sharpe"] - sub["static_test_sharpe"]).median()), 4),
        median_test_exposure=round(float(sub["test_exposure"].median()), 3),
    )


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def main():
    t0 = time.time()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    bld = Builder(panel)
    px = bld.px

    # ---- benchmarks ------------------------------------------------------------------------
    bench = {}
    und_cache = {}
    for a in ("btc", "alts", "top20"):
        r = backtest(bld.underlying(a), px)
        und_cache[a] = r
        bench[f"{a}_buy_and_hold"] = dict(train=slim(r.metrics(*TRAIN)), test=slim(r.metrics(*TEST)),
                                          extra_oos_2014_2017=slim(r.metrics(*EXTRA_OOS)),
                                          test_years=yearly(r.returns))
    und_cache["top20_each"] = und_cache["top20"]
    assert abs(bench["btc_buy_and_hold"]["test"]["sharpe"] - 0.479) < 0.01

    grid = build_grid()
    audit = lookahead_audit(panel, grid)
    print("lookahead audit:", audit, flush=True)

    rows = []
    for g in grid:
        w = bld.weights(g)
        res = backtest(w, px)
        sm = res.split_metrics()
        und = und_cache[g["asset"]].metrics(*TEST)
        # Static de-risking benchmark: the underlying held at a constant fraction equal to the
        # strategy's average TEST gross exposure (same average risk budget, no timing).
        k = float(sm["test"]["exposure"])
        static = backtest(bld.underlying(g["asset"]) * k, px).metrics(*TEST)
        row = dict(g)
        row["train"] = slim(sm["train"])
        row["test"] = slim(sm["test"])
        row["test_years"] = yearly(res.returns)
        row["extra_oos_2014_2017"] = slim(res.metrics(*EXTRA_OOS))
        row["underlying_test"] = dict(sharpe=round(und["sharpe"], 4), max_dd_pct=round(und["max_dd_pct"], 2))
        row["static_same_exposure_test"] = dict(k=round(k, 4), sharpe=round(static["sharpe"], 4),
                                                max_dd_pct=round(static["max_dd_pct"], 2),
                                                total_pct=round(static["total_pct"], 2))
        rows.append(row)
    print(f"grid: {len(rows)} configs in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame([{
        "id": r["id"], "name": r["name"], "hyp": r["hyp"], "asset": r["asset"], "rebal": r["rebal"],
        "group": f'{r["hyp"]}|{r["asset"]}',
        "train_sharpe": r["train"]["sharpe"], "train_dd": r["train"]["max_dd_pct"],
        "test_sharpe": r["test"]["sharpe"], "test_total": r["test"]["total_pct"],
        "test_cagr": r["test"]["cagr_pct"], "test_dd": r["test"]["max_dd_pct"],
        "test_exposure": r["test"]["exposure"], "test_turnover": r["test"]["turnover_per_year"],
        "und_test_sharpe": r["underlying_test"]["sharpe"], "und_test_dd": r["underlying_test"]["max_dd_pct"],
        "static_test_sharpe": r["static_same_exposure_test"]["sharpe"],
        "static_test_dd": r["static_same_exposure_test"]["max_dd_pct"],
        "extra_sharpe": r["extra_oos_2014_2017"]["sharpe"], "extra_dd": r["extra_oos_2014_2017"]["max_dd_pct"],
        **{f"y{y}": r["test_years"][str(y)] for y in TEST_YEARS},
    } for r in rows])
    df.to_csv(OUT_DIR / "grid.csv", index=False)

    summary = {"all": grid_stats(df)}
    summary["by_hypothesis"] = {h: grid_stats(sub) for h, sub in df.groupby("hyp")}
    summary["by_group"] = {k: grid_stats(sub) for k, sub in df.groupby("group")}
    summary["by_group_rebal"] = {f"{k[0]}|{k[1]}": grid_stats(sub)
                                 for k, sub in df.groupby(["group", "rebal"])}
    summary["by_hyp_param"] = {}
    for h, cols in (("a_VT", ("target", "vol_win")), ("b_RF", ("ma",)), ("c_VB", ("pct", "high", "pct_mode")),
                    ("d_DD", ("stop", "high"))):
        gdf = pd.DataFrame([r for r in grid if r["hyp"] == h]).set_index("id")
        for c in cols:
            for v, ids in gdf.groupby(c).groups.items():
                summary["by_hyp_param"][f"{h}|{c}={v}"] = grid_stats(df.set_index("id").loc[list(ids)])
    print(json.dumps(summary["all"], indent=1), flush=True)
    for k, v in summary["by_group"].items():
        print(f'{k:22s} n={v["n"]:3d} medTEST={v["median_test_sharpe"]:6.3f} beat={v["n_beat_btc_test"]:2d} '
              f'>und={v["n_sharpe_gt_underlying"]:2d} ddBetterUnd={v["n_dd_better_than_underlying"]:2d} '
              f'ddBetterStatic={v["n_dd_better_than_static_same_exposure"]:2d} medDD={v["median_test_max_dd"]}',
              flush=True)

    # ---- selection on TRAIN only -----------------------------------------------------------
    best_id = int(df.loc[df["train_sharpe"].idxmax(), "id"])
    best_per_hyp = {h: int(sub.loc[sub["train_sharpe"].idxmax(), "id"]) for h, sub in df.groupby("hyp")}
    best_per_group = {k: int(sub.loc[sub["train_sharpe"].idxmax(), "id"]) for k, sub in df.groupby("group")}

    def detail(gid, n_placebo, seed):
        g = grid[gid]
        w = bld.weights(g)
        res = backtest(w, px)
        sm = res.split_metrics()
        d = dict(rows[gid])
        d["description"] = describe(g)
        d["full"] = slim(sm["full"])
        d["test_sharpe_cost30"] = round(test_sharpe(backtest(w, px, cost_bps=30)), 4)
        d["test_sharpe_lag0"] = round(test_sharpe(backtest(w, px, lag=0)), 4)
        d["test_sharpe_lag2"] = round(test_sharpe(backtest(w, px, lag=2)), 4)
        if n_placebo:
            d["placebo"] = placebo_block(bld, g, n_placebo, seed)
        return d

    tp = time.time()
    chosen = detail(best_id, N_PLACEBO_CHOSEN, SEED)
    print(f"placebo for chosen: {time.time() - tp:.1f}s", flush=True)
    pb = chosen["placebo"]
    pct_primary = pb["timing_circular_shift"]["percentile"]
    pct_min = min(v["percentile"] for v in pb.values())
    yrs = [v for v in chosen["test_years"].values() if v is not None]
    checks = {
        "test_sharpe_gt_btc": chosen["test"]["sharpe"] > BTC_BENCH_TEST_SHARPE,
        "placebo_pct_ge_95_all_variants": pct_min >= 95,
        "cost30_test_sharpe_gt_btc": chosen["test_sharpe_cost30"] > BTC_BENCH_TEST_SHARPE,
        "positive_years_ge_3_of_5": sum(v > 0 for v in yrs) >= 3,
        "grid_median_test_sharpe_gt_0": summary["all"]["median_test_sharpe"] > 0,
    }
    chosen["placebo_primary_percentile"] = pct_primary
    chosen["placebo_min_percentile"] = pct_min
    chosen["robust_checks"] = checks
    chosen["robust_candidate"] = bool(all(checks.values()))

    per_hyp = {}
    for h, gid in best_per_hyp.items():
        per_hyp[h] = chosen if gid == best_id else detail(gid, N_PLACEBO_INFO, SEED + 7 * gid)
        d = per_hyp[h]
        print(h, d["name"], "train", d["train"]["sharpe"], "test", d["test"]["sharpe"], "dd", d["test"]["max_dd_pct"],
              {k: v["percentile"] for k, v in d["placebo"].items()}, flush=True)
    per_group = {}
    for k, gid in best_per_group.items():
        d = detail(gid, 0, 0)
        per_group[k] = {kk: d[kk] for kk in ("id", "name", "description", "train", "test", "test_years",
                                              "test_sharpe_cost30", "test_sharpe_lag0", "underlying_test",
                                              "static_same_exposure_test", "extra_oos_2014_2017")}

    tp = time.time()
    posthoc = posthoc_btc_timing(bld, grid, N_PLACEBO_INFO, SEED + 99991)
    print(f"post-hoc BTC placebos: {time.time() - tp:.1f}s", flush=True)

    results = dict(
        meta=dict(seed=SEED, cost_bps=10, lag=1, short_fee_annual=0.05, train=TRAIN, test=TEST,
                  extra_oos=EXTRA_OOS, universe="top_n=20, min_history=180, min_volume=1e6",
                  grid=dict(vt=dict(assets=VT_ASSETS, targets=VT_TARGETS, vol_windows=VT_WINDOWS, rebals=REBALS),
                            rf=dict(assets=RF_ASSETS, mas=RF_MAS, rebals=REBALS),
                            vb=dict(assets=VB_ASSETS, pcts=VB_PCTS, highs=VB_HIGHS, pct_modes=VB_PCT_MODES,
                                    vol_win=VB_VOL_WIN, pct_min_periods=VB_PCT_MINP, exit="prior high/2-day low"),
                            dd=dict(assets=DD_ASSETS, stops=DD_STOPS, highs=DD_HIGHS)),
                  n_configs_tested=len(rows), selection="max TRAIN Sharpe over the whole grid",
                  n_placebo_chosen=N_PLACEBO_CHOSEN, n_placebo_per_hypothesis=N_PLACEBO_INFO,
                  runtime_s=None),
        benchmarks=bench,
        lookahead_audit=audit,
        grid_summary=summary,
        chosen=chosen,
        best_per_hypothesis=per_hyp,
        best_per_group_no_placebo=per_group,
        post_hoc_btc_timing_information_only=posthoc,
        grid=rows,
    )
    results["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print(json.dumps({k: chosen[k] for k in ("name", "description", "train", "test", "test_years",
                                             "test_sharpe_cost30", "test_sharpe_lag0", "test_sharpe_lag2",
                                             "underlying_test", "static_same_exposure_test",
                                             "extra_oos_2014_2017", "placebo", "robust_checks",
                                             "robust_candidate")}, indent=1))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
