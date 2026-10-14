"""On-chain valuation and activity signals on the crypto panel.

Run:  cd /home/user/11 && python3 research/onchain/run.py

Hypotheses (every signal uses data <= t only; the harness trades row t at close t+lag):
  (a) MVRV   time-series MVRV timing for BTC, ETH and a 50/50 BTC+ETH book.
             Three zones: exposure 1 when cheap, `mid` in between, 0 when expensive.
             "level": cheap = MVRV < 1, expensive = MVRV >= hi (hi in 2.5 / 3.0 / 3.5).
             "pct":   cheap = own percentile < 0.2, expensive = own percentile > p_hi, the percentile
                      being expanding (all history up to t) or rolling 4 years.
             "inv":   exposure = 1 - own percentile (continuous).
  (b) XSMV   cross-sectional MVRV on the point-in-time top-20 universe, rebalanced weekly or
             monthly: long the k "cheapest" coins (long-only) or long k cheapest / short k dearest
             (long-short, 0.5 gross each side). Cheapness = raw MVRV, expanding own percentile,
             rolling 365-day own percentile, or 365-day z-score of log MVRV.
  (c) ACT    network-activity momentum on the top-20 universe, weekly: score = 30/90-day log growth
             of 7-day-mean AdrActCnt or TxCnt, minus the same-horizon log price growth ("rel") or
             activity growth alone ("act"); long the k highest (long-only or long-short).
  (d) FLOW   BTC/ETH exchange net inflow (FlowInExUSD - FlowOutExUSD) summed over 7/30 days as a
             bearish signal: flat when the net inflow is "high", otherwise long.
             "sign": net inflow > 0.  "z0"/"z1": 365-day rolling z-score of the normalised net flow
             above 0 / 1.  "pct80": expanding percentile of the normalised net flow above 0.8.
             Normalisation: by market cap ("mcap") or by gross exchange flow ("gross").

The whole grid is fixed below and run in full. The chosen config is the best TRAIN Sharpe
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
from pinetest.portfolio import (TEST, TRAIN, backtest, buy_and_hold, circular_shift,
                                permute_assets, rebalance_on)

OUT_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/claude-0/research-onchain")

SEED = 20261014
BTC_BENCH_TEST_SHARPE = 0.48     # BTC buy&hold TEST Sharpe (data/benchmarks.json: 0.479)
N_PLACEBO_CHOSEN = 500
N_PLACEBO_INFO = 200             # best config of each hypothesis (information only)
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
FFILL_LIMIT = 3                  # carry a metric over short gaps (past values only)

# --- grid (fixed up front) -------------------------------------------------------------------
A_ASSETS = ("btc", "eth", "btc+eth")
A_REBALS = ("W", "D")
A_RULES = (
    [("level", dict(hi=hi, lo=1.0, mid=mid)) for hi in (2.5, 3.0, 3.5) for mid in (0.5, 1.0)]
    + [("pct", dict(win=win, p_hi=p_hi, p_lo=0.2, mid=mid))
       for win in ("exp", "4y") for p_hi in (0.8, 0.9, 0.95) for mid in (0.5, 1.0)]
    + [("inv", dict(win=win)) for win in ("exp", "4y")]
)
B_SCORES = ("raw", "exp_pct", "roll_pct", "z365")
B_KS = (3, 5, 10)
B_DIRS = ("long", "ls")
B_REBALS = ("W", "M")
C_METRICS = ("AdrActCnt", "TxCnt")
C_LOOKBACKS = (30, 90)
C_SCORES = ("rel", "act")
C_KS = (3, 5, 10)
C_DIRS = ("long", "ls")
D_ASSETS = ("btc", "eth", "btc+eth")
D_WINDOWS = (7, 30)
D_RULES = [("sign", None)] + [(r, n) for r in ("z0", "z1", "pct80") for n in ("mcap", "gross")]
D_REBALS = ("D", "W")
FREQ = {"W": "W-SUN", "M": "M"}


# --------------------------------------------------------------------------------------------
# Helpers (every function maps data <= t to row t)
# --------------------------------------------------------------------------------------------

def own_percentile(x: pd.DataFrame | pd.Series, win, min_periods: int):
    """Percentile of today's value within the asset's own history up to and including t."""
    roll = x.expanding(min_periods=min_periods) if win == "exp" else \
        x.rolling(win, min_periods=min_periods)
    return roll.rank(pct=True)


def assets_of(spec: str) -> list[str]:
    return spec.split("+")


def timing_book(px: pd.DataFrame, exposure: dict[str, pd.Series], rebal: str) -> pd.DataFrame:
    """Each asset gets an equal slot; the slot is invested `exposure` (0..1), the rest is cash."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for a, e in exposure.items():
        w[a] = e.reindex(px.index).astype(float) / len(exposure)
    return w if rebal == "D" else rebalance_on(w, FREQ[rebal])


def xs_book(score: pd.DataFrame, eligible: pd.DataFrame, k: int, direction: str,
            rebal: str) -> pd.DataFrame:
    """Long the k highest `score` (equal weight); "ls" also shorts the k lowest (0.5 per side)."""
    s = score.where(eligible)
    n = s.notna().sum(axis=1).to_numpy()
    hi = s.rank(axis=1, ascending=False, method="first").to_numpy()
    lo = s.rank(axis=1, ascending=True, method="first").to_numpy()
    if direction == "long":
        kk = np.minimum(k, n).astype(float)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(hi <= kk, 1.0 / kk, 0.0)
    else:
        kk = np.minimum(k, n // 2).astype(float)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(hi <= kk, 0.5 / kk, 0.0) - np.where(lo <= kk, 0.5 / kk, 0.0)
    w = pd.DataFrame(np.nan_to_num(w), index=score.index, columns=score.columns)
    return rebalance_on(w, FREQ[rebal])


# --------------------------------------------------------------------------------------------
# Strategies: panel-derived inputs -> weights
# --------------------------------------------------------------------------------------------

class Data:
    def __init__(self, panel):
        self.px = panel["PriceUSD"]
        idx, cols = self.px.index, self.px.columns
        get = lambda f: panel[f].reindex(index=idx, columns=cols).ffill(limit=FFILL_LIMIT)  # noqa
        self.mvrv = get("CapMVRVCur")
        self.mvrv = self.mvrv.where(self.mvrv > 0)
        self.act = {m: get(m).where(get(m) > 0) for m in C_METRICS}
        self.fin, self.fout = get("FlowInExUSD"), get("FlowOutExUSD")
        self.mcap = get("CapMrktCurUSD")
        self.members = universe(panel, top_n=20, min_history=180, min_volume=1e6)
        self._cache = {}

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


def a_mvrv_timing(d: Data, assets: str, rule: str, p: dict, rebal: str) -> pd.DataFrame:
    expo = {}
    for a in assets_of(assets):
        x = d.mvrv[a]
        e = pd.Series(np.nan, index=x.index)
        if rule == "level":
            e[:] = p["mid"]
            e[x < p["lo"]] = 1.0
            e[x >= p["hi"]] = 0.0
            e[x.isna()] = p["mid"]
        else:
            win = "exp" if p["win"] == "exp" else 4 * 365
            pct = d.cached(("pct", a, p["win"]), lambda: own_percentile(x, win, 365))
            if rule == "pct":
                e[:] = p["mid"]
                e[pct < p["p_lo"]] = 1.0
                e[pct > p["p_hi"]] = 0.0
            else:   # inv
                e = (1.0 - pct).fillna(0.5)
        expo[a] = e
    return timing_book(d.px, expo, rebal)


def b_score(d: Data, score: str) -> pd.DataFrame:
    """Higher = cheaper (more attractive to own)."""
    def build():
        m = d.mvrv
        if score == "raw":
            return -m
        if score == "exp_pct":
            return -own_percentile(m, "exp", 180)
        if score == "roll_pct":
            return -own_percentile(m, 365, 180)
        if score == "z365":
            lm = np.log(m)
            mu, sd = lm.rolling(365, min_periods=180).mean(), lm.rolling(365, min_periods=180).std()
            return -((lm - mu) / sd.where(sd > 0))
        raise ValueError(score)
    return d.cached(("b", score), build)


def b_xs_mvrv(d: Data, score: str, k: int, direction: str, rebal: str):
    s = b_score(d, score)
    eligible = d.members & s.notna()
    return xs_book(s, eligible, k, direction, rebal), eligible


def c_score(d: Data, metric: str, lookback: int, score: str) -> pd.DataFrame:
    def build():
        a = d.act[metric].rolling(7, min_periods=5).mean()
        g_act = np.log(a / a.shift(lookback))
        if score == "act":
            return g_act
        g_px = np.log(d.px / d.px.shift(lookback))
        return g_act - g_px
    return d.cached(("c", metric, lookback, score), build)


def c_activity(d: Data, metric: str, lookback: int, score: str, k: int, direction: str):
    s = c_score(d, metric, lookback, score)
    s = s.replace([np.inf, -np.inf], np.nan)
    eligible = d.members & s.notna()
    return xs_book(s, eligible, k, direction, "W"), eligible


def d_flow_signal(d: Data, a: str, window: int, rule: str, norm) -> pd.Series:
    """True = bearish (net inflow to exchanges is high)."""
    net = (d.fin[a] - d.fout[a]).rolling(window, min_periods=window).sum()
    if rule == "sign":
        return net > 0
    if norm == "mcap":
        x = net / d.mcap[a]
    else:
        gross = (d.fin[a] + d.fout[a]).rolling(window, min_periods=window).sum()
        x = net / gross.where(gross > 0)
    if rule in ("z0", "z1"):
        mu, sd = x.rolling(365, min_periods=180).mean(), x.rolling(365, min_periods=180).std()
        z = (x - mu) / sd.where(sd > 0)
        return z > (0.0 if rule == "z0" else 1.0)
    if rule == "pct80":
        return own_percentile(x, "exp", 365) > 0.8
    raise ValueError(rule)


def d_flows(d: Data, assets: str, window: int, rule: str, norm, rebal: str) -> pd.DataFrame:
    expo = {a: (~d_flow_signal(d, a, window, rule, norm)).astype(float) for a in assets_of(assets)}
    return timing_book(d.px, expo, rebal)


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------

def grid():
    """Yield (hypothesis, params, builder) for the whole fixed grid."""
    for assets, (rule, p), rebal in itertools.product(A_ASSETS, A_RULES, A_REBALS):
        params = dict(assets=assets, rule=rule, rebal=rebal, **p)
        yield "a_mvrv_timing", params, (lambda d, a=assets, r=rule, p=p, rb=rebal:
                                        (a_mvrv_timing(d, a, r, p, rb), None))
    for score, k, direction, rebal in itertools.product(B_SCORES, B_KS, B_DIRS, B_REBALS):
        params = dict(score=score, k=k, direction=direction, rebal=rebal)
        yield "b_xs_mvrv", params, (lambda d, s=score, k=k, dr=direction, rb=rebal:
                                    b_xs_mvrv(d, s, k, dr, rb))
    for metric, lb, score, k, direction in itertools.product(C_METRICS, C_LOOKBACKS, C_SCORES,
                                                             C_KS, C_DIRS):
        params = dict(metric=metric, lookback=lb, score=score, k=k, direction=direction,
                      rebal="W")
        yield "c_activity", params, (lambda d, m=metric, lb=lb, s=score, k=k, dr=direction:
                                     c_activity(d, m, lb, s, k, dr))
    for assets, window, (rule, norm), rebal in itertools.product(D_ASSETS, D_WINDOWS, D_RULES,
                                                                 D_REBALS):
        params = dict(assets=assets, window=window, rule=rule, norm=norm, rebal=rebal)
        yield "d_flows", params, (lambda d, a=assets, w=window, r=rule, n=norm, rb=rebal:
                                  (d_flows(d, a, w, r, n, rb), None))


PLACEBO_KIND = {"a_mvrv_timing": "timing", "d_flows": "timing",
                "b_xs_mvrv": "selection", "c_activity": "selection"}


def describe(hyp: str, p: dict) -> str:
    if hyp == "a_mvrv_timing":
        if p["rule"] == "level":
            z = f"exposure 1 if MVRV<{p['lo']}, 0 if MVRV>={p['hi']}, else {p['mid']}"
        elif p["rule"] == "pct":
            z = (f"exposure 1 if own {p['win']} MVRV percentile<{p['p_lo']}, 0 if >{p['p_hi']}, "
                 f"else {p['mid']}")
        else:
            z = f"exposure = 1 - own {p['win']} MVRV percentile"
        return f"MVRV timing on {p['assets']}: {z}; rebalance {p['rebal']}"
    if hyp == "b_xs_mvrv":
        side = "long-only" if p["direction"] == "long" else "long cheapest / short dearest"
        return (f"Cross-sectional MVRV, top-20 universe, score={p['score']}, k={p['k']}, {side}, "
                f"rebalance {p['rebal']}")
    if hyp == "c_activity":
        side = "long-only" if p["direction"] == "long" else "long top / short bottom"
        return (f"Activity momentum, {p['metric']} {p['lookback']}d growth "
                f"{'minus price growth' if p['score'] == 'rel' else 'alone'}, top-20 universe, "
                f"k={p['k']}, {side}, weekly")
    return (f"Exchange net-inflow filter on {p['assets']}: flat when {p['window']}d net inflow "
            f"rule={p['rule']}{'' if p['norm'] is None else ' norm=' + p['norm']} is bearish, "
            f"else long; rebalance {p['rebal']}")


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

def yearly(res, years=TEST_YEARS) -> dict:
    r = res.returns
    return {str(y): float(100 * ((1 + r.loc[str(y)]).prod() - 1)) for y in years
            if len(r.loc[str(y)])}


def test_only_shift(weights: pd.DataFrame, prices: pd.DataFrame, rng, min_shift=90):
    """Supplementary timing placebo: rotate each asset's position series only inside TEST, so the
    TEST exposure profile is exactly the real one (circular_shift draws positions from the whole
    history, which also changes the average TEST exposure)."""
    rows = weights.notna().any(axis=1)
    held = weights.ffill().fillna(0.0)
    out = held.copy()
    t0 = int(np.searchsorted(weights.index, pd.Timestamp(TEST[0])))
    for j, col in enumerate(weights.columns):
        valid = np.flatnonzero(prices[col].reindex(weights.index).notna().to_numpy())
        valid = valid[valid >= t0]
        if len(valid) <= 2 * min_shift:
            continue
        a, b = valid[0], valid[-1] + 1
        span = held[col].to_numpy()[a:b]
        out.iloc[a:b, j] = np.roll(span, rng.integers(min_shift, len(span) - min_shift))
    out.loc[~rows.to_numpy()] = np.nan
    return out


def placebo(weights, eligible, kind, d: Data, n, seed):
    """Return {placebo_name: array of (train_sharpe, test_sharpe)}."""
    out = {}
    kinds = ["circular_shift", "test_only_shift"] if kind == "timing" else ["permute_assets"]
    for i, name in enumerate(kinds):
        rng = np.random.default_rng(seed + i)
        vals = []
        for _ in range(n):
            if name == "circular_shift":
                w = circular_shift(weights, d.px, rng)
            elif name == "test_only_shift":
                w = test_only_shift(weights, d.px, rng)
            else:
                w = permute_assets(weights, eligible, rng)
            m = backtest(w, d.px).split_metrics()
            vals.append((m["train"]["sharpe"], m["test"]["sharpe"]))
        out[name] = np.array(vals)
    return out


def placebo_summary(real_test, real_train, arr, test_only=False):
    tr, te = arr[:, 0], arr[:, 1]
    return {
        "n": int(len(arr)),
        "test_percentile": float(100 * np.mean(te < real_test)),
        "test_share_ties": float(100 * np.mean(te == real_test)),
        # test_only_shift leaves TRAIN untouched, so a TRAIN percentile is meaningless there
        "train_percentile": None if test_only else float(100 * np.mean(tr < real_train)),
        "placebo_test_sharpe_median": float(np.median(te)),
        "placebo_test_sharpe_p05_p95": [float(np.quantile(te, .05)), float(np.quantile(te, .95))],
    }


def underlying_bh(d: Data, assets: str) -> dict:
    """Buy & hold of the same asset(s), equal slots, weekly rebalance (for timing configs)."""
    w = pd.DataFrame(0.0, index=d.px.index, columns=d.px.columns)
    for a in assets_of(assets):
        w[a] = 1.0 / len(assets_of(assets))
    m = backtest(rebalance_on(w, "W-SUN"), d.px).split_metrics()
    return {"train_sharpe": m["train"]["sharpe"], "test_sharpe": m["test"]["sharpe"],
            "test_total_pct": m["test"]["total_pct"]}


def deep_dive(d: Data, row, weights, eligible, n_placebo, seed):
    hyp, p = row["hypothesis"], row["params"]
    res = backtest(weights, d.px)
    m = res.split_metrics()
    out = {
        "id": row["id"], "hypothesis": hyp, "params": p, "description": describe(hyp, p),
        "train": m["train"], "test": m["test"], "full": m["full"],
        "test_yearly_pct": yearly(res),
        "train_yearly_pct": yearly(res, (2018, 2019, 2020, 2021)),
        "test_sharpe_cost30": backtest(weights, d.px, cost_bps=30).metrics(*TEST)["sharpe"],
        "test_sharpe_lag0": backtest(weights, d.px, lag=0).metrics(*TEST)["sharpe"],
        "test_sharpe_lag2": backtest(weights, d.px, lag=2).metrics(*TEST)["sharpe"],
        "placebo_kind": PLACEBO_KIND[hyp],
    }
    if PLACEBO_KIND[hyp] == "timing":
        out["same_assets_buy_hold"] = underlying_bh(d, p["assets"])
    pl = placebo(weights, eligible, PLACEBO_KIND[hyp], d, n_placebo, seed)
    out["placebo"] = {k: placebo_summary(m["test"]["sharpe"], m["train"]["sharpe"], v,
                                         test_only=(k == "test_only_shift"))
                      for k, v in pl.items()}
    ge = res.gross_exposure.loc[TRAIN[0]:]
    out["gross_exposure_by_year"] = {str(y): float(v) for y, v in ge.groupby(ge.index.year).mean().items()}
    return out


def coverage(panel, d: Data) -> dict:
    cov = {}
    m = d.members
    for f in ("CapMVRVCur", "AdrActCnt", "TxCnt", "FlowInExUSD", "FlowOutExUSD"):
        df = panel[f]
        have = sorted(c for c in df.columns if df[c].notna().any())
        x = df.reindex(index=d.px.index, columns=d.px.columns).notna()
        share = ((x & m).sum(axis=1) / m.sum(axis=1).replace(0, np.nan))
        cov[f] = {
            "n_assets_with_data": len(have),
            "assets": have if len(have) <= 5 else None,
            "share_of_top20_members_covered_train": float(share.loc[TRAIN[0]:TRAIN[1]].mean()),
            "share_of_top20_members_covered_test": float(share.loc[TEST[0]:].mean()),
        }
        if len(have) > 5:
            cov[f]["top20_members_missing_in_test"] = sorted(
                c for c in m.columns if (m[c] & ~x[c]).loc[TEST[0]:].sum() > 30)
    # Priced coins that universe() can never select in TEST because market cap or volume is missing
    px_t = d.px.loc[TEST[0]:]
    cap_t = panel["CapMrktCurUSD"].reindex(index=d.px.index, columns=d.px.columns).loc[TEST[0]:]
    vol_t = panel["volume_reported_spot_usd_1d"].reindex(
        index=d.px.index, columns=d.px.columns).loc[TEST[0]:]
    priced = px_t.notna().sum()
    cov["universe_gaps_test"] = {
        c: {"priced_days": int(priced[c]), "mcap_days": int(cap_t[c].notna().sum()),
            "volume_days": int(vol_t[c].notna().sum()),
            "days_in_top20": int(m[c].loc[TEST[0]:].sum())}
        for c in d.px.columns
        if priced[c] > 365 and (cap_t[c].notna().sum() < 0.5 * priced[c]
                                or vol_t[c].notna().sum() < 0.5 * priced[c])}
    return cov


def main():
    t0 = time.time()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    d = Data(panel)

    bench = backtest(buy_and_hold(d.px, "btc"), d.px).split_metrics()
    print(f"BTC buy&hold TRAIN Sharpe {bench['train']['sharpe']:.3f}  "
          f"TEST Sharpe {bench['test']['sharpe']:.3f}")

    rows, store = [], {}
    for i, (hyp, params, build) in enumerate(grid()):
        w, eligible = build(d)
        res = backtest(w, d.px)
        m = res.split_metrics()
        rid = f"{hyp}#{i:03d}"
        rows.append({"id": rid, "hypothesis": hyp, "params": params,
                     "train": m["train"], "test": m["test"], "full": m["full"],
                     "test_yearly_pct": yearly(res)})
        store[rid] = build
    n_configs = len(rows)
    print(f"{n_configs} configs in {time.time() - t0:.0f}s")

    flat = pd.DataFrame([{
        "id": r["id"], "hypothesis": r["hypothesis"],
        "params": json.dumps(r["params"]),
        **{f"train_{k}": r["train"][k] for k in ("sharpe", "cagr_pct", "total_pct", "max_dd_pct",
                                                 "exposure", "turnover_per_year")},
        **{f"test_{k}": r["test"][k] for k in ("sharpe", "cagr_pct", "total_pct", "max_dd_pct",
                                               "exposure", "turnover_per_year")},
        **{f"test_{y}_pct": r["test_yearly_pct"].get(y) for y in map(str, TEST_YEARS)},
    } for r in rows])
    flat.to_csv(OUT_DIR / "grid.csv", index=False)

    def grid_stats(df):
        return {"n": int(len(df)),
                "median_train_sharpe": float(df.train_sharpe.median()),
                "median_test_sharpe": float(df.test_sharpe.median()),
                "n_test_sharpe_gt_btc": int((df.test_sharpe > BTC_BENCH_TEST_SHARPE).sum()),
                "n_test_sharpe_gt_0": int((df.test_sharpe > 0).sum()),
                "max_test_sharpe": float(df.test_sharpe.max()),
                "spearman_train_vs_test_sharpe": float(
                    df.train_sharpe.rank().corr(df.test_sharpe.rank())),
                }

    summary = {"all": grid_stats(flat)}
    for h, g in flat.groupby("hypothesis"):
        summary[h] = grid_stats(g)
    # Sub-groups worth seeing (still the same grid, just sliced)
    slices = {}
    for h, g in flat.groupby("hypothesis"):
        ps = pd.DataFrame([json.loads(x) for x in g.params], index=g.index)
        for col in ps.columns:
            if ps[col].nunique() > 1:
                for val, gg in g.groupby(ps[col].astype(str)):
                    slices[f"{h}|{col}={val}"] = {
                        "n": int(len(gg)), "median_train_sharpe": float(gg.train_sharpe.median()),
                        "median_test_sharpe": float(gg.test_sharpe.median())}
    summary["slices"] = slices
    # Within each slice, what would TRAIN selection have picked, and how did it do on TEST?
    within = {}
    for h, g in flat.groupby("hypothesis"):
        ps = pd.DataFrame([json.loads(x) for x in g.params], index=g.index)
        for col in ps.columns:
            if ps[col].nunique() > 1:
                for val, gg in g.groupby(ps[col].astype(str)):
                    b = gg.loc[gg.train_sharpe.idxmax()]
                    within[f"{h}|{col}={val}"] = {"id": b.id, "train_sharpe": float(b.train_sharpe),
                                                  "test_sharpe": float(b.test_sharpe)}
    summary["train_best_within_slice"] = within
    summary["n_slices_whose_train_best_beats_btc_on_test"] = int(
        sum(v["test_sharpe"] > BTC_BENCH_TEST_SHARPE for v in within.values()))

    # --- choose on TRAIN only ----------------------------------------------------------------
    best_row = rows[int(flat.train_sharpe.idxmax())]
    w, eligible = store[best_row["id"]](d)
    chosen = deep_dive(d, best_row, w, eligible, N_PLACEBO_CHOSEN, SEED)

    per_hyp = {}
    for j, (h, g) in enumerate(flat.groupby("hypothesis")):
        r = rows[int(g.train_sharpe.idxmax())]
        if r["id"] == best_row["id"]:
            per_hyp[h] = {"same_as_chosen": True, "id": r["id"]}
            continue
        w2, e2 = store[r["id"]](d)
        per_hyp[h] = deep_dive(d, r, w2, e2, N_PLACEBO_INFO, SEED + 100 * (j + 1))

    # --- robust-candidate check (strict) --------------------------------------------------------
    tsh = chosen["test"]["sharpe"]
    pcts = [v["test_percentile"] for k, v in chosen["placebo"].items() if k != "test_only_shift"]
    yrs = chosen["test_yearly_pct"]
    checks = {
        "test_sharpe_gt_btc": tsh > BTC_BENCH_TEST_SHARPE,
        "placebo_percentile_ge_95": min(pcts) >= 95,
        "cost30_test_sharpe_gt_btc": chosen["test_sharpe_cost30"] > BTC_BENCH_TEST_SHARPE,
        "positive_years_ge_3_of_5": sum(v > 0 for v in yrs.values()) >= 3,
        "grid_median_test_sharpe_gt_0": summary["all"]["median_test_sharpe"] > 0,
        "hypothesis_median_test_sharpe_gt_0":
            summary[chosen["hypothesis"]]["median_test_sharpe"] > 0,
    }
    chosen["robust_checks"] = checks
    chosen["robust_candidate"] = all(checks.values())

    out = {
        "family": "on-chain valuation and activity signals",
        "seed": SEED, "n_configs_tested": n_configs,
        "selection": "max TRAIN Sharpe (2018-01-01..2021-12-31) over the whole grid",
        "btc_buy_hold": {"train": bench["train"], "test": bench["test"]},
        "data_coverage": coverage(panel, d),
        "grid_summary": summary,
        "chosen": chosen,
        "best_per_hypothesis": per_hyp,
        "grid": rows,
        "runtime_s": time.time() - t0,
    }
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    # --- console report ----------------------------------------------------------------------
    print("\nGrid summary (TEST):")
    for k in ("all", "a_mvrv_timing", "b_xs_mvrv", "c_activity", "d_flows"):
        s = summary[k]
        print(f"  {k:15s} n={s['n']:3d} median TRAIN {s['median_train_sharpe']:+.2f} "
              f"median TEST {s['median_test_sharpe']:+.2f}  >BTC {s['n_test_sharpe_gt_btc']:3d}  "
              f"max TEST {s['max_test_sharpe']:+.2f}  rank-corr {s['spearman_train_vs_test_sharpe']:+.2f}")
    for name, c in [("CHOSEN", chosen)] + [(h, v) for h, v in per_hyp.items()
                                           if not v.get("same_as_chosen")]:
        print(f"\n{name}: {c['id']} {c['description']}")
        for s in ("train", "test"):
            print(f"  {s:5s} Sharpe {c[s]['sharpe']:+.2f} CAGR {c[s]['cagr_pct']:+.1f}% "
                  f"total {c[s]['total_pct']:+.1f}% maxDD {c[s]['max_dd_pct']:.1f}% "
                  f"expo {c[s]['exposure']:.2f} turn/y {c[s]['turnover_per_year']:.1f}")
        print(f"  TEST Sharpe cost30 {c['test_sharpe_cost30']:+.2f} lag0 "
              f"{c['test_sharpe_lag0']:+.2f} lag2 {c['test_sharpe_lag2']:+.2f}")
        print("  TEST years " + ", ".join(f"{y}: {v:+.0f}%" for y, v in c["test_yearly_pct"].items()))
        if "same_assets_buy_hold" in c:
            print(f"  same assets buy&hold: {c['same_assets_buy_hold']}")
        for k, v in c["placebo"].items():
            trp = "n/a" if v["train_percentile"] is None else f"{v['train_percentile']:.1f}"
            print(f"  placebo {k}: TEST pct {v['test_percentile']:.1f} (ties {v['test_share_ties']:.1f}) "
                  f"TRAIN pct {trp} median placebo TEST {v['placebo_test_sharpe_median']:+.2f}")
        print("  gross exposure by year " + ", ".join(
            f"{y}: {v:.2f}" for y, v in c["gross_exposure_by_year"].items()))
    print(f"\nslices whose TRAIN-best config beats BTC on TEST: "
          f"{summary['n_slices_whose_train_best_beats_btc_on_test']} of "
          f"{len(summary['train_best_within_slice'])}")
    print("universe gaps in TEST (priced but missing mcap/volume):",
          list(out["data_coverage"]["universe_gaps_test"]))
    print("\nrobust checks:", checks, "->", chosen["robust_candidate"])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
