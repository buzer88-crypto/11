"""Mean reversion in crypto: full fixed grid, TRAIN-only selection, placebos, robustness.

Run:  cd /home/user/11 && python3 research/mean_reversion/run.py

Hypotheses (every weight row t is decided with data <= close t; the shared harness trades it at
close t+lag, lag=1 by default, 10 bps per side, 5%/yr fee on shorts):

  ts_mr    (a) time-series reversal. z = k-day log return / (60-day daily vol measured BEFORE the
           move * sqrt(k)). A universe member with z < -thr is bought ("long" side), one with
           z > +thr is shorted ("short" side), or both ("both"). Each triggered coin gets a fixed
           1/top_n of equity and the position is held for H days after the last trigger.
  xs_rev   (b) cross-sectional reversal. Rank the universe by L-day return (raw, or in vol units)
           and buy the k worst ("long", 1/k each) or buy the k worst and short the k best ("ls",
           +-0.5/k each, dollar neutral). Rebalanced daily ("D") or weekly on Sundays ("W").
  pair     (c) BTC/ETH relative value. z of log(ETH/BTC) against its rolling W-day mean and std.
           z < -e: ETH cheap -> long ETH / short BTC; z > +e: the reverse; exit when z crosses 0.
           "pair" = +-0.5 dollar neutral; "rotate" = 100% long the cheap coin only, cash when flat.
  alt_btc  (d) alt vs BTC. z of the alt's k-day log return minus BTC's, in units of the 60-day
           vol of the daily alt-minus-BTC return (measured before the move). z < -thr: long the
           alt and short BTC by the same amount, 1/(2*(top_n-1)) per alt, held H days.

The grid is fixed up front and run in full. The "best" config is the highest TRAIN Sharpe over
the whole grid. TEST (2022-01-01..) is only reported. Everything is seeded.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinetest.crypto_data import load_panel, universe  # noqa: E402
from pinetest.portfolio import (TEST, TRAIN, backtest, buy_and_hold, circular_shift,  # noqa: E402
                                equal_weight, permute_assets, rebalance_on)

OUT = Path(__file__).resolve().parent
TMP = Path("/tmp/claude-0/research-mean_reversion")
SEED = 20261014
# Trading starts at the last Sunday before TRAIN. Signals still use the full earlier history
# (vol windows are warm). Earlier starts let a thin 2010-2017 universe dominate nothing useful.
START = "2017-12-31"
BTC_TEST_SHARPE = 0.48
VOL_WIN = 60          # trailing window of daily log-return vol (a, b-vol, d), fixed up front
VOL_MINP = 40
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
TRAIN_YEARS = (2018, 2019, 2020, 2021)

# ---- parameter grid, fixed up front ----------------------------------------------------------
TS_K = (1, 3, 7)
TS_H = (1, 3, 7)
TS_THR = (1.5, 2.0, 2.5)
TS_SIDE = ("long", "short", "both")
TS_TOPN = (10, 20)

XS_L = (1, 2, 3)
XS_K = (3, 5)
XS_SIDE = ("long", "ls")
XS_FREQ = ("D", "W")
XS_SCORE = ("raw", "vol")
XS_TOPN = 20

PAIR_W = (30, 90, 180)
PAIR_E = (1.0, 1.5, 2.0, 2.5)
PAIR_MODE = ("pair", "rotate")

AB_K = (3, 7, 14)
AB_THR = (1.5, 2.0, 2.5)
AB_H = (3, 7, 14)
AB_TOPN = (10, 20)

N_PLACEBO_CHOSEN = 300   # per placebo type, TRAIN-chosen config
N_PLACEBO_HYP = 200      # per placebo type, TRAIN-best config of each other hypothesis

# Placebos that fit each hypothesis. The official percentile is the MIN over this set (strict).
PLACEBOS = {
    "ts_mr": ("circular_shift", "permute_assets"),        # timing + which coin
    "xs_rev": ("permute_assets", "circular_shift"),       # selection + timing
    "pair": ("joint_shift", "circular_shift"),            # timing (pair kept intact / per leg)
    "alt_btc": ("alt_shift_rehedge", "permute_alts_keep_btc"),  # timing + which alt, hedge kept
}
SUPPLEMENTARY_PLACEBOS = {"alt_btc": ("circular_shift",)}


def build_grid():
    g = []
    for k in TS_K:
        for H in TS_H:
            for thr in TS_THR:
                for side in TS_SIDE:
                    for n in TS_TOPN:
                        g.append(dict(hyp="ts_mr", k=k, H=H, thr=thr, side=side, top_n=n))
    for L in XS_L:
        for k in XS_K:
            for side in XS_SIDE:
                for f in XS_FREQ:
                    for sc in XS_SCORE:
                        g.append(dict(hyp="xs_rev", L=L, k=k, side=side, freq=f, score=sc,
                                      top_n=XS_TOPN))
    for W in PAIR_W:
        for e in PAIR_E:
            for mode in PAIR_MODE:
                g.append(dict(hyp="pair", W=W, entry=e, mode=mode, top_n=2))
    for k in AB_K:
        for thr in AB_THR:
            for H in AB_H:
                for n in AB_TOPN:
                    g.append(dict(hyp="alt_btc", k=k, thr=thr, H=H, top_n=n))
    for c in g:
        c["id"] = cfg_id(c)
    return g


def cfg_id(c):
    h = c["hyp"]
    if h == "ts_mr":
        return f"ts_mr_k{c['k']}_H{c['H']}_z{c['thr']}_{c['side']}_n{c['top_n']}"
    if h == "xs_rev":
        return f"xs_rev_L{c['L']}_k{c['k']}_{c['side']}_{c['freq']}_{c['score']}_n{c['top_n']}"
    if h == "pair":
        return f"pair_W{c['W']}_e{c['entry']}_{c['mode']}"
    return f"alt_btc_k{c['k']}_z{c['thr']}_H{c['H']}_n{c['top_n']}"


# ---- signals: panel -> daily target weights (data <= t only) ---------------------------------

def _logp(panel):
    px = panel["PriceUSD"]
    return np.log(px.where(px > 0))


def _vol(r1):
    return r1.rolling(VOL_WIN, min_periods=VOL_MINP).std()


def _hold(trigger: pd.DataFrame, H: int) -> pd.DataFrame:
    """True on t if the trigger fired on any of t-H+1..t (all past/current)."""
    return trigger.astype(float).rolling(H, min_periods=1).max() > 0


def w_ts_mr(panel, members, c):
    lp = _logp(panel)
    r1 = lp.diff()
    k = c["k"]
    z = (lp - lp.shift(k)) / (_vol(r1).shift(k) * np.sqrt(k))
    pos = pd.DataFrame(0.0, index=lp.index, columns=lp.columns)
    if c["side"] in ("long", "both"):
        pos += _hold((z < -c["thr"]) & members, c["H"]).astype(float)
    if c["side"] in ("short", "both"):
        pos -= _hold((z > c["thr"]) & members, c["H"]).astype(float)
    return pos.where(members, 0.0) / c["top_n"]


def w_xs_rev(panel, members, c):
    lp = _logp(panel)
    L, k = c["L"], c["k"]
    score = lp - lp.shift(L)
    if c["score"] == "vol":
        score = score / (_vol(lp.diff()).shift(L) * np.sqrt(L))
    s = score.where(members & np.isfinite(score))
    n_valid = s.notna().sum(axis=1)
    worst = s.rank(axis=1, ascending=True, method="first") <= k
    best = s.rank(axis=1, ascending=False, method="first") <= k
    if c["side"] == "long":
        w = worst.astype(float) / k
        w = w.mul((n_valid >= k).astype(float), axis=0)
    else:
        w = (0.5 / k) * worst.astype(float) - (0.5 / k) * best.astype(float)
        w = w.mul((n_valid >= 2 * k).astype(float), axis=0)
    return w


def pair_state(z: np.ndarray, entry: float) -> np.ndarray:
    """+1 = ETH cheap (long ETH / short BTC), -1 = ETH expensive. Exit when z crosses 0."""
    st = np.zeros(len(z))
    s = 0
    for i, v in enumerate(z):
        if np.isnan(v):
            s = 0
        else:
            if s == 1 and v >= 0:
                s = 0
            elif s == -1 and v <= 0:
                s = 0
            if s == 0:
                if v < -entry:
                    s = 1
                elif v > entry:
                    s = -1
        st[i] = s
    return st


def w_pair(panel, members, c):
    px = panel["PriceUSD"]
    x = np.log(px["eth"] / px["btc"])
    W = c["W"]
    z = (x - x.rolling(W, min_periods=W).mean()) / x.rolling(W, min_periods=W).std()
    st = pd.Series(pair_state(z.to_numpy(), c["entry"]), index=px.index)
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    if c["mode"] == "pair":
        w["eth"], w["btc"] = 0.5 * st, -0.5 * st
    else:
        w["eth"], w["btc"] = (st > 0).astype(float), (st < 0).astype(float)
    return w


def w_alt_btc(panel, members, c):
    lp = _logp(panel)
    r1 = lp.diff()
    k = c["k"]
    rel_k = (lp - lp.shift(k)).sub(lp["btc"] - lp["btc"].shift(k), axis=0)
    vol_rel = r1.sub(r1["btc"], axis=0).rolling(VOL_WIN, min_periods=VOL_MINP).std()
    z = rel_k / (vol_rel.shift(k) * np.sqrt(k))
    alts = members.copy()
    alts["btc"] = False
    alts = alts.mul(members["btc"], axis=0).astype(bool)
    act = _hold((z < -c["thr"]) & alts, c["H"]) & alts
    w = act.astype(float) / (2 * (c["top_n"] - 1))
    w["btc"] = -w.sum(axis=1)
    return w


BUILDERS = {"ts_mr": w_ts_mr, "xs_rev": w_xs_rev, "pair": w_pair, "alt_btc": w_alt_btc}


def daily_weights(panel, members, c):
    return BUILDERS[c["hyp"]](panel, members, c)


def strategy(panel, members, c):
    w = daily_weights(panel, members, c).loc[START:]
    if c["hyp"] == "xs_rev" and c["freq"] == "W":
        w = rebalance_on(w, "W-SUN")
    return w


# ---- placebos ---------------------------------------------------------------------------------

def joint_shift(weights, rng, min_shift=90):
    """Timing placebo for a pair: rotate the whole weight matrix by ONE random offset, so the
    two legs stay together (the per-asset circular_shift would break the hedge)."""
    held = weights.ffill().fillna(0.0).to_numpy()
    k = rng.integers(min_shift, len(held) - min_shift)
    out = pd.DataFrame(np.roll(held, k, axis=0), index=weights.index, columns=weights.columns)
    out.loc[~weights.notna().any(axis=1).to_numpy()] = np.nan
    return out


def alt_shift_rehedge(weights, prices, rng):
    """Timing placebo for alt_btc: circular_shift only the alt legs, then rebuild the BTC hedge
    as minus the sum of the shifted alt longs (keeps the alt-vs-BTC structure)."""
    alt = weights.copy()
    alt["btc"] = 0.0
    out = circular_shift(alt, prices, rng)
    out["btc"] = 0.0
    out["btc"] = -out.fillna(0.0).sum(axis=1)
    return out


def permute_alts_keep_btc(weights, members, rng):
    """Selection placebo for alt_btc: BTC keeps its weight on every row, the alt weights go to
    random non-BTC universe members."""
    j = weights.columns.get_loc("btc")
    m = members.reindex_like(weights).fillna(False).to_numpy().copy()
    m[:, j] = False
    w = weights.to_numpy()
    res = w.copy()
    for t in np.flatnonzero(~np.isnan(w).all(axis=1)):
        vals = np.nan_to_num(w[t])
        alt = vals.copy()
        alt[j] = 0.0
        nz = alt[alt != 0]
        pool = np.flatnonzero(m[t])
        row = np.zeros(len(vals))
        row[j] = vals[j]
        if len(nz) and len(pool) >= len(nz):
            row[rng.choice(pool, len(nz), replace=False)] = rng.permutation(nz)
        res[t] = row
    return pd.DataFrame(res, index=weights.index, columns=weights.columns)


def make_placebo(kind, weights, members, px, rng):
    if kind == "circular_shift":
        return circular_shift(weights, px, rng)
    if kind == "permute_assets":
        return permute_assets(weights, members, rng)
    if kind == "joint_shift":
        return joint_shift(weights, rng)
    if kind == "alt_shift_rehedge":
        return alt_shift_rehedge(weights, px, rng)
    if kind == "permute_alts_keep_btc":
        return permute_alts_keep_btc(weights, members, rng)
    raise ValueError(kind)


# permute_* placebos re-draw the coins on EVERY non-NaN row. Strategies here emit a row every day
# while holding for H days, so a permuted placebo trades far more than the real strategy and pays
# more costs, which flatters the real percentile. For these kinds we also compare gross of costs
# (cost_bps=0 for both real and placebo) and the official percentile takes the minimum.
GROSS_KINDS = {"permute_assets", "permute_alts_keep_btc"}


def run_placebo(kind, weights, members, px, n, rng, real, real_gross=None):
    tr, te, te0, tr0 = [], [], [], []
    for _ in range(n):
        pw = make_placebo(kind, weights, members, px, rng)
        m = backtest(pw, px).split_metrics()
        tr.append(m["train"].get("sharpe", 0.0))
        te.append(m["test"].get("sharpe", 0.0))
        if kind in GROSS_KINDS:
            m0 = backtest(pw, px, cost_bps=0).split_metrics()
            tr0.append(m0["train"].get("sharpe", 0.0))
            te0.append(m0["test"].get("sharpe", 0.0))
    tr, te = np.array(tr), np.array(te)
    out = {
        "n": n,
        "test_percentile": float(100 * np.mean(te < real["test"]["sharpe"])),
        "train_percentile": float(100 * np.mean(tr < real["train"]["sharpe"])),
        "test_placebo_median": float(np.median(te)),
        "test_placebo_p95": float(np.quantile(te, 0.95)),
        "train_placebo_median": float(np.median(tr)),
    }
    if kind in GROSS_KINDS:
        te0, tr0 = np.array(te0), np.array(tr0)
        out.update({
            "gross_test_percentile": float(100 * np.mean(te0 < real_gross["test"]["sharpe"])),
            "gross_train_percentile": float(100 * np.mean(tr0 < real_gross["train"]["sharpe"])),
            "gross_test_placebo_median": float(np.median(te0)),
            "gross_test_placebo_p95": float(np.quantile(te0, 0.95)),
            "real_gross_test_sharpe": float(real_gross["test"]["sharpe"]),
        })
    return out


def official_percentile(plc, kinds):
    vals = []
    for k in kinds:
        vals.append(plc[k]["test_percentile"])
        if "gross_test_percentile" in plc[k]:
            vals.append(plc[k]["gross_test_percentile"])
    return float(min(vals))


# ---- helpers ------------------------------------------------------------------------------------

def clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in d.items()}


def yearly_returns(ret, years):
    out = {}
    for y in years:
        r = ret.loc[str(y)]
        out[str(y)] = float(100 * ((1 + r).prod() - 1)) if len(r) else None
    return out


def beta_alpha(ret, bench, start, end):
    r, b = ret.loc[start:end].to_numpy(), bench.loc[start:end].to_numpy()
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1]
    resid = r - beta * b
    return {"beta": float(beta), "alpha_ann_pct": float(100 * resid.mean() * 365),
            "resid_sharpe": float(resid.mean() / resid.std() * np.sqrt(365)) if resid.std() > 0
            else 0.0,
            "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))}


def members_for(c, members):
    return members[c["top_n"]] if c["top_n"] in members else members[20]


def check_no_lookahead(panel, members_fn, cfgs, cut="2023-06-18"):
    """Weights up to `cut` must not change when every panel field is truncated at `cut`."""
    cut = pd.Timestamp(cut)
    trunc = {k: v.loc[:cut] for k, v in panel.items()}
    for c in cfgs:
        n = c["top_n"] if c["top_n"] >= 10 else 20
        full = daily_weights(panel, universe(panel, top_n=n), c).loc[:cut]
        part = daily_weights(trunc, universe(trunc, top_n=n), c)
        pd.testing.assert_frame_equal(full, part, check_freq=False)


def event_study(panel, members, px_full):
    """Diagnostic only (uses future returns to MEASURE, never to trade): mean return of coins
    after a k-day move beyond +-2 vol units, in excess of the equal-weight universe over the same
    window. 'd0' = the day right after the signal close (what lag=0 would catch); 'lag1_Hd' =
    the H days that a lag=1 trade actually holds."""
    lp = _logp(panel)
    r1 = lp.diff()
    vol = _vol(r1)
    mem = members
    rets = px_full.pct_change(fill_method=None)
    ew = rets.where(mem.shift(1, fill_value=False)).mean(axis=1)
    out = {}
    for k in (1, 3, 7):
        z = (lp - lp.shift(k)) / (vol.shift(k) * np.sqrt(k))
        for direction, trig in (("drop", (z < -2) & mem), ("pump", (z > 2) & mem)):
            for label, off, H in (("d0", 1, 1), ("lag1_1d", 2, 1), ("lag1_3d", 2, 3),
                                  ("lag1_7d", 2, 7)):
                # forward log return from close t+off-1 to close t+off-1+H, minus EW over same days
                fwd = lp.shift(-(off - 1 + H)) - lp.shift(-(off - 1))
                ew_fwd = np.log1p(ew.fillna(0.0)).rolling(H).sum().shift(-(off - 1 + H))
                ex = fwd.sub(ew_fwd, axis=0).where(trig)
                for per, (a, b) in (("train", TRAIN), ("test", TEST)):
                    v = ex.loc[a:b].stack().dropna().to_numpy()
                    if len(v) < 5:
                        continue
                    out[f"k{k}_{direction}_{label}_{per}"] = {
                        "n_events": int(len(v)),
                        "mean_excess_bps": float(1e4 * v.mean()),
                        "t_naive": float(v.mean() / v.std() * np.sqrt(len(v))),
                        "hit_rate_pct": float(100 * np.mean(v > 0)),
                    }
    return out


def row_metrics(c, w, px):
    res = backtest(w, px)
    sm = res.split_metrics()
    row = dict(c)
    for split in ("train", "test", "full"):
        m = sm[split]
        for key in ("sharpe", "cagr_pct", "total_pct", "max_dd_pct", "vol_pct",
                    "turnover_per_year", "exposure"):
            row[f"{split}_{key}"] = float(m.get(key, np.nan))
    r0 = backtest(w, px, cost_bps=0).split_metrics()
    row["train_sharpe_cost0"] = float(r0["train"]["sharpe"])
    row["test_sharpe_cost0"] = float(r0["test"]["sharpe"])
    row["test_sharpe_cost30"] = float(backtest(w, px, cost_bps=30).metrics(*TEST)["sharpe"])
    row["test_sharpe_lag0"] = float(backtest(w, px, lag=0).metrics(*TEST)["sharpe"])
    row["died"] = bool((res.returns <= -1).any())
    return row, res


def summarize(d):
    return {
        "n": int(len(d)),
        "median_train_sharpe": float(d.train_sharpe.median()),
        "median_test_sharpe": float(d.test_sharpe.median()),
        "mean_test_sharpe": float(d.test_sharpe.mean()),
        "median_test_sharpe_cost0": float(d.test_sharpe_cost0.median()),
        "median_test_sharpe_cost30": float(d.test_sharpe_cost30.median()),
        "median_test_sharpe_lag0": float(d.test_sharpe_lag0.median()),
        "median_train_sharpe_cost0": float(d.train_sharpe_cost0.median()),
        "n_test_sharpe_gt_btc_0.48": int((d.test_sharpe > BTC_TEST_SHARPE).sum()),
        "n_test_sharpe_gt_0": int((d.test_sharpe > 0).sum()),
        "n_test_cost0_sharpe_gt_0": int((d.test_sharpe_cost0 > 0).sum()),
        "n_test_cost30_sharpe_gt_btc_0.48": int((d.test_sharpe_cost30 > BTC_TEST_SHARPE).sum()),
        "median_test_turnover_per_year": float(d.test_turnover_per_year.median()),
        "median_test_exposure": float(d.test_exposure.median()),
        "n_died": int(d.died.sum()),
        "spearman_train_test_sharpe": float(d.train_sharpe.rank().corr(d.test_sharpe.rank()))
        if len(d) > 2 else None,
        "best_test_sharpe": float(d.test_sharpe.max()),
        "best_test_id": d.loc[d.test_sharpe.idxmax(), "id"],
        "best_train_id": d.loc[d.train_sharpe.idxmax(), "id"],
    }


def deep_dive(c, panel, members, px, rng, n_placebo, grid_df, btc_rets):
    mem = members_for(c, members)
    w = strategy(panel, mem, c)
    res = backtest(w, px)
    sm = res.split_metrics()
    out = {"config": c, "train": clean(sm["train"]), "test": clean(sm["test"]),
           "full": clean(sm["full"])}
    t0 = time.time()
    sm_gross = backtest(w, px, cost_bps=0).split_metrics()
    plc = {}
    for kind in PLACEBOS[c["hyp"]] + SUPPLEMENTARY_PLACEBOS.get(c["hyp"], ()):
        plc[kind] = run_placebo(kind, w, mem, px, n_placebo, rng, sm, sm_gross)
    out["placebos"] = plc
    official = official_percentile(plc, PLACEBOS[c["hyp"]])
    out["placebo_official_kinds"] = list(PLACEBOS[c["hyp"]])
    out["placebo_official_rule"] = ("min over the listed kinds of the net TEST percentile and, "
                                    "for permute kinds, the gross (cost 0 vs cost 0) percentile")
    out["placebo_test_percentile_official_min"] = official
    out["placebo_seconds"] = round(time.time() - t0, 1)
    out["test_sharpe_cost0"] = float(backtest(w, px, cost_bps=0).metrics(*TEST)["sharpe"])
    out["test_sharpe_cost30"] = float(backtest(w, px, cost_bps=30).metrics(*TEST)["sharpe"])
    out["test_sharpe_lag0"] = float(backtest(w, px, lag=0).metrics(*TEST)["sharpe"])
    out["test_sharpe_lag2"] = float(backtest(w, px, lag=2).metrics(*TEST)["sharpe"])
    out["train_sharpe_cost30"] = float(backtest(w, px, cost_bps=30).metrics(*TRAIN)["sharpe"])
    out["test_yearly_returns_pct"] = yearly_returns(res.returns, TEST_YEARS)
    out["train_yearly_returns_pct"] = yearly_returns(res.returns, TRAIN_YEARS)
    out["vs_btc_test"] = beta_alpha(res.returns, btc_rets, *TEST)
    out["vs_btc_train"] = beta_alpha(res.returns, btc_rets, *TRAIN)
    held = res.held
    for name, (a, b) in (("train", TRAIN), ("test", TEST)):
        h = held.loc[a:b]
        out[f"positions_{name}"] = {
            "share_days_invested": float((h.abs().sum(axis=1) > 1e-9).mean()),
            "avg_gross": float(h.abs().sum(axis=1).mean()),
            "avg_net": float(h.sum(axis=1).mean()),
            "avg_n_long": float((h > 1e-9).sum(axis=1).mean()),
            "avg_n_short": float((h < -1e-9).sum(axis=1).mean()),
        }
    fam = grid_df[grid_df.hyp == c["hyp"]]
    yrs = out["test_yearly_returns_pct"]
    pos_years = sum(1 for v in yrs.values() if v is not None and v > 0)
    checks = {
        "test_sharpe_gt_0.48": bool(sm["test"]["sharpe"] > BTC_TEST_SHARPE),
        "placebo_pct_ge_95": bool(official >= 95),
        "cost30_test_sharpe_gt_0.48": bool(out["test_sharpe_cost30"] > BTC_TEST_SHARPE),
        "positive_test_years_ge_3_of_5": bool(pos_years >= 3),
        "full_grid_median_test_sharpe_gt_0": bool(grid_df.test_sharpe.median() > 0),
        "hypothesis_grid_median_test_sharpe_gt_0": bool(fam.test_sharpe.median() > 0),
    }
    out["positive_test_years"] = pos_years
    out["robust_checks"] = checks
    out["robust_candidate"] = bool(all(checks.values()))
    return out


def main():
    t_start = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    px = panel["PriceUSD"].loc[START:]          # backtest window; signals use the full panel
    members = {n: universe(panel, top_n=n) for n in (10, 20)}
    grid = build_grid()
    print(f"grid: {len(grid)} configs")

    probe = [c for c in grid if c["id"] in (
        "ts_mr_k3_H3_z2.0_both_n20", "ts_mr_k1_H1_z1.5_long_n10",
        "xs_rev_L1_k5_ls_D_vol_n20", "xs_rev_L3_k3_long_W_raw_n20",
        "pair_W90_e1.5_pair", "pair_W30_e2.0_rotate", "alt_btc_k7_z2.0_H7_n20")]
    check_no_lookahead(panel, members, probe)
    print(f"lookahead truncation check passed on {len(probe)} configs")

    bench = {"btc_buy_hold": backtest(buy_and_hold(px, "btc"), px),
             "eth_buy_hold": backtest(buy_and_hold(px, "eth"), px),
             "ew_top20_weekly": backtest(equal_weight(members[20].loc[START:], "W-SUN"), px)}
    btc_rets = bench["btc_buy_hold"].returns
    bench_out = {k: {s: clean(v) for s, v in r.split_metrics().items()} for k, r in bench.items()}
    for k, v in bench_out.items():
        print(f"{k:16s} train SR {v['train']['sharpe']:.2f} test SR {v['test']['sharpe']:.2f}"
              f" test total {v['test']['total_pct']:.0f}%")

    rows = []
    for i, c in enumerate(grid):
        w = strategy(panel, members_for(c, members), c)
        row, _ = row_metrics(c, w, px)
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i + 1}/{len(grid)} {c['id']}: train SR {row['train_sharpe']:.2f} "
                  f"test SR {row['test_sharpe']:.2f} (cost0 {row['test_sharpe_cost0']:.2f}, "
                  f"turnover {row['test_turnover_per_year']:.0f}x/yr)  [{time.time() - t_start:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grid.csv", index=False)

    summary = {"all": summarize(df)}
    for h, d in df.groupby("hyp", sort=False):
        summary[h] = summarize(d)
    for col in ("side", "k", "H", "thr", "top_n"):
        for v, d in df[df.hyp == "ts_mr"].groupby(col):
            summary[f"ts_mr_{col}={v}"] = summarize(d)
    for col in ("side", "freq", "score", "L", "k"):
        for v, d in df[df.hyp == "xs_rev"].groupby(col):
            summary[f"xs_rev_{col}={v}"] = summarize(d)
    for col in ("mode", "W", "entry"):
        for v, d in df[df.hyp == "pair"].groupby(col):
            summary[f"pair_{col}={v}"] = summarize(d)
    for col in ("k", "H", "thr", "top_n"):
        for v, d in df[df.hyp == "alt_btc"].groupby(col):
            summary[f"alt_btc_{col}={v}"] = summarize(d)

    # TRAIN-only selection over the whole grid.
    best = df.loc[df.train_sharpe.idxmax()]
    best_cfg = next(c for c in grid if c["id"] == best["id"])
    print(f"\nTRAIN-chosen: {best_cfg['id']} train SR {best.train_sharpe:.2f} "
          f"test SR {best.test_sharpe:.2f}")
    rng = np.random.default_rng(SEED)
    chosen = deep_dive(best_cfg, panel, members, px, rng, N_PLACEBO_CHOSEN, df, btc_rets)
    print(json.dumps({k: chosen[k] for k in (
        "test", "placebos", "test_sharpe_cost30", "test_sharpe_lag0", "test_yearly_returns_pct",
        "robust_checks")}, indent=1, default=str))

    # Supplementary: TRAIN-best config within each hypothesis (still TRAIN-only selection).
    per_hyp = {}
    for h, d in df.groupby("hyp", sort=False):
        bid = d.loc[d.train_sharpe.idxmax(), "id"]
        if bid == best_cfg["id"]:
            per_hyp[h] = {"same_as_chosen": True, "id": bid}
            continue
        cfg = next(c for c in grid if c["id"] == bid)
        per_hyp[h] = deep_dive(cfg, panel, members, px, np.random.default_rng(SEED + len(per_hyp) + 1),
                               N_PLACEBO_HYP, df, btc_rets)
        dd = per_hyp[h]
        print(f"{h:8s} TRAIN-best {bid}: train SR {dd['train']['sharpe']:.2f} test SR "
              f"{dd['test']['sharpe']:.2f} placebo {dd['placebo_test_percentile_official_min']:.0f}"
              f" cost30 {dd['test_sharpe_cost30']:.2f} lag0 {dd['test_sharpe_lag0']:.2f}")

    events = event_study(panel, members[20], panel["PriceUSD"])

    results = {
        "description": __doc__,
        "settings": {"seed": SEED, "start": START, "train": TRAIN, "test": TEST,
                     "default_cost_bps": 10, "default_lag": 1, "short_fee_annual": 0.05,
                     "vol_window": VOL_WIN, "n_placebo_chosen_per_type": N_PLACEBO_CHOSEN,
                     "n_placebo_per_hypothesis_best": N_PLACEBO_HYP,
                     "placebo_kinds": {k: list(v) for k, v in PLACEBOS.items()},
                     "grid": {"ts_mr": dict(k=TS_K, H=TS_H, thr=TS_THR, side=TS_SIDE, top_n=TS_TOPN),
                              "xs_rev": dict(L=XS_L, k=XS_K, side=XS_SIDE, freq=XS_FREQ,
                                             score=XS_SCORE, top_n=XS_TOPN),
                              "pair": dict(W=PAIR_W, entry=PAIR_E, mode=PAIR_MODE),
                              "alt_btc": dict(k=AB_K, thr=AB_THR, H=AB_H, top_n=AB_TOPN)}},
        "n_configs_tested": int(len(df)),
        "benchmarks": bench_out,
        "grid_summary": summary,
        "chosen": chosen,
        "per_hypothesis_train_best": per_hyp,
        "event_study_top20_z2": events,
        "grid": df.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nwrote {OUT / 'results.json'} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
