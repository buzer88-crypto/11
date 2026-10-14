"""Check 1: independent reimplementation of SMA50 / vol-target 40% / daily on BTC.

Does NOT import run.py and does NOT use pinetest.portfolio. Prices are read straight from the
raw Coin Metrics CSV (and compared to load_panel), backtest and metrics are written here.
"""
import json
import itertools
import numpy as np
import pandas as pd

RAW = "/home/user/coinmetrics/data/csv/{}.csv"
PERIODS = {"train": ("2018-01-01", "2021-12-31"), "test": ("2022-01-01", "2026-12-31"),
           "pre": ("2014-01-01", "2017-12-31")}
ANN = 365


def raw_price(asset):
    d = pd.read_csv(RAW.format(asset), usecols=["time", "PriceUSD"])
    s = pd.Series(d["PriceUSD"].to_numpy(), index=pd.to_datetime(d["time"]), name=asset).astype(float)
    s = s.loc[s.first_valid_index():]
    return s


def target(p, signal="sma", n=50, sizing="vt40"):
    """Target weight decided at close t with closes <= t (explicit loops, no pandas rolling)."""
    v = p.to_numpy()
    T = len(v)
    r = np.full(T, np.nan)
    r[1:] = v[1:] / v[:-1] - 1
    w = np.zeros(T)
    for t in range(T):
        if np.isnan(v[t]):
            continue
        if signal == "sma":
            if t < n - 1:
                continue
            win = v[t - n + 1:t + 1]
            if np.isnan(win).any():
                continue
            on = v[t] > win.mean()
        else:
            if t < n or np.isnan(v[t - n]):
                continue
            on = v[t] / v[t - n] - 1 > 0
        if not on:
            continue
        if sizing == "full":
            w[t] = 1.0
        else:
            if t < 60:
                continue
            rw = r[t - 59:t + 1]
            if np.isnan(rw).any():
                continue
            vol = rw.std(ddof=1) * np.sqrt(ANN)
            w[t] = min(1.0, 0.40 / vol)
    return pd.Series(w, index=p.index)


def run(p, w, cost_bps=10.0, lag=1, rebal_mask=None):
    """Single-asset long/flat. w[t] decided at close t, executed at close t+lag.
    Position over (t, t+1] is what was held after the trade at close t."""
    v = p.to_numpy()
    T = len(v)
    r = np.zeros(T)
    ok = ~np.isnan(v)
    r[1:] = np.where(ok[1:] & ok[:-1], v[1:] / np.where(ok[:-1], v[:-1], 1.0) - 1, 0.0)
    wv = w.to_numpy()
    tgt = np.zeros(T)
    tgt[lag:] = wv[:T - lag] if lag else wv
    tgt = np.where(ok, tgt, 0.0)
    trade = np.ones(T, bool)
    if rebal_mask is not None:
        m = rebal_mask.to_numpy()
        trade = np.zeros(T, bool)
        trade[lag:] = m[:T - lag] if lag else m
    pos = 0.0
    net = np.zeros(T)
    expo = np.zeros(T)
    for t in range(T):
        expo[t] = pos
        g = pos * r[t]
        drift = pos * (1 + r[t]) / (1 + g)
        new = tgt[t] if trade[t] else drift
        net[t] = (1 + g) * (1 - cost_bps * 1e-4 * abs(new - drift)) - 1
        pos = new
    return pd.Series(net, p.index), pd.Series(expo, p.index)


def metrics(ret, expo, a, b):
    x = ret.loc[a:b]
    eq = (1 + x).cumprod()
    return {"sharpe": round(float(x.mean() / x.std(ddof=1) * np.sqrt(ANN)), 3),
            "total_pct": round(float(100 * (eq.iloc[-1] - 1)), 1),
            "max_dd_pct": round(float(100 * (eq / eq.cummax() - 1).min()), 1),
            "exposure": round(float(expo.loc[a:b].mean()), 3),
            "start": str(x.index[0].date()), "end": str(x.index[-1].date()), "n_days": len(x)}


if __name__ == "__main__":
    out = {}
    btc = raw_price("btc")
    eth = raw_price("eth")
    # compare raw against the panel used by run.py
    from pinetest.crypto_data import load_panel
    px = load_panel()["PriceUSD"]
    pb = px["btc"].dropna()
    common = btc.dropna().index.intersection(pb.index)
    out["raw_vs_panel_btc_max_abs_rel_diff"] = float(((btc.loc[common] - pb.loc[common]).abs() / pb.loc[common]).max())
    out["raw_btc_span"] = [str(btc.index[0].date()), str(btc.dropna().index[-1].date()), int(btc.isna().sum())]
    # align raw to the panel calendar (panel ends 2026-05-24 with BTC NaN on the last day)
    btc = btc.reindex(px.index).loc[btc.index[0]:]
    eth = eth.reindex(px.index).loc[eth.first_valid_index():]

    res = {}
    for name, p in (("btc", btc), ("eth", eth)):
        w = target(p, "sma", 50, "vt40")
        ret, ex = run(p, w)
        bh_ret, bh_ex = run(p, pd.Series(np.where(p.notna(), 1.0, 0.0), p.index))
        res[name] = {"strategy": {k: metrics(ret, ex, *P) for k, P in PERIODS.items() if p.loc[P[0]:P[1]].notna().sum() > 100},
                     "buy_hold": {k: metrics(bh_ret, bh_ex, *P) for k, P in PERIODS.items() if p.loc[P[0]:P[1]].notna().sum() > 100}}
        if name == "btc":
            res["btc"]["sensitivity"] = {
                f"cost{c}": metrics(*run(p, w, cost_bps=c), *PERIODS["test"])["sharpe"] for c in (0, 30, 50)}
            res["btc"]["sensitivity"].update({f"lag{l}": metrics(*run(p, w, lag=l), *PERIODS["test"])["sharpe"] for l in (0, 2, 3)})
            res["btc"]["test_years"] = {str(y): round(100 * float((1 + ret.loc[str(y)]).prod() - 1), 1) for y in range(2022, 2027)}
            res["btc"]["bh_test_years"] = {str(y): round(100 * float((1 + bh_ret.loc[str(y)]).prod() - 1), 1) for y in range(2022, 2027)}
            # de-levered benchmarks: constant weight equal to mean exposure, and vol-target without trend
            for k, P in PERIODS.items():
                e = metrics(ret, ex, *P)["exposure"]
                cw_ret, cw_ex = run(p, pd.Series(np.where(p.notna(), e, 0.0), p.index))
                vt = target(p, "ret", 1, "vt40")  # placeholder replaced below
                res["btc"].setdefault("constant_weight_same_exposure", {})[k] = metrics(cw_ret, cw_ex, *P)
            # always-on vol target (no trend filter)
            v = btc.to_numpy(); r = btc.pct_change(fill_method=None)
            vol = r.rolling(60).std() * np.sqrt(ANN)
            vt_w = (0.40 / vol).clip(upper=1.0).fillna(0.0)
            vt_ret, vt_ex = run(btc, vt_w)
            res["btc"]["vol_target_only"] = {k: metrics(vt_ret, vt_ex, *P) for k, P in PERIODS.items()}
            ret.to_frame("strat").assign(bh=bh_ret, expo=ex).to_pickle(
                "/home/user/11/research/btc_trend/review_repro_lookahead/repro_btc_returns.pkl")
            w.to_pickle("/home/user/11/research/btc_trend/review_repro_lookahead/repro_btc_weights.pkl")
    out["chosen"] = res

    # grid reproduction (BTC only), with own code
    grid = []
    rw_mask = None
    for sig, n, siz, reb in itertools.product(["sma", "ret"], [50, 100, 150, 200], ["full", "vt40"], ["D", "W"]):
        w = target(btc, sig, n, siz)
        mask = None
        if reb == "W":
            mask = pd.Series(btc.index.dayofweek == 6, btc.index)  # Sundays (W-SUN period end)
        ret, ex = run(btc, w, rebal_mask=mask)
        grid.append({"cfg": f"{sig}{n}_{siz}_{reb}", **{k: metrics(ret, ex, *P) for k, P in PERIODS.items()}})
    bh = res["btc"]["buy_hold"]
    best = max(grid, key=lambda g: g["train"]["sharpe"])
    out["grid"] = {
        "best_on_train": best["cfg"],
        "n_beat_bh_sharpe": {k: sum(g[k]["sharpe"] > bh[k]["sharpe"] for g in grid) for k in PERIODS},
        "n_better_dd": {k: sum(g[k]["max_dd_pct"] > bh[k]["max_dd_pct"] for g in grid) for k in PERIODS},
        "median_sharpe": {k: float(np.median([g[k]["sharpe"] for g in grid])) for k in PERIODS},
        "train_rank_top5": [(g["cfg"], g["train"]["sharpe"], g["test"]["sharpe"]) for g in sorted(grid, key=lambda g: -g["train"]["sharpe"])[:5]],
        "test_sharpes": {g["cfg"]: g["test"]["sharpe"] for g in grid},
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open("/home/user/11/research/btc_trend/review_repro_lookahead/repro.json", "w"), indent=1)
