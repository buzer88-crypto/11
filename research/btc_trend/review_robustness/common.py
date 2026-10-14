"""Shared helpers for the robustness review (reproduces run.py's chosen config)."""
import numpy as np
import pandas as pd

from pinetest.crypto_data import load_panel
from pinetest.portfolio import TEST, TRAIN, backtest, buy_and_hold, rebalance_on

PRE = ("2014-01-01", "2017-12-31")
panel = load_panel()
PX = panel["PriceUSD"][["btc"]]  # only BTC is ever held; identical results, much faster
P = PX["btc"]
RET = P.pct_change(fill_method=None)
VOL60 = RET.rolling(60).std() * np.sqrt(365)


def trend_w(n=50, vt=0.40, signal="sma", rebal="D", px=PX):
    p = px["btc"]
    on = (p > p.rolling(n).mean()) if signal == "sma" else (p / p.shift(n) - 1 > 0)
    size = pd.Series(1.0, index=p.index) if vt is None else (vt / VOL60.reindex(p.index)).clip(upper=1.0)
    w = pd.DataFrame({"btc": (on.astype(float) * size).where(p.notna(), 0.0).fillna(0.0)})
    return rebalance_on(w, "W-SUN") if rebal == "W" else w


def static_w(frac, rebal="W"):
    w = pd.DataFrame({"btc": P.notna().astype(float) * frac})
    return rebalance_on(w, "W-SUN") if rebal == "W" else w


def vt_only_w(vt=0.40, rebal="D"):
    w = pd.DataFrame({"btc": (vt / VOL60).clip(upper=1.0).where(P.notna(), 0.0).fillna(0.0)})
    return rebalance_on(w, "W-SUN") if rebal == "W" else w


def bt(w, **kw):
    return backtest(w, PX, **kw)


def stats(r: pd.Series):
    r = r.dropna()
    eq = (1 + r).cumprod()
    sd = r.std()
    yrs = len(r) / 365
    return {"sharpe": round(float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, 3),
            "total_pct": round(100 * (eq.iloc[-1] - 1), 1),
            "cagr_pct": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
            "vol_pct": round(100 * sd * np.sqrt(365), 1),
            "max_dd_pct": round(100 * float((eq / eq.cummax() - 1).min()), 1)}


def met(res, start, end):
    x = res.metrics(start, end)
    return {k: round(x[k], 3) for k in ("sharpe", "total_pct", "cagr_pct", "vol_pct", "max_dd_pct", "exposure")}


BH = bt(buy_and_hold(PX, "btc"))
STRAT_W = trend_w()
STRAT = bt(STRAT_W)
