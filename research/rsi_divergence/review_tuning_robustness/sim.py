"""Evaluate one RSI-divergence config on a timeframe, returning trade R by signal year.
Mirrors tune.evaluate (same rsi_divergences/simulate calls, cost 0.1%/side) but allows
asymmetric pivots and other timeframes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

YEARS = list(range(2014, 2027))
_DF = {}


def df_for(tf):
    if tf not in _DF:
        _DF[tf] = load(tf)
    return _DF[tf]


def evaluate(cfg, tf="4h", with_trades=False):
    df = df_for(tf)
    left = cfg.get("left", cfg["lb"])
    right = cfg.get("right", cfg["lb"])
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=left, right=right,
                             range_lower=cfg.get("range_lower", 5), range_upper=cfg.get("range_upper", 60),
                             rr=cfg["rr"], buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"],
                             confirm=cfg["confirm"], trend=cfg["trend"], extreme=cfg["extreme"],
                             sides=(1, -1) if cfg["sides"] == "both" else (1,))
    years = df.index.year.to_numpy()
    by_year = {y: [] for y in YEARS}
    trades = simulate(df, setups, COST)
    for t in trades:
        y = int(years[t.bar])
        if y in by_year:
            by_year[y].append(round(t.r, 4))
    return (by_year, trades) if with_trades else by_year


def period(by_year, years):
    r = np.array([x for y in years for x in by_year[y]], dtype=float)
    if len(r) == 0:
        return {"n": 0, "avg_r": None, "t": None, "total_r": 0.0}
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"n": int(len(r)), "avg_r": round(float(r.mean()), 3),
            "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else None,
            "total_r": round(float(r.sum()), 2)}


def row(by_year):
    return {"per_year": {y: {"n": len(by_year[y]), "avg_r": round(float(np.mean(by_year[y])), 3) if by_year[y] else None}
                         for y in YEARS},
            "2014_2016": period(by_year, range(2014, 2017)),
            "2017_2026": period(by_year, range(2017, 2027)),
            "2018_2026": period(by_year, range(2018, 2027)),
            "2014_2026": period(by_year, range(2014, 2027))}
