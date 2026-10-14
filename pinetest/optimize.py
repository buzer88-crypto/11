"""Grid search with an in-sample / out-of-sample split.

Parameters are chosen on the first `train_frac` of the data and then scored on the rest,
which the search never saw. A large gap between the two scores means overfitting.
"""
import itertools
from dataclasses import replace

import pandas as pd

from .backtest import Signals, run


def _slice(signals: Signals, start: int, stop: int) -> Signals:
    def cut(s):
        return None if s is None else s.iloc[start:stop].reset_index(drop=True)
    return replace(signals, entry=cut(signals.entry), exit=cut(signals.exit),
                   short=cut(signals.short), long_stop=cut(signals.long_stop),
                   short_stop=cut(signals.short_stop))


def backtest_range(df: pd.DataFrame, signals: Signals, start: int, stop: int, **kw):
    # Signals are causal, so computing them on the full series only adds indicator warm-up.
    return run(df.iloc[start:stop].reset_index(drop=True), _slice(signals, start, stop), **kw)


def grid_search(df: pd.DataFrame, strategy, grid: dict, train_frac: float = 0.7,
                valid=lambda p: True, **run_kw) -> pd.DataFrame:
    split = int(len(df) * train_frac)
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        if not valid(params):
            continue
        signals = strategy(df, **params)
        train = backtest_range(df, signals, 0, split, **run_kw)
        test = backtest_range(df, signals, split, len(df), **run_kw)
        rows.append({**params,
                     "train_pct": 100 * train.net_profit / train.initial_capital,
                     "test_pct": 100 * test.net_profit / test.initial_capital,
                     "train_trades": len(train.trades)})
    return pd.DataFrame(rows).sort_values("train_pct", ascending=False).reset_index(drop=True)


EMA_GRID = {"fast_len": range(3, 31, 2), "slow_len": range(10, 101, 5)}


def ema_valid(p):
    return p["fast_len"] < p["slow_len"]


if __name__ == "__main__":
    import sys

    from .strategies import ema_cross

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m pinetest.optimize <ohlc.csv>")
    data = pd.read_csv(sys.argv[1])
    data.columns = [c.strip().lower() for c in data.columns]
    result = grid_search(data, ema_cross, EMA_GRID, valid=ema_valid)
    print(result.head(15).round(2).to_string(index=False))
    print(f"\nTrain/test correlation: {result.train_pct.corr(result.test_pct):.2f}")
