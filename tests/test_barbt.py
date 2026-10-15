import numpy as np
import pandas as pd
import pytest

from pinetest.barbt import backtest, summary, timing_placebo


def bars(rows, start="2024-01-01"):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    df.index = pd.date_range(start, periods=len(df), freq="h")
    return df


def flags(n, idx):
    return np.array([i in idx for i in range(n)])


FLAT = [[100, 100, 100, 100]]


def test_signal_entry_and_exit_fill_at_next_open():
    df = bars(FLAT + [[100, 101, 99, 100], [102, 103, 101, 102], [104, 105, 103, 104], [106, 106, 106, 106]])
    (t,) = backtest(df, flags(5, {0}), flags(5, {2}), cost_pct=0)
    assert (t.entry_bar, t.entry, t.exit_bar, t.exit, t.reason) == (0, 100, 3, 104, "signal")
    assert t.ret == pytest.approx(0.04)


def test_stoploss_and_gap():
    df = bars(FLAT + [[100, 101, 94, 95], [95, 95, 95, 95]])
    (t,) = backtest(df, flags(3, {0}), stoploss=0.05, cost_pct=0)
    assert (t.exit, t.reason) == (95, "stop")
    df = bars(FLAT + [[100, 101, 99, 100], [90, 91, 89, 90]])
    (t,) = backtest(df, flags(3, {0}), stoploss=0.05, cost_pct=0)
    assert (t.exit_bar, t.exit, t.reason) == (2, 90, "stop")


def test_roi_table_uses_time_in_trade():
    # 5% from the start, 1% after 2 bars.
    df = bars(FLAT + [[100, 102, 99, 101], [101, 102, 100, 101], [101, 101.5, 100, 101]])
    (t,) = backtest(df, flags(4, {0}), roi=[(0, 0.05), (2, 0.01)], cost_pct=0)
    assert (t.exit_bar, t.exit, t.reason) == (3, 101, "roi")


def test_trailing_stop_follows_the_high():
    df = bars(FLAT + [[100, 110, 100, 109], [109, 109, 104, 105], [105, 105, 105, 105]])
    (t,) = backtest(df, flags(4, {0}), trailing=0.05, cost_pct=0)
    assert (t.exit_bar, t.exit, t.reason) == (2, pytest.approx(104.5), "stop")


def test_short_and_costs():
    df = bars(FLAT + [[100, 100, 90, 90], [90, 90, 90, 90]])
    (t,) = backtest(df, flags(3, {0}), side=-1, max_hold=1, cost_pct=0.1)
    assert (t.exit, t.reason) == (90, "time")
    assert t.ret == pytest.approx(0.10 - 0.001 * (1 + 0.9))


def test_one_position_at_a_time_and_summary():
    df = bars(FLAT * 2 + [[100, 101, 99, 100]] * 6)
    trades = backtest(df, flags(8, {0, 1, 2, 5}), max_hold=3, cost_pct=0)
    assert [t.entry_bar for t in trades] == [0, 5]
    s = summary(trades, df)
    assert s["n"] == 2 and s["exposure_pct"] > 0


def test_timing_placebo_matches_drift():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(np.full(2000, 0.001)))
    o = np.r_[c[0], c[:-1]]
    df = bars(np.c_[o, np.maximum(o, c), np.minimum(o, c), c])
    trades = backtest(df, flags(2000, set(range(0, 1900, 50))), max_hold=10, cost_pct=0)
    plac = timing_placebo(df, trades, rng, draws=50, cost_pct=0)
    real = np.mean([t.ret for t in trades])
    assert plac.mean() == pytest.approx(real, rel=0.05)    # steady drift: timing does not matter
