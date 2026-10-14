import pandas as pd
import pytest

from pinetest.backtest import Signals, run
from pinetest.strategies import ema_cross, rsi_reversion

from .conftest import make_ohlc


def flags(n, idx):
    return pd.Series([i in idx for i in range(n)])


def test_market_orders_fill_next_open_with_commission():
    df = make_ohlc([100, 100, 110, 120, 130])
    res = run(df, Signals(entry=flags(5, {0}), exit=flags(5, {2})), commission_pct=0.1)
    (t,) = res.trades
    assert (t.entry_bar, t.entry_price) == (1, 100)   # open of bar 1 = close of bar 0
    assert (t.exit_bar, t.exit_price, t.reason) == (3, 110, "signal")
    qty = 10_000 / (100 * 1.001)
    assert t.pnl == pytest.approx(qty * 110 * 0.999 - 10_000)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_stop_loss_triggers_from_bar_after_entry():
    df = make_ohlc([100, 100, 100, 100], spread=0)
    df.loc[1, "low"] = 50          # entry bar itself: stop not active yet
    df.loc[2, "low"] = 90
    res = run(df, Signals(entry=flags(4, {0}), exit=flags(4, set()), stop_pct=5, limit_pct=10),
              commission_pct=0)
    (t,) = res.trades
    assert (t.exit_bar, t.exit_price, t.reason) == (2, 95, "stop")


def test_gap_through_stop_fills_at_open():
    df = make_ohlc([100, 100, 100, 80, 80], spread=0)
    df.loc[3, ["open", "high"]] = 80   # bar opens below the 95 stop
    res = run(df, Signals(entry=flags(5, {0}), exit=flags(5, set()), stop_pct=5), commission_pct=0)
    assert (res.trades[0].exit_price, res.trades[0].reason) == (80, "stop")


@pytest.mark.parametrize("open_, expected", [(108, "limit"), (96, "stop")])
def test_both_levels_in_one_bar_follow_tradingview_path(open_, expected):
    df = make_ohlc([100, 100, 100, 100], spread=0)
    df.loc[2, ["open", "high", "low"]] = [open_, 115, 90]
    res = run(df, Signals(entry=flags(4, {0}), exit=flags(4, set()), stop_pct=5, limit_pct=10),
              commission_pct=0)
    assert res.trades[0].reason == expected


def test_open_position_is_closed_at_end():
    df = make_ohlc([100, 100, 105])
    res = run(df, Signals(entry=flags(3, {0}), exit=flags(3, set())), commission_pct=0)
    assert res.trades[0].reason == "end"
    assert res.net_profit == pytest.approx(10_000 * 0.05)


def test_ema_cross_profits_on_v_shaped_trend():
    closes = list(range(200, 100, -2)) + list(range(100, 300, 2))
    res = run(make_ohlc(closes), ema_cross(make_ohlc(closes)))
    assert len(res.trades) == 1
    assert res.net_profit > 0


def test_rsi_reversion_buys_dip_and_takes_profit():
    closes = [100] * 20 + list(range(100, 70, -2)) + list(range(70, 120, 1))
    df = make_ohlc(closes, spread=0.2)
    res = run(df, rsi_reversion(df))
    assert res.trades and res.trades[0].reason == "limit"
    assert res.trades[0].pnl > 0


def test_no_lookahead(random_walk):
    """Results on a prefix must not change when future bars are appended."""
    for strat in (ema_cross, rsi_reversion):
        full = run(random_walk, strat(random_walk))
        prefix = random_walk.iloc[:300]
        part = run(prefix, strat(prefix))
        closed_full = [t for t in full.trades if t.exit_bar < 299]
        closed_part = [t for t in part.trades if t.reason != "end"]
        assert closed_full == closed_part


def test_summary_is_sane(random_walk):
    s = run(random_walk, ema_cross(random_walk)).summary()
    assert s["trades"] > 0
    assert 0 <= s["win_rate_pct"] <= 100
    assert 0 <= s["max_drawdown_pct"] < 100
