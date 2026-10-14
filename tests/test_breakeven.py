import pandas as pd
import pytest

from pinetest.backtest import Signals, run
from pinetest.strategies import ema_breakeven

from .conftest import make_ohlc


def flags(n, idx):
    return pd.Series([i in idx for i in range(n)])


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)


def test_short_profits_when_price_falls():
    df = make_ohlc([100, 100, 90, 80])
    res = run(df, Signals(entry=flags(4, set()), exit=flags(4, {2}), short=flags(4, {0})),
              commission_pct=0.1)
    (t,) = res.trades
    assert (t.side, t.entry_price, t.exit_price) == (-1, 100, 90)
    qty = 10_000 / (100 * 1.001)
    assert t.pnl == pytest.approx(qty * 10 - qty * 190 * 0.001)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_short_stop_and_limit_are_mirrored():
    df = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 104, 99, 103]])
    res = run(df, Signals(entry=flags(3, set()), exit=flags(3, set()), short=flags(3, {0}),
                          stop_pct=3, limit_pct=9), commission_pct=0)
    assert (res.trades[0].exit_price, res.trades[0].reason) == (pytest.approx(103), "stop")

    df = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 101, 90, 92]])
    res = run(df, Signals(entry=flags(3, set()), exit=flags(3, set()), short=flags(3, {0}),
                          stop_pct=3, limit_pct=9), commission_pct=0)
    assert (res.trades[0].exit_price, res.trades[0].reason) == (pytest.approx(91), "limit")


def be_signals(n, short=False, be=3):
    entry = flags(n, set() if short else {0})
    return Signals(entry=entry, exit=flags(n, set()), short=flags(n, {0}) if short else None,
                   stop_pct=3, limit_pct=9, be_pct=be)


def test_breakeven_moves_stop_to_entry_from_next_bar():
    df = bars([
        [100, 100, 100, 100],
        [100, 100, 100, 100],   # entry at 100
        [100, 104, 99, 103],    # +4% touches the 3% trigger; old stop (97) still in force
        [103, 103, 99, 99],     # falls below entry -> breakeven stop at 100
    ])
    (t,) = run(df, be_signals(4), commission_pct=0).trades
    assert (t.exit_bar, t.exit_price, t.reason) == (3, 100, "be")

    (t,) = run(df, be_signals(4, be=None), commission_pct=0).trades
    assert t.reason == "end"        # without breakeven the trade is still open at 99


def test_breakeven_trade_still_pays_commission():
    df = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 104, 100, 103], [103, 103, 99, 99]])
    (t,) = run(df, be_signals(4), commission_pct=0.1).trades
    assert t.reason == "be" and t.pnl < 0
    assert t.pnl == pytest.approx(-(10_000 / 100.1) * 200 * 0.001)


def test_breakeven_gap_fills_at_open_below_entry():
    df = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 104, 100, 103], [95, 96, 94, 95]])
    (t,) = run(df, be_signals(4), commission_pct=0).trades
    assert (t.exit_price, t.reason) == (95, "be")


def test_short_breakeven():
    df = bars([[100, 100, 100, 100], [100, 100, 100, 100], [100, 101, 96, 97], [97, 102, 97, 101]])
    (t,) = run(df, be_signals(4, short=True), commission_pct=0).trades
    assert (t.side, t.exit_price, t.reason) == (-1, 100, "be")


def test_ema_breakeven_trades_both_directions(random_walk):
    res = run(random_walk, ema_breakeven(random_walk))
    assert {t.side for t in res.trades} == {1, -1}
    assert {t.reason for t in res.trades} <= {"stop", "limit", "be", "end"}
    for a, b in zip(res.trades, res.trades[1:]):
        assert b.entry_bar > a.exit_bar  # never more than one position at a time
