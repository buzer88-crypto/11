import numpy as np
import pandas as pd
import pytest

from pinetest.backtest import Signals, run
from pinetest.indicators import ema, ema_cross_price
from pinetest.strategies import ema_anticipate


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)


def test_cross_price_makes_emas_equal_on_next_bar(random_walk):
    close = random_walk["close"]
    level = ema_cross_price(ema(close, 9), ema(close, 21), 9, 21)
    i = 100
    nxt = close.iloc[: i + 1].tolist() + [level.iloc[i]]
    s = pd.Series(nxt)
    assert ema(s, 9).iloc[-1] == pytest.approx(ema(s, 21).iloc[-1])


def test_cross_price_rejects_equal_lengths():
    with pytest.raises(ValueError):
        ema_cross_price(pd.Series([1.0]), pd.Series([1.0]), 5, 5)


def stop_signals(n, long_level=None, short_level=None, at=0):
    nan = pd.Series(np.nan, index=range(n))
    never = pd.Series(False, index=range(n))
    ls, ss = nan.copy(), nan.copy()
    if long_level is not None:
        ls[at] = long_level
    if short_level is not None:
        ss[at] = short_level
    return Signals(entry=never, exit=never, long_stop=ls, short_stop=ss)


def test_stop_entry_fills_at_level_when_touched():
    df = bars([[100, 100, 100, 100], [100, 103, 99, 102], [102, 102, 102, 102]])
    (t,) = run(df, stop_signals(3, long_level=101.5), commission_pct=0).trades
    assert (t.entry_bar, t.entry_price, t.side) == (1, 101.5, 1)


def test_stop_entry_gap_fills_at_open():
    df = bars([[100, 100, 100, 100], [104, 105, 103, 104], [104, 104, 104, 104]])
    (t,) = run(df, stop_signals(3, long_level=101.5), commission_pct=0).trades
    assert t.entry_price == 104


def test_stop_entry_not_touched_and_expires():
    df = bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 110, 100, 110]])
    assert run(df, stop_signals(3, long_level=101.5), commission_pct=0).trades == []


def test_short_stop_entry():
    df = bars([[100, 100, 100, 100], [100, 100, 97, 98], [98, 98, 98, 98]])
    (t,) = run(df, stop_signals(3, short_level=98.5), commission_pct=0).trades
    assert (t.entry_price, t.side) == (98.5, -1)


def test_anticipate_enters_no_later_than_confirmed_cross(random_walk):
    from pinetest.strategies import ema_breakeven
    early = run(random_walk, ema_anticipate(random_walk)).trades
    late = run(random_walk, ema_breakeven(random_walk)).trades
    assert early and late
    # The first anticipated trade starts no later than the first confirmed-cross trade.
    assert early[0].entry_bar <= late[0].entry_bar
    sig = ema_anticipate(random_walk)
    both = sig.long_stop.notna() & sig.short_stop.notna()
    assert not both.any()
