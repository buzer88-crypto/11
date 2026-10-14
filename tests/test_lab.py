import numpy as np
import pandas as pd
import pytest

from pinetest.backtest import Signals, run
from pinetest.strategies import ENTRY_MODES, EXIT_MODES, ema_anticipate, ema_breakeven, ema_lab


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)


def series(n, values):
    s = pd.Series(np.nan, index=range(n))
    for i, v in values.items():
        s[i] = v
    return s


def flags(n, idx):
    return pd.Series([i in idx for i in range(n)])


FLAT = [[100, 100, 100, 100]] * 2


def test_exit_level_tightens_stop_and_is_labelled_cross():
    df = bars(FLAT + [[100, 101, 97.5, 98], [98, 98, 98, 98]])
    sig = Signals(entry=flags(4, {0}), exit=flags(4, set()), stop_pct=3,
                  long_exit_stop=series(4, {1: 98.5}))
    (t,) = run(df, sig, commission_pct=0).trades
    assert (t.exit_bar, t.exit_price, t.reason) == (2, 98.5, "cross")


def test_looser_exit_level_does_not_override_stop():
    df = bars(FLAT + [[100, 101, 96, 96], [96, 96, 96, 96]])
    sig = Signals(entry=flags(4, {0}), exit=flags(4, set()), stop_pct=3,
                  long_exit_stop=series(4, {1: 90}))
    (t,) = run(df, sig, commission_pct=0).trades
    assert (t.exit_price, t.reason) == (97, "stop")


def test_reverse_flips_at_cross_level():
    df = bars(FLAT + [[100, 101, 97, 97], [97, 97, 95, 95]])
    sig = Signals(entry=flags(4, {0}), exit=flags(4, set()), reverse=True,
                  long_exit_stop=series(4, {1: 98}))
    first, second = run(df, sig, commission_pct=0).trades
    assert (first.side, first.exit_price, first.reason) == (1, 98, "cross")
    assert (second.side, second.entry_bar, second.entry_price) == (-1, 2, 98)
    assert second.pnl == pytest.approx(second.qty * 3)        # short from 98 to 95


def test_no_reverse_when_stop_loss_is_hit():
    df = bars(FLAT + [[100, 101, 96, 96], [96, 96, 96, 96]])
    sig = Signals(entry=flags(4, {0}), exit=flags(4, set()), reverse=True, stop_pct=3,
                  long_exit_stop=series(4, {1: 90}))
    trades = run(df, sig, commission_pct=0).trades
    assert len(trades) == 1 and trades[0].reason == "stop"


def test_side_specific_market_exit():
    df = bars(FLAT * 3)
    long_exit = flags(6, {2})
    sig = Signals(entry=flags(6, set()), exit=flags(6, set()), short=flags(6, {0}), long_exit=long_exit)
    assert run(df, sig, commission_pct=0).trades[0].reason == "end"     # long exit ignored for a short
    sig.short_exit = long_exit
    (t,) = run(df, sig, commission_pct=0).trades
    assert (t.exit_bar, t.reason) == (3, "signal")


def test_time_stop_closes_after_max_bars():
    df = bars(FLAT * 4)
    sig = Signals(entry=flags(8, {0}), exit=flags(8, set()), max_bars=3)
    (t,) = run(df, sig, commission_pct=0).trades
    assert (t.entry_bar, t.exit_bar, t.reason) == (1, 4, "time")


@pytest.mark.parametrize("entry_mode", ENTRY_MODES)
@pytest.mark.parametrize("exit_mode", EXIT_MODES)
def test_every_mode_combination_runs(random_walk, entry_mode, exit_mode):
    res = run(random_walk, ema_lab(random_walk, entry_mode=entry_mode, exit_mode=exit_mode,
                                   max_bars=10))
    assert np.isfinite(res.equity).all()
    for a, b in zip(res.trades, res.trades[1:]):
        assert b.entry_bar >= a.exit_bar


def test_lab_matches_dedicated_strategies(random_walk):
    a = run(random_walk, ema_lab(random_walk)).trades
    assert a == run(random_walk, ema_breakeven(random_walk)).trades
    b = run(random_walk, ema_lab(random_walk, entry_mode="anticipate")).trades
    assert b == run(random_walk, ema_anticipate(random_walk)).trades


def test_predict_entry_fires_before_confirmed_cross(random_walk):
    sig = ema_lab(random_walk, entry_mode="predict")
    fast, slow = sig.meta["fast"], sig.meta["slow"]
    # Every predicted long fires while the fast EMA is still below the slow one.
    assert sig.entry.any() and (fast[sig.entry] < slow[sig.entry]).all()
    assert (fast[sig.short] > slow[sig.short]).all()
    assert not (sig.entry & sig.entry.shift(1, fill_value=False)).any()   # events, not states


def test_fade_trades_against_failed_cross(random_walk):
    sig = ema_lab(random_walk, entry_mode="fade")
    fast, slow, level = sig.meta["fast"], sig.meta["slow"], sig.meta["level"]
    assert sig.short.any()
    i = sig.short[sig.short].index[0]
    assert fast[i - 1] < slow[i - 1] and fast[i] < slow[i]
    assert random_walk["high"][i] >= level[i - 1]


def test_unknown_mode_rejected(random_walk):
    with pytest.raises(ValueError):
        ema_lab(random_walk, entry_mode="magic")
