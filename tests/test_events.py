import numpy as np
import pandas as pd
import pytest

from pinetest.events import Setup, simulate, summary
from pinetest.smc import fair_value_gaps, liquidity_sweeps, order_blocks


def bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)


FLAT = [[100, 100, 100, 100]] * 2


def test_market_entry_hits_target():
    df = bars(FLAT + [[100, 103, 99.5, 102], [102, 102, 102, 102]])
    (t,) = simulate(df, [Setup(1, 1, "market", stop=99, target=102)])
    assert (t.fill_bar, t.entry, t.exit, t.reason, t.r) == (2, 100, 102, "target", 2.0)


def test_stop_wins_when_both_levels_in_one_bar():
    df = bars(FLAT + [[100, 103, 98, 100]])
    (t,) = simulate(df, [Setup(1, 1, "market", stop=99, target=102)])
    assert (t.reason, t.r) == ("stop", -1.0)


def test_gap_through_stop_fills_at_open():
    df = bars(FLAT + [[100, 100.5, 99.5, 100], [97, 98, 96, 97]])
    (t,) = simulate(df, [Setup(1, 1, "market", stop=99, target=103)])
    assert (t.exit, t.reason, t.r) == (97, "stop", -3.0)


def test_limit_fill_cancel_and_expiry():
    df = bars(FLAT + [[100, 100, 99, 99.5], [99.5, 101, 99.5, 101], [101, 102, 101, 102]])
    (t,) = simulate(df, [Setup(1, 1, "limit", stop=98, target=102, price=99)])
    assert (t.fill_bar, t.entry, t.reason, t.r) == (2, 99, "target", pytest.approx(3.0))   # risk 1, gain 3
    # Target trades before the limit is reached: cancelled.
    df2 = bars(FLAT + [[100, 102.5, 99.5, 102], [102, 102, 98, 98]])
    assert simulate(df2, [Setup(1, 1, "limit", stop=98, target=102, price=99)]) == []
    # Never reached within `valid` bars: expired.
    df3 = bars(FLAT + [[100, 100.5, 99.5, 100]] * 3 + [[100, 100, 98.5, 99]])
    assert simulate(df3, [Setup(1, 1, "limit", stop=98, target=102, price=99, valid=3)]) == []


def test_limit_fill_bar_that_reaches_stop_is_a_loss():
    df = bars(FLAT + [[100, 101, 97, 100], [100, 103, 100, 103]])
    (t,) = simulate(df, [Setup(1, 1, "limit", stop=98, target=102, price=99)])
    assert (t.fill_bar, t.reason, t.r) == (2, "stop", -1.0)


def test_short_time_exit_and_costs():
    df = bars(FLAT + [[100, 100.5, 99.5, 99.5]] * 3)
    (t,) = simulate(df, [Setup(1, -1, "market", stop=101, target=97, max_hold=2)], cost_pct=0.1)
    assert (t.exit_bar, t.exit, t.reason) == (3, 99.5, "time")
    assert t.r == pytest.approx((0.5 - 0.001 * 199.5) / 1.0)


def test_overlapping_setups_are_skipped():
    df = bars(FLAT + [[100, 100.5, 99.5, 100]] * 4 + [[100, 103, 100, 103]])
    trades = simulate(df, [Setup(1, 1, "market", 99, 102), Setup(2, 1, "market", 99, 102),
                           Setup(6, 1, "market", 99, 102)])
    assert [t.bar for t in trades] == [1]
    assert summary(trades)["n"] == 1


def trending(n=200, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.3, "low": np.minimum(o, c) - 0.3, "close": c})


def test_sweep_detector_finds_rejections_only():
    df = trending()
    for s in liquidity_sweeps(df, lookback=10):
        i = s.bar
        prior_low = df["low"].iloc[i - 10:i].min()
        prior_high = df["high"].iloc[i - 10:i].max()
        if s.side > 0:
            assert df["low"].iloc[i] < prior_low < df["close"].iloc[i] and s.stop < df["low"].iloc[i]
        else:
            assert df["high"].iloc[i] > prior_high > df["close"].iloc[i] and s.stop > df["high"].iloc[i]


def test_fvg_detector_uses_the_gap():
    df = bars([[100, 101, 99, 100]] * 14 + [[100, 101, 99.5, 100.5], [100.5, 104, 100.5, 104], [104, 105, 102, 104.5]])
    (s,) = [s for s in fair_value_gaps(df, min_gap_atr=0.1) if s.bar == 16]
    assert s.side == 1 and s.price == pytest.approx((102 + 101) / 2) and s.stop < 101


def test_order_block_is_last_opposite_candle_before_break():
    rows = [[100, 101, 99, 100]] * 14 + [[100, 100.5, 98, 98.5], [98.5, 103, 98.5, 102.5]]
    df = bars(rows)
    (s,) = [s for s in order_blocks(df, lookback=10) if s.bar == 15]
    assert s.side == 1 and s.price == 100.5 and s.stop < 98


def test_pivots_need_right_bars_and_strict_extreme():
    from pinetest.divergence import pivots
    x = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 1, 2], dtype=float)
    lows = pivots(x, 2, 2, low=True)
    assert np.flatnonzero(lows).tolist() == [4]         # bar 10 lacks 2 right bars
    assert not pivots(np.array([2, 1, 1, 2.0]), 1, 1, low=True)[1]      # ties are not pivots


def test_regular_bullish_divergence_signals_after_confirmation():
    from pinetest.divergence import rsi_divergences
    # Fall hard to a first low, bounce, then grind to a slightly lower low on weaker momentum.
    steps = [0.5] * 20 + [-3, 1] * 8 + [2.5] * 3 + [-1.4, 0.6] * 13 + [-0.4] + [2] * 12
    c = 100 + np.cumsum(steps)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.2, "low": np.minimum(o, c) - 0.2, "close": c})
    setups = rsi_divergences(df, kind="regular", left=3, right=3, sides=(1,))
    assert setups, "expected a bullish divergence"
    s = setups[-1]
    p = s.bar - 3                                        # the pivot is 3 bars before the signal
    assert df["low"].iloc[p] <= df["low"].iloc[:p].min() + 1e-9 and s.stop < df["low"].iloc[p]
    # No lookahead: the same setups appear when later bars are cut off.
    cut = rsi_divergences(df.iloc[: s.bar + 2], kind="regular", left=3, right=3, sides=(1,))
    assert [x.bar for x in cut] == [x.bar for x in setups if x.bar <= s.bar]


def test_macd_matches_its_definition():
    from pinetest.indicators import ema, macd
    rng = np.random.default_rng(1)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)))
    line, sig, hist = macd(c)
    assert np.allclose(line.dropna(), (ema(c, 12) - ema(c, 26)).dropna())
    assert line.first_valid_index() == 25 and sig.first_valid_index() == 25 + 8
    assert np.allclose(hist.dropna(), (line - sig).dropna())


def test_confirmations_filter_or_delay_the_base_setups():
    from pinetest.divergence import CONFIRM, rsi_divergences
    rng = np.random.default_rng(7)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1500)))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c})
    for kind in ("regular", "hidden"):
        base = {(s.bar, s.side, s.stop) for s in rsi_divergences(df, kind=kind)}
        assert base
        for rule in CONFIRM[1:-1]:
            sub = {(s.bar, s.side, s.stop) for s in rsi_divergences(df, kind=kind, confirm=rule)}
            assert sub <= base, rule
        delayed = rsi_divergences(df, kind=kind, confirm="ema20_reclaim", wait=6)
        stops = {(side, stop): bar for bar, side, stop in base}
        for s in delayed:
            b = stops[(s.side, s.stop)]
            assert b <= s.bar <= b + 6
    with pytest.raises(ValueError):
        rsi_divergences(df, confirm="magic")
