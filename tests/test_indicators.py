import numpy as np
import pandas as pd
import pytest

from pinetest.indicators import crossover, crossunder, ema, rma, rsi, sma, volume_delta


def test_ema_is_sma_seeded_then_recursive():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = ema(s, 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)          # SMA(1,2,3)
    assert out.iloc[3] == pytest.approx(0.5 * 4 + 0.5 * 2.0)
    assert out.iloc[4] == pytest.approx(0.5 * 5 + 0.5 * 3.0)


def test_rma_uses_wilder_alpha():
    s = pd.Series([2.0, 4, 6, 8])
    out = rma(s, 2)
    assert out.iloc[1] == pytest.approx(3.0)
    assert out.iloc[2] == pytest.approx(0.5 * 6 + 0.5 * 3.0)


def test_rsi_extremes_and_range(random_walk):
    up = pd.Series(np.arange(1.0, 31))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(up[::-1].reset_index(drop=True), 14).iloc[-1] == pytest.approx(0.0)
    r = rsi(random_walk["close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert rsi(up, 14).iloc[:14].isna().all()  # warm-up: first change is na, then 14 bars


def test_rsi_matches_hand_computation():
    closes = pd.Series([10.0, 11, 10, 12, 11])
    # changes: +1, -1, +2, -1 -> length 2: up seed=(1+0)/2, down seed=(0+1)/2
    up, down = 0.5, 0.5
    up, down = 0.5 * 2 + 0.5 * up, 0.5 * 0 + 0.5 * down   # bar 3
    up, down = 0.5 * 0 + 0.5 * up, 0.5 * 1 + 0.5 * down   # bar 4
    assert rsi(closes, 2).iloc[-1] == pytest.approx(100 - 100 / (1 + up / down))


def test_crossover_and_crossunder():
    a = pd.Series([1.0, 2, 3, 2, 1])
    b = pd.Series([2.0, 2, 2, 2, 2])
    assert crossover(a, b).tolist() == [False, False, True, False, False]
    assert crossunder(a, b).tolist() == [False, False, False, False, True]
    assert crossover(a, 2.5).tolist() == [False, False, True, False, False]


def test_sma_includes_current_bar():
    out = sma(pd.Series([1, 2, 3, 4]), 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [2.0, 3.0]


def test_volume_delta_signs_by_candle_then_previous_close():
    df = pd.DataFrame({"open":   [10.0, 10, 11, 11, 11],
                       "close":  [10.0, 11, 10, 11, 11],
                       "volume": [5.0, 1, 2, 3, 4]})
    # bar0 doji, no previous close -> 0; bar3 doji above prev close -> +; bar4 flat -> 0
    assert volume_delta(df).tolist() == [0, 1, -2, 3, 0]
