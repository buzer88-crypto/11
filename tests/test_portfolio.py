import numpy as np
import pandas as pd
import pytest

from pinetest.portfolio import (backtest, buy_and_hold, circular_shift, equal_weight,
                                permute_assets, rebalance_on)


def frame(data, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(next(iter(data.values()))), freq="D")
    return pd.DataFrame(data, index=idx, dtype=float)


def test_signal_earns_only_returns_after_the_trade():
    prices = frame({"a": [100, 110, 121, 133.1, 146.41]})        # +10% every day
    w = frame({"a": [0, 1, 1, 1, 1]})                              # decided at close of day 1
    r0 = backtest(w, prices, cost_bps=0, lag=0).returns
    assert r0.tolist() == pytest.approx([0, 0, 0.1, 0.1, 0.1])     # traded at close 1, earns day 2+
    r1 = backtest(w, prices, cost_bps=0, lag=1).returns
    assert r1.tolist() == pytest.approx([0, 0, 0, 0.1, 0.1])       # one day later


def test_perfect_foresight_is_impossible_with_same_day_weights():
    rng = np.random.default_rng(0)
    prices = frame({"a": 100 * np.cumprod(1 + rng.normal(0, 0.05, 400))})
    today_up = (prices["a"].pct_change() > 0).astype(float).to_frame("a")
    res = backtest(today_up, prices, cost_bps=0, lag=0)
    # Knowing today's move only helps if the harness leaked it into today's return.
    assert abs(res.metrics()["sharpe"]) < 1.5


def test_costs_charged_on_turnover():
    prices = frame({"a": [100] * 5})
    w = frame({"a": [1, 1, 0, 0, 0]})
    res = backtest(w, prices, cost_bps=10, lag=0)
    assert res.turnover.tolist() == pytest.approx([1, 0, 1, 0, 0])
    assert res.returns.tolist() == pytest.approx([-0.001, 0, -0.001, 0, 0])


def test_drift_is_not_traded_without_rebalance_rows():
    prices = frame({"a": [100, 200, 200, 200], "b": [100, 100, 100, 100]})
    w = frame({"a": [0.5, np.nan, np.nan, 0.5], "b": [0.5, np.nan, np.nan, 0.5]})
    res = backtest(w, prices, cost_bps=0, lag=0)
    held = res.held
    assert held.iloc[2].tolist() == pytest.approx([2 / 3, 1 / 3])     # drifted, not traded
    assert res.turnover.iloc[3] == pytest.approx(1 / 3)               # back to 50/50 on day 3


def test_short_profits_and_pays_fee():
    prices = frame({"a": [100, 100, 90]})
    w = frame({"a": [-1, -1, -1]})
    res = backtest(w, prices, cost_bps=0, lag=0, short_fee_annual=0.365)
    assert res.returns.iloc[1] == pytest.approx(-0.001)
    assert res.returns.iloc[2] == pytest.approx(0.1 - 0.001)


def test_no_price_no_position_and_dead_asset_is_frozen():
    prices = frame({"a": [np.nan, 100, 110, np.nan, np.nan], "b": [100] * 5})
    w = frame({"a": [1, 1, 1, 1, 1], "b": [0] * 5})
    res = backtest(w, prices, cost_bps=0, lag=0)
    assert res.held["a"].iloc[1] == 0          # day 0 target ignored: no price on trade date
    assert res.returns.iloc[2] == pytest.approx(0.1)
    assert res.returns.iloc[3] == 0            # price gone: valued at last price
    assert res.held["a"].iloc[4] == 0          # sold at next rebalance


def test_blowup_stops_at_minus_100():
    prices = frame({"a": [100, 100, 250, 300]})
    res = backtest(frame({"a": [-1, -1, -1, -1]}), prices, cost_bps=0, lag=0, short_fee_annual=0)
    assert res.returns.iloc[2] == -1.0
    assert res.equity().iloc[-1] == 0


def test_rebalance_on_keeps_period_ends():
    w = frame({"a": np.arange(14)}, start="2024-01-01")          # Monday
    r = rebalance_on(w, "W-SUN")
    assert r["a"].dropna().index.strftime("%a").tolist() == ["Sun", "Sun"]


def test_benchmarks():
    prices = frame({"btc": [1, 2, 4], "eth": [1, 1, 1]})
    assert backtest(buy_and_hold(prices), prices, cost_bps=0, lag=0).equity().iloc[-1] == 4
    members = frame({"btc": [1, 1, 1], "eth": [1, 1, 1]}).astype(bool)
    ew = equal_weight(members, "D")
    assert ew.iloc[0].tolist() == [0.5, 0.5]


def test_placebos_preserve_shape():
    rng = np.random.default_rng(1)
    n = 400
    prices = frame({"a": np.linspace(1, 2, n), "b": np.linspace(2, 1, n), "c": np.ones(n)})
    w = rebalance_on(frame({"a": (np.arange(n) % 50 < 25).astype(float),
                            "b": np.zeros(n), "c": np.zeros(n)}), "W-SUN")
    shifted = circular_shift(w, prices, rng)
    assert shifted.notna().any(axis=1).equals(w.notna().any(axis=1))
    assert shifted["a"].sum() == pytest.approx(w["a"].sum(), rel=0.1)
    members = pd.DataFrame(True, index=w.index, columns=w.columns)
    perm = permute_assets(w, members, rng)
    rows = w.notna().any(axis=1)
    assert (perm[rows].sum(axis=1) == w[rows].sum(axis=1)).all()
    assert (perm[rows] != 0).sum(axis=1).equals((w[rows] != 0).sum(axis=1))
