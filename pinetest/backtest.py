"""A small long/short backtester that mimics TradingView's broker emulator defaults.

- Signals are computed on the close of bar i; market orders fill at the open of bar i+1.
- Stop/limit exits become active on the bar after the entry fill (as in the Pine scripts,
  which only place strategy.exit once strategy.position_size > 0).
- When stop and limit could both trigger inside one bar, the intrabar path follows
  TradingView's assumption: if the open is closer to the high, open -> high -> low -> close,
  otherwise open -> low -> high -> close.
- Breakeven: once price moves `be_pct` in favour (checked on bar close), the stop moves
  to the entry price from the next bar on.
- Position size is 100% of equity; commission is a percentage of each fill's value.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    qty: float
    reason: str
    pnl: float
    side: int = 1


@dataclass
class Result:
    trades: list[Trade]
    equity: pd.Series
    initial_capital: float

    @property
    def net_profit(self) -> float:
        return float(self.equity.iloc[-1] - self.initial_capital)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl > 0 for t in self.trades) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        peak = self.equity.cummax()
        return float(((peak - self.equity) / peak).max())

    def summary(self) -> dict:
        return {
            "trades": len(self.trades),
            "net_profit": round(self.net_profit, 2),
            "net_profit_pct": round(100 * self.net_profit / self.initial_capital, 2),
            "win_rate_pct": round(100 * self.win_rate, 2),
            "max_drawdown_pct": round(100 * self.max_drawdown, 2),
        }


@dataclass
class Signals:
    entry: pd.Series
    exit: pd.Series
    stop_pct: float | None = None
    limit_pct: float | None = None
    short: pd.Series | None = None
    be_pct: float | None = None
    meta: dict = field(default_factory=dict)


def _flags(series: pd.Series | None, n: int) -> np.ndarray:
    if series is None:
        return np.zeros(n, dtype=bool)
    return series.fillna(False).to_numpy(dtype=bool)


def run(df: pd.DataFrame, signals: Signals, initial_capital: float = 10_000.0,
        commission_pct: float = 0.1) -> Result:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    long_entry = _flags(signals.entry, len(c))
    short_entry = _flags(signals.short, len(c))
    exit_ = _flags(signals.exit, len(c))
    comm = commission_pct / 100

    cash = initial_capital
    side = 0  # +1 long, -1 short, 0 flat
    qty = 0.0
    entry_price = 0.0
    entry_bar = -1
    be_active = False
    trades: list[Trade] = []
    equity = np.empty(len(c))

    def open_position(bar: int, price: float, direction: int) -> None:
        nonlocal cash, side, qty, entry_price, entry_bar, be_active
        side, entry_price, entry_bar, be_active = direction, price, bar, False
        qty = cash / (price * (1 + comm))
        cash -= qty * price * comm

    def close_position(bar: int, price: float, reason: str) -> None:
        nonlocal cash, side, qty
        gross = side * qty * (price - entry_price)
        trades.append(Trade(entry_bar, entry_price, bar, price, qty, reason,
                            gross - qty * (entry_price + price) * comm, side))
        cash += gross - qty * price * comm
        side, qty = 0, 0.0

    for i in range(len(c)):
        if i > 0:
            # Market orders from the previous bar's signals fill at this bar's open.
            if side and exit_[i - 1]:
                close_position(i, o[i], "signal")
            elif not side and long_entry[i - 1]:
                open_position(i, o[i], 1)
            elif not side and short_entry[i - 1]:
                open_position(i, o[i], -1)

        if side and i > entry_bar:
            hit = _protective_fill(side, o[i], h[i], l[i], entry_price, signals.stop_pct,
                                   signals.limit_pct, be_active)
            if hit:
                close_position(i, *hit)

        # Breakeven is armed on the bar's close, so the moved stop applies from the next bar
        # (the Pine script re-places strategy.exit once per bar).
        if side and signals.be_pct:
            trigger = entry_price * (1 + side * signals.be_pct / 100)
            if (h[i] >= trigger) if side > 0 else (l[i] <= trigger):
                be_active = True

        equity[i] = cash + side * qty * (c[i] - entry_price)

    if side:
        close_position(len(c) - 1, c[-1], "end")
    equity[-1] = cash
    return Result(trades, pd.Series(equity, index=df.index), initial_capital)


def _protective_fill(side, o, h, l, entry_price, stop_pct, limit_pct, be_active=False):
    if be_active:
        stop = entry_price
    else:
        stop = entry_price * (1 - side * stop_pct / 100) if stop_pct else None
    limit = entry_price * (1 + side * limit_pct / 100) if limit_pct else None
    if side < 0:
        # Mirror prices so that "favourable" is always up; mirror the fill back at the end.
        o, h, l = -o, -l, -h
        stop = -stop if stop is not None else None
        limit = -limit if limit is not None else None
    hit = _long_fill(o, h, l, stop, limit, "be" if be_active else "stop")
    if hit and side < 0:
        hit = (-hit[0], hit[1])
    return hit


def _long_fill(o, h, l, stop, limit, reason_stop):
    # Gaps through a level fill at the open.
    if stop is not None and o <= stop:
        return o, reason_stop
    if limit is not None and o >= limit:
        return o, "limit"
    stop_hit = stop is not None and l <= stop
    limit_hit = limit is not None and h >= limit
    if stop_hit and limit_hit:
        high_first = (h - o) <= (o - l)
        return (limit, "limit") if high_first else (stop, reason_stop)
    if stop_hit:
        return stop, reason_stop
    if limit_hit:
        return limit, "limit"
    return None
