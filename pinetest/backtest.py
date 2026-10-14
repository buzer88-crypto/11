"""A small long-only backtester that mimics TradingView's broker emulator defaults.

- Signals are computed on the close of bar i; market orders fill at the open of bar i+1.
- Stop/limit exits become active on the bar after the entry fill (as in the Pine scripts,
  which only place strategy.exit once strategy.position_size > 0).
- When stop and limit could both trigger inside one bar, the intrabar path follows
  TradingView's assumption: if the open is closer to the high, open -> high -> low -> close,
  otherwise open -> low -> high -> close.
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
    meta: dict = field(default_factory=dict)


def run(df: pd.DataFrame, signals: Signals, initial_capital: float = 10_000.0,
        commission_pct: float = 0.1) -> Result:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    entry = signals.entry.fillna(False).to_numpy(dtype=bool)
    exit_ = signals.exit.fillna(False).to_numpy(dtype=bool)
    comm = commission_pct / 100

    cash = initial_capital
    qty = 0.0
    entry_price = 0.0
    entry_bar = -1
    entry_cost = 0.0
    trades: list[Trade] = []
    equity = np.empty(len(c))

    def close_position(bar: int, price: float, reason: str) -> None:
        nonlocal cash, qty
        proceeds = qty * price * (1 - comm)
        trades.append(Trade(entry_bar, entry_price, bar, price, qty, reason, proceeds - entry_cost))
        cash += proceeds
        qty = 0.0

    for i in range(len(c)):
        if i > 0:
            # Market orders from the previous bar's signals fill at this bar's open.
            if qty > 0 and exit_[i - 1]:
                close_position(i, o[i], "signal")
            elif qty == 0 and entry[i - 1]:
                entry_price = o[i]
                qty = cash / (entry_price * (1 + comm))
                entry_cost = qty * entry_price * (1 + comm)
                cash -= entry_cost
                entry_bar = i

        if qty > 0 and i > entry_bar:
            hit = _protective_fill(o[i], h[i], l[i], entry_price, signals.stop_pct, signals.limit_pct)
            if hit:
                close_position(i, *hit)

        equity[i] = cash + qty * c[i]

    if qty > 0:
        close_position(len(c) - 1, c[-1], "end")
    equity[-1] = cash
    return Result(trades, pd.Series(equity, index=df.index), initial_capital)


def _protective_fill(o, h, l, entry_price, stop_pct, limit_pct):
    stop = entry_price * (1 - stop_pct / 100) if stop_pct else None
    limit = entry_price * (1 + limit_pct / 100) if limit_pct else None
    # Gaps through a level fill at the open.
    if stop is not None and o <= stop:
        return o, "stop"
    if limit is not None and o >= limit:
        return o, "limit"
    stop_hit = stop is not None and l <= stop
    limit_hit = limit is not None and h >= limit
    if stop_hit and limit_hit:
        high_first = (h - o) <= (o - l)
        return (limit, "limit") if high_first else (stop, "stop")
    if stop_hit:
        return stop, "stop"
    if limit_hit:
        return limit, "limit"
    return None
