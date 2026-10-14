"""Signal generators mirroring pine/*.pine. Keep defaults in sync with the Pine inputs."""
import pandas as pd

from .backtest import Signals
from .indicators import crossover, crossunder, ema, rsi


def ema_cross(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21) -> Signals:
    fast = ema(df["close"], fast_len)
    slow = ema(df["close"], slow_len)
    return Signals(entry=crossover(fast, slow), exit=crossunder(fast, slow),
                   meta={"fast": fast, "slow": slow})


def rsi_reversion(df: pd.DataFrame, rsi_len: int = 14, oversold: float = 30,
                  overbought: float = 70, sl_pct: float = 5, tp_pct: float = 10) -> Signals:
    r = rsi(df["close"], rsi_len)
    return Signals(entry=crossover(r, oversold), exit=crossover(r, overbought),
                   stop_pct=sl_pct, limit_pct=tp_pct, meta={"rsi": r})


def ema_breakeven(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21, sl_pct: float = 3,
                  tp_pct: float = 9, be_pct: float | None = 3) -> Signals:
    """Long on a bullish EMA cross, short on a bearish one; exits only via SL, TP or breakeven."""
    fast = ema(df["close"], fast_len)
    slow = ema(df["close"], slow_len)
    never = pd.Series(False, index=df.index)
    return Signals(entry=crossover(fast, slow), exit=never, short=crossunder(fast, slow),
                   stop_pct=sl_pct, limit_pct=tp_pct, be_pct=be_pct,
                   meta={"fast": fast, "slow": slow})


STRATEGIES = {"ema_cross": ema_cross, "rsi_reversion": rsi_reversion,
              "ema_breakeven": ema_breakeven}
