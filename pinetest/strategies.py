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


STRATEGIES = {"ema_cross": ema_cross, "rsi_reversion": rsi_reversion}
