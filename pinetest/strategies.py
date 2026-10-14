"""Signal generators mirroring pine/*.pine. Keep defaults in sync with the Pine inputs."""
import pandas as pd

from .backtest import Signals
from .indicators import crossover, crossunder, ema, rsi, sma, volume_delta


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
                  tp_pct: float = 9, be_pct: float | None = 3, vol_filter: bool = False,
                  vol_len: int = 20, vol_mult: float = 1.5, cvd_filter: bool = False,
                  cvd_len: int = 20) -> Signals:
    """Long on a bullish EMA cross, short on a bearish one; exits only via SL, TP or breakeven.

    With `vol_filter`, a cross only counts if the signal bar's volume exceeds
    `vol_mult` times the SMA of volume over `vol_len` bars.
    With `cvd_filter`, the net volume delta over the last `cvd_len` bars (the change in CVD)
    must agree with the direction: positive for longs, negative for shorts.
    """
    fast = ema(df["close"], fast_len)
    slow = ema(df["close"], slow_len)
    long_, short = crossover(fast, slow), crossunder(fast, slow)
    vol_ok = volume_ok(df, vol_len, vol_mult) if vol_filter else True
    if cvd_filter:
        cvd_net = volume_delta(df).rolling(cvd_len).sum()
        long_, short = long_ & (cvd_net > 0), short & (cvd_net < 0)
    never = pd.Series(False, index=df.index)
    return Signals(entry=long_ & vol_ok, exit=never, short=short & vol_ok,
                   stop_pct=sl_pct, limit_pct=tp_pct, be_pct=be_pct,
                   meta={"fast": fast, "slow": slow})


def volume_ok(df: pd.DataFrame, length: int, mult: float) -> pd.Series:
    return df["volume"] > mult * sma(df["volume"], length)


STRATEGIES = {"ema_cross": ema_cross, "rsi_reversion": rsi_reversion,
              "ema_breakeven": ema_breakeven}
