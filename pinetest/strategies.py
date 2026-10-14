"""Signal generators mirroring pine/*.pine. Keep defaults in sync with the Pine inputs."""
import numpy as np
import pandas as pd

from .backtest import Signals
from .indicators import crossover, crossunder, ema, ema_cross_price, rsi, sma, volume_delta


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


def ema_anticipate(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21, sl_pct: float = 3,
                   tp_pct: float = 9, be_pct: float | None = 3) -> Signals:
    """Like ema_breakeven, but enters intrabar with a stop order at the exact price where the
    EMAs would cross, instead of waiting for the close and entering on the next open.
    If price touches the level and then falls back, the cross never confirms (a false start)."""
    fast = ema(df["close"], fast_len)
    slow = ema(df["close"], slow_len)
    level = ema_cross_price(fast, slow, fast_len, slow_len)
    never = pd.Series(False, index=df.index)
    return Signals(entry=never, exit=never, long_stop=level.where(fast < slow),
                   short_stop=level.where(fast > slow), stop_pct=sl_pct, limit_pct=tp_pct,
                   be_pct=be_pct, meta={"fast": fast, "slow": slow, "level": level})


ENTRY_MODES = ("confirmed", "anticipate", "predict", "fade")
EXIT_MODES = ("none", "anticipate", "predict", "reverse")


def ema_lab(df: pd.DataFrame, fast_len: int = 9, slow_len: int = 21, sl_pct: float = 3,
            tp_pct: float = 9, be_pct: float = 3, entry_mode: str = "confirmed",
            exit_mode: str = "none", predict_bars: int = 2, max_bars: int = 0) -> Signals:
    """EMA cross experiments; mirrors pine/ema_lab.pine. A value of 0 disables SL/TP/BE/max_bars.

    entry_mode:
      confirmed   market entry on the open after a confirmed cross
      anticipate  stop order at the exact price that would make the EMAs cross on the next bar
      predict     market entry when a linear extrapolation of the EMA spread puts the cross
                  within `predict_bars` bars (the first bar this becomes true)
      fade        trade against a failed cross: price touched the cross price but the bar
                  closed without crossing
    exit_mode (on top of SL/TP/BE):
      anticipate  stop at the price of the opposite cross; market exit if it already crossed
      predict     market exit when the opposite cross is predicted within `predict_bars` bars
      reverse     like anticipate, but flip into the opposite position at the cross price
    """
    if entry_mode not in ENTRY_MODES or exit_mode not in EXIT_MODES:
        raise ValueError(f"unknown mode {entry_mode!r}/{exit_mode!r}")
    fast = ema(df["close"], fast_len)
    slow = ema(df["close"], slow_len)
    level = ema_cross_price(fast, slow, fast_len, slow_len)
    spread = fast - slow
    vel = spread.diff()
    bars_to_cross = (-spread / vel).where((vel != 0) & (np.sign(vel) != np.sign(spread)))
    pred_up = (spread < 0) & (bars_to_cross <= predict_bars)
    pred_down = (spread > 0) & (bars_to_cross <= predict_bars)
    never = pd.Series(False, index=df.index)

    sig = Signals(entry=never, exit=never, stop_pct=sl_pct or None, limit_pct=tp_pct or None,
                  be_pct=be_pct or None, max_bars=max_bars or None,
                  meta={"fast": fast, "slow": slow, "level": level})
    if entry_mode == "confirmed":
        sig.entry, sig.short = crossover(fast, slow), crossunder(fast, slow)
    elif entry_mode == "anticipate":
        sig.long_stop, sig.short_stop = level.where(spread < 0), level.where(spread > 0)
    elif entry_mode == "predict":
        sig.entry = pred_up & ~pred_up.shift(1, fill_value=False)
        sig.short = pred_down & ~pred_down.shift(1, fill_value=False)
    else:  # fade
        prev_level, prev_spread = level.shift(1), spread.shift(1)
        sig.short = (prev_spread < 0) & (df["high"] >= prev_level) & (spread < 0)
        sig.entry = (prev_spread > 0) & (df["low"] <= prev_level) & (spread > 0)

    if exit_mode in ("anticipate", "reverse"):
        sig.long_exit_stop, sig.short_exit_stop = level.where(spread > 0), level.where(spread < 0)
        sig.reverse = exit_mode == "reverse"
    if exit_mode == "anticipate":
        sig.long_exit, sig.short_exit = spread < 0, spread > 0
    elif exit_mode == "predict":
        sig.long_exit, sig.short_exit = pred_down | (spread < 0), pred_up | (spread > 0)
    return sig


STRATEGIES = {"ema_cross": ema_cross, "rsi_reversion": rsi_reversion,
              "ema_breakeven": ema_breakeven, "ema_anticipate": ema_anticipate,
              "ema_lab": ema_lab}
