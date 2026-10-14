"""Indicators matching Pine Script's ta.* semantics (SMA-seeded EMA/RMA, NaN warm-up)."""
import numpy as np
import pandas as pd


def _seeded_ma(src: pd.Series, length: int, alpha: float) -> pd.Series:
    values = src.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if length < 1 or len(values) < length:
        return pd.Series(out, index=src.index)
    out[length - 1] = values[:length].mean()
    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=src.index)


def sma(src: pd.Series, length: int) -> pd.Series:
    """ta.sma: plain moving average including the current bar, NaN until `length` bars exist."""
    return src.astype(float).rolling(length).mean()


def ema(src: pd.Series, length: int) -> pd.Series:
    """ta.ema: alpha = 2 / (length + 1), seeded with the SMA of the first `length` values."""
    return _seeded_ma(src, length, 2 / (length + 1))


def rma(src: pd.Series, length: int) -> pd.Series:
    """ta.rma (Wilder's smoothing): alpha = 1 / length, SMA-seeded."""
    return _seeded_ma(src, length, 1 / length)


def rsi(src: pd.Series, length: int) -> pd.Series:
    """ta.rsi: RMA of gains over RMA of losses."""
    change = src.diff()
    up = rma(change.clip(lower=0).iloc[1:], length).reindex(src.index)
    down = rma((-change).clip(lower=0).iloc[1:], length).reindex(src.index)
    rs = up / down
    out = 100 - 100 / (1 + rs)
    out[(down == 0) & up.notna()] = 100.0
    return out


def crossover(a: pd.Series | float, b: pd.Series | float) -> pd.Series:
    """ta.crossover: a > b now and a <= b on the previous bar."""
    a, b = _align(a, b)
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series | float, b: pd.Series | float) -> pd.Series:
    """ta.crossunder: a < b now and a >= b on the previous bar."""
    a, b = _align(a, b)
    return (a < b) & (a.shift(1) >= b.shift(1))


def _align(a, b):
    if not isinstance(a, pd.Series):
        a = pd.Series(float(a), index=b.index)
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return a, b
