import numpy as np
import pandas as pd
import pytest


def make_ohlc(closes, spread=0.5):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) + spread,
        "low": np.minimum(opens, closes) - spread,
        "close": closes,
    })


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
    return make_ohlc(closes, spread=0.3)
