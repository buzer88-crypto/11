"""Static checks that the Pine scripts and their Python ports stay in sync."""
import inspect
import re
from pathlib import Path

import pytest

from pinetest.strategies import ema_breakeven, ema_cross, rsi_reversion

PINE = Path(__file__).parent.parent / "pine"

CASES = [
    ("ema_cross.pine", ema_cross, {"fastLen": "fast_len", "slowLen": "slow_len"}),
    ("rsi_reversion.pine", rsi_reversion, {"rsiLen": "rsi_len", "oversold": "oversold",
     "overbought": "overbought", "slPct": "sl_pct", "tpPct": "tp_pct"}),
    ("ema_breakeven.pine", ema_breakeven, {"fastLen": "fast_len", "slowLen": "slow_len",
     "slPct": "sl_pct", "tpPct": "tp_pct", "beTrig": "be_pct", "volLen": "vol_len",
     "volMult": "vol_mult"}),
]


@pytest.mark.parametrize("fname,func,mapping", CASES)
def test_pine_inputs_match_python_defaults(fname, func, mapping):
    src = (PINE / fname).read_text()
    assert src.startswith("//@version=6")
    assert "strategy(" in src
    assert "default_qty_value = 100" in src and "commission_value = 0.1" in src
    defaults = inspect.signature(func).parameters
    for pine_name, py_name in mapping.items():
        m = re.search(rf"^{pine_name}\s*=\s*input\.\w+\(([\d.]+)", src, re.M)
        assert m, f"{pine_name} input not found in {fname}"
        assert float(m.group(1)) == float(defaults[py_name].default)


@pytest.mark.parametrize("fname", [c[0] for c in CASES])
def test_pine_brackets_balanced(fname):
    src = re.sub(r'"[^"]*"|//.*', "", (PINE / fname).read_text())
    for o, c in ("()", "[]", "{}"):
        assert src.count(o) == src.count(c)


def test_pine_volume_filter_off_by_default():
    src = (PINE / "ema_breakeven.pine").read_text()
    assert re.search(r"^useVol\s*=\s*input\.bool\(false", src, re.M)
    assert inspect.signature(ema_breakeven).parameters["vol_filter"].default is False


@pytest.mark.parametrize("fname", [c[0] for c in CASES])
def test_pine_ta_calls_not_inside_if_blocks(fname):
    """ta.* inside an if block is not evaluated on every bar, which breaks its history."""
    for line in (PINE / fname).read_text().splitlines():
        if line.startswith((" ", "\t")):
            assert "ta." not in line, line
