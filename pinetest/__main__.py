"""Usage: python -m pinetest <strategy> <ohlc.csv>

The CSV needs open, high, low, close columns (TradingView's "Export chart data" works).
"""
import json
import sys

import pandas as pd

from .backtest import run
from .strategies import STRATEGIES


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in STRATEGIES:
        print(__doc__.strip(), f"\nStrategies: {', '.join(STRATEGIES)}", file=sys.stderr)
        return 2
    df = pd.read_csv(argv[1])
    df.columns = [col.strip().lower() for col in df.columns]
    result = run(df, STRATEGIES[argv[0]](df))
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
