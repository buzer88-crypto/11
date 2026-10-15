"""Build BTC/USD 1h, 4h and 1d OHLCV bars (UTC) from Bitstamp 1-minute data.

Source: github.com/ff137/bitstamp-btcusd-minute-data (historical 2012-2025 + latest updates),
cloned to /home/user/ff137/bitstamp-btcusd-minute-data (override with BITSTAMP_DIR).
Output: data/BTCUSD_{5m,15m,1h,4h,12h,1d}.csv in the repo (git-ignored).
"""
import os
from pathlib import Path

import pandas as pd

SRC = Path(os.environ.get("BITSTAMP_DIR", "/home/user/ff137/bitstamp-btcusd-minute-data")) / "data"
OUT = Path(__file__).resolve().parents[2] / "data"
FILES = [SRC / "historical" / "btcusd_bitstamp_1min_2012-2025.csv.gz", SRC / "updates" / "btcusd_bitstamp_1min_latest.csv"]
FRAMES = {"5m": 300, "15m": 900, "1h": 3600, "4h": 4 * 3600, "12h": 12 * 3600, "1d": 86400}
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def main():
    parts = {k: [] for k in FRAMES}
    last_ts = -1
    for f in FILES:
        for chunk in pd.read_csv(f, chunksize=2_000_000):
            chunk = chunk[chunk["timestamp"] > last_ts].sort_values("timestamp")
            if chunk.empty:
                continue
            last_ts = int(chunk["timestamp"].iloc[-1])
            for k, sec in FRAMES.items():
                parts[k].append(chunk.groupby(chunk["timestamp"] // sec * sec).agg(AGG))
    OUT.mkdir(exist_ok=True)
    for k, ps in parts.items():
        # A bucket can straddle two chunks: combine again in order.
        df = pd.concat(ps).groupby(level=0, sort=True).agg(AGG)
        df.index = pd.to_datetime(df.index, unit="s", utc=True).tz_localize(None)
        df.index.name = "date"
        df.to_csv(OUT / f"BTCUSD_{k}.csv")
        print(k, len(df), df.index[0], "->", df.index[-1])


if __name__ == "__main__":
    main()
