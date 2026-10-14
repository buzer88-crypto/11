"""Load Bitstamp 1-minute data (same files and dedup rule as build_data.py) into a numpy cache."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.environ.get("BITSTAMP_DIR", "/home/user/ff137/bitstamp-btcusd-minute-data")) / "data"
FILES = [SRC / "historical" / "btcusd_bitstamp_1min_2012-2025.csv.gz", SRC / "updates" / "btcusd_bitstamp_1min_latest.csv"]
CACHE = Path(os.environ.get("MINUTE_CACHE", "/tmp/claude-0/-home-user-11/a2d2f1f9-f75b-5d1f-882c-b11976b97671/scratchpad/minutes.npz"))


def load():
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    parts, last_ts = [], -1
    for f in FILES:
        for chunk in pd.read_csv(f, chunksize=2_000_000):
            chunk = chunk[chunk["timestamp"] > last_ts].sort_values("timestamp")
            if chunk.empty:
                continue
            last_ts = int(chunk["timestamp"].iloc[-1])
            parts.append(chunk)
    m = pd.concat(parts, ignore_index=True)
    m = m[m["timestamp"] >= 1356998400]          # 2013-01-01, like run.load
    out = {"ts": m["timestamp"].to_numpy(np.int64)}
    for k in ("open", "high", "low", "close", "volume"):
        out[k] = m[k].to_numpy(np.float64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load()
    ts = d["ts"]
    print(len(ts), pd.to_datetime(ts[0], unit="s"), pd.to_datetime(ts[-1], unit="s"))
    print("monotonic", bool((np.diff(ts) > 0).all()), "non-minute-aligned", int((ts % 60 != 0).sum()))
    gaps = np.diff(ts) // 60 - 1
    print("missing minutes total", int(gaps.sum()), "share", float(gaps.sum() / (len(ts) + gaps.sum())))
    print("seconds-of-hour==0 rows (bucket starts present):", int((ts % 14400 == 0).sum()), "of", (ts[-1] - ts[0]) // 14400)
