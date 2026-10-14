"""Re-run all 2400 cached configs once and keep per-trade metadata (signal bar, fill bar, exit bar,
side, risk as % of entry, R). Verifies the R values per year equal tune_cache.pkl exactly.
Output: trades_meta.pkl = {"configs": [...], "trades": [np.ndarray (k, 6)], "match": bool}.
Run from /home/user/11 with PYTHONPATH=. (2 processes)."""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "rsi_divergence"))
from run import COST, load  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import simulate  # noqa: E402

OUT = Path(__file__).parent
_df = None


def _init():
    global _df
    _df = load("4h")


def evaluate(cfg):
    df = _df
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"],
                             rr=cfg["rr"], buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"],
                             confirm=cfg["confirm"], trend=cfg["trend"], extreme=cfg["extreme"],
                             sides=(1, -1) if cfg["sides"] == "both" else (1,))
    rows = []
    for t in simulate(df, setups, COST):
        risk = t.side * (t.entry - [s for s in setups if s.bar == t.bar and s.side == t.side][0].stop)
        rows.append((t.bar, t.fill_bar, t.exit_bar, t.side, risk / t.entry, t.r))
    return np.array(rows, dtype=float).reshape(-1, 6)


def main():
    cache = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
    configs = cache["configs"]
    with ProcessPoolExecutor(2, initializer=_init) as ex:
        trades = list(ex.map(evaluate, configs, chunksize=25))
    df = load("4h")
    years = df.index.year.to_numpy()
    ok = True
    for k, (tr, by_year) in enumerate(zip(trades, cache["results"])):
        got = {y: [] for y in cache["years"]}
        for row in tr:
            y = int(years[int(row[0])])
            if y in got:
                got[y].append(round(row[5], 4))
        if got != {y: list(v) for y, v in by_year.items()}:
            ok = False
            print("mismatch config", k)
    print("all configs match cache:", ok)
    (OUT / "trades_meta.pkl").write_bytes(pickle.dumps({"configs": configs, "trades": trades, "match": ok}))


if __name__ == "__main__":
    main()
