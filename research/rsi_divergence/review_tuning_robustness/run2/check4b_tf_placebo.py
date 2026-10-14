"""Check 4b: long-drift baseline per timeframe variant of check 4.

For each variant, long market entries at random bars of the same calendar year as each real
trade, with that trade's risk (% of entry), the variant's rr and max_hold, 0.1%/side, one trade
at a time. Avg R per draw for 2018-2026; where does the real result sit?
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUTD = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check4_timeframes import JOBS  # noqa: E402
from sim import YEARS, df_for, evaluate  # noqa: E402

from pinetest.divergence import rsi_divergences  # noqa: E402
from pinetest.events import Setup, simulate  # noqa: E402

N_DRAWS = 200


def run(job):
    tf, name, cfg = job
    if cfg["sides"] != "long":
        return None
    df = df_for(tf)
    o = df["open"].to_numpy()
    years = df.index.year.to_numpy()
    setups = rsi_divergences(df, kind=cfg["kind"], rsi_len=cfg["rsi_len"], left=cfg["lb"], right=cfg["lb"],
                             range_lower=cfg.get("range_lower", 5), range_upper=cfg.get("range_upper", 60),
                             rr=cfg["rr"], buffer_atr=cfg["buffer_atr"], max_hold=cfg["max_hold"],
                             confirm=cfg["confirm"], trend=cfg["trend"], extreme=cfg["extreme"], sides=(1,))
    stop = {s.bar: s.stop for s in setups}
    by, trades = evaluate(cfg, tf, with_trades=True)
    real = [x for y in range(2018, 2027) for x in by[y]]
    rp = [(int(years[t.bar]), (t.entry - stop[t.bar]) / t.entry) for t in trades if 2018 <= int(years[t.bar]) <= 2026]
    year_bars = {y: np.flatnonzero(years == y) for y in YEARS}
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(N_DRAWS):
        st = []
        for y, r in rp:
            b = int(year_bars[y][rng.integers(len(year_bars[y]) - 1)])
            if b + 1 >= len(o):
                continue
            ref = o[b + 1]
            st.append(Setup(b, 1, "market", ref * (1 - r), ref * (1 + cfg["rr"] * r), max_hold=cfg["max_hold"]))
        tr = simulate(df, st, 0.1)
        vals.append(float(np.mean([t.r for t in tr])) if tr else 0.0)
    vals = np.array(vals)
    ra = float(np.mean(real)) if real else None
    return {"tf": tf, "variant": name, "n_real_2018_2026": len(real), "real_avg_r_2018_2026": round(ra, 3) if ra is not None else None,
            "placebo_median": round(float(np.median(vals)), 3), "placebo_p95": round(float(np.quantile(vals, .95)), 3),
            "pct_placebo_below_real": round(100 * float(np.mean(vals < ra)), 1) if ra is not None else None}


def main():
    with ProcessPoolExecutor(1) as ex:
        res = [r for r in ex.map(run, JOBS) if r is not None]
    for r in res:
        print(json.dumps(r))
    (OUTD / "check4b_tf_placebo.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
