"""Check 2b: two more fresh samples of 600 configs (seeds 2026 and 31337), same space/simulator/costs
as tune.py, each run through the headline walk-forward (expanding, avg R) at min_trades 10..40 and the
3-year avg-R rule. Run from /home/user/11 with PYTHONPATH=. ; 2 processes.
"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUTD = Path(__file__).resolve().parent
HERE = OUTD.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from wf import TEST_YEARS, summarize, walk_forward  # noqa: E402
import tune  # noqa: E402

sys.path.insert(0, str(OUTD))
from check2_sampling import fresh_configs  # noqa: E402

SEEDS = (2026, 31337)


def main():
    out = {}
    for seed in SEEDS:
        cfgs = fresh_configs(seed, 600)
        p = OUTD / f"fresh_{seed}.pkl"
        if p.exists():
            res = pickle.loads(p.read_bytes())["results"]
        else:
            with ProcessPoolExecutor(2, initializer=tune._init) as ex:
                res = list(ex.map(tune.evaluate, cfgs, chunksize=10))
            p.write_bytes(pickle.dumps({"configs": cfgs, "results": res, "seed": seed}))
        has_dom = any(c == {"kind": "hidden", "rsi_len": 21, "lb": 8, "rr": 4.0, "buffer_atr": 0.0, "max_hold": 60,
                            "confirm": "ema_9_21", "trend": False, "sides": "long", "extreme": False} for c in cfgs)
        rows = {}
        for window, mts in ((0, (10, 15, 20, 30, 40)), (3, (20,))):
            for mt in mts:
                pk, oos = walk_forward(res, window, "avg_r", mt)
                s = summarize(pk, oos, cfgs)
                rows[f"{window or 'exp'}_avg_r_min{mt}"] = {k: s[k] for k in ("n", "avg_r", "t", "n_ex2017", "avg_r_ex2017", "t_ex2017",
                                                                             "n_distinct", "long_only_share", "per_year_avg_r")}
                print(seed, window or "exp", mt, s["n"], s["avg_r"], s["t"], s["n_ex2017"], s["avg_r_ex2017"], s["t_ex2017"],
                      s["long_only_share"], flush=True)
        long_idx = [i for i, c in enumerate(cfgs) if c["sides"] == "long"]
        rng = np.random.default_rng(seed)
        null, nullx = [], []
        for _ in range(4000):
            ch = {y: long_idx[int(rng.integers(len(long_idx)))] for y in TEST_YEARS}
            rs = [r for y in TEST_YEARS for r in res[ch[y]][y]]
            ex = [r for y in TEST_YEARS if y != 2017 for r in res[ch[y]][y]]
            null.append(np.mean(rs) if rs else 0.0)
            nullx.append(np.mean(ex) if ex else 0.0)
        null, nullx = np.array(null), np.array(nullx)
        h = rows["exp_avg_r_min20"]
        out[str(seed)] = {"contains_dominant": has_dom, "rows": rows,
                          "random_long_null_median": round(float(np.median(null)), 3),
                          "random_long_null_median_ex2017": round(float(np.median(nullx)), 3),
                          "headline_pct_random_long_below": round(100 * float(np.mean(null < h["avg_r"])), 1),
                          "headline_pct_random_long_below_ex2017": round(100 * float(np.mean(nullx < h["avg_r_ex2017"])), 1)}
        print(seed, "dominant in sample:", has_dom, "null", out[str(seed)]["random_long_null_median"],
              "pct below", out[str(seed)]["headline_pct_random_long_below"], out[str(seed)]["headline_pct_random_long_below_ex2017"])
    (OUTD / "check2b_more_fresh.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
