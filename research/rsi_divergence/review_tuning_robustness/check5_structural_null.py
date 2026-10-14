"""Check 5 (extra): is the selected edge more than the structure 'long-only, wide target, long hold'?

From the cache: OOS avg R (2017-2026 and 2018-2026) of a random config per year drawn from
pools that share the dominant pick's structural features, versus the headline walk-forward.
Also the dominant config's own record and the share of configs in each pool positive in 2018-2026.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from wf import TEST_YEARS, load_cache, stats  # noqa: E402

HEAD = {"avg_r": 0.581, "avg_r_ex2017": 0.237}


def main():
    configs, results = load_cache()
    pools = {
        "all": lambda c: True,
        "long": lambda c: c["sides"] == "long",
        "long_rr4": lambda c: c["sides"] == "long" and c["rr"] == 4.0,
        "long_rr4_hold60": lambda c: c["sides"] == "long" and c["rr"] == 4.0 and c["max_hold"] == 60,
        "long_rr>=3_hold60": lambda c: c["sides"] == "long" and c["rr"] >= 3.0 and c["max_hold"] == 60,
        "long_hidden_rr4_hold60": lambda c: c["sides"] == "long" and c["kind"] == "hidden" and c["rr"] == 4.0 and c["max_hold"] == 60,
    }
    rng = np.random.default_rng(5)
    out = {}
    for name, f in pools.items():
        idx = [i for i, c in enumerate(configs) if f(c)]
        v, vx = [], []
        for _ in range(4000):
            ch = {y: idx[int(rng.integers(len(idx)))] for y in TEST_YEARS}
            rs = [r for y in TEST_YEARS for r in results[ch[y]][y]]
            ex = [r for y in TEST_YEARS if y != 2017 for r in results[ch[y]][y]]
            v.append(np.mean(rs) if rs else 0.0)
            vx.append(np.mean(ex) if ex else 0.0)
        v, vx = np.array(v), np.array(vx)
        # Each config traded every year 2018-2026 (pooled), and its sign.
        own = [stats([r for y in range(2018, 2027) for r in results[i][y]]) for i in idx]
        own_avg = np.array([s["avg_r"] for s in own if s["n"] >= 20])
        out[name] = {"n_configs": len(idx),
                     "random_pick_median_2017_2026": round(float(np.median(v)), 3),
                     "random_pick_p95_2017_2026": round(float(np.quantile(v, .95)), 3),
                     "pct_below_headline": round(100 * float(np.mean(v < HEAD["avg_r"])), 1),
                     "random_pick_median_2018_2026": round(float(np.median(vx)), 3),
                     "random_pick_p95_2018_2026": round(float(np.quantile(vx, .95)), 3),
                     "pct_below_headline_ex2017": round(100 * float(np.mean(vx < HEAD["avg_r_ex2017"])), 1),
                     "configs_2018_2026_median_avg_r": round(float(np.median(own_avg)), 3) if len(own_avg) else None,
                     "configs_2018_2026_share_positive_pct": round(100 * float(np.mean(own_avg > 0)), 1) if len(own_avg) else None}
        print(name, json.dumps(out[name]))
    (HERE / "check5_structural_null.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
