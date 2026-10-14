"""Session-2: random-config-pick nulls from the cache (each year a random config from a pool),
to compare with tune.py's 99.7% / 98.6% and with pools shaped like the picks."""
import json, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
d = pickle.loads((ROOT / "data" / "tune_cache.pkl").read_bytes())
cf, res = d["configs"], d["results"]
TEST = list(range(2017, 2027))
rng = np.random.default_rng(5150)
real = {"all": 0.581, "ex2017": 0.2373}
pools = {"all": list(range(len(cf))),
         "long": [i for i, c in enumerate(cf) if c["sides"] == "long"],
         "long_rr4_hold60": [i for i, c in enumerate(cf) if c["sides"] == "long" and c["rr"] == 4.0 and c["max_hold"] == 60],
         "long_rr4_hold60_hidden": [i for i, c in enumerate(cf) if c["sides"] == "long" and c["rr"] == 4.0 and c["max_hold"] == 60 and c["kind"] == "hidden"]}
out = {}
for name, pool in pools.items():
    pool = np.array(pool)
    a, e = [], []
    for _ in range(5000):
        ra, re_ = [], []
        for y in TEST:
            rs = res[int(pool[rng.integers(len(pool))])][y]
            ra += rs
            if y != 2017:
                re_ += rs
        a.append(np.mean(ra) if ra else np.nan); e.append(np.mean(re_) if re_ else np.nan)
    a, e = np.array(a), np.array(e)
    a, e = a[np.isfinite(a)], e[np.isfinite(e)]
    out[name] = {"pool": int(len(pool)), "all_median": round(float(np.median(a)), 3), "all_pct_below_real": round(100 * float((a < real["all"]).mean()), 1),
                 "ex2017_median": round(float(np.median(e)), 3), "ex2017_pct_below_real": round(100 * float((e < real["ex2017"]).mean()), 1)}
    print(name, out[name])
(Path(__file__).parent / "v2_randpick.json").write_text(json.dumps(out, indent=1))
