"""Extra: random-pick null restricted to configs shaped like the picks (long-only, rr 4,
max_hold 60), drawn from the same tune_cache. Also: with-replacement walk-forward null that
picks among those configs each year."""
import json
from pathlib import Path
import numpy as np
from lib import TEST, cache

OUT = Path(__file__).parent
d = cache()
cf, res = d["configs"], d["results"]
real_all, real_ex = 0.581, 0.237
rng = np.random.default_rng(11)
pools = {
    "long_only": [i for i, c in enumerate(cf) if c["sides"] == "long"],
    "long_rr4": [i for i, c in enumerate(cf) if c["sides"] == "long" and c["rr"] == 4.0],
    "long_rr4_hold60": [i for i, c in enumerate(cf) if c["sides"] == "long" and c["rr"] == 4.0 and c["max_hold"] == 60],
    "long_rr4_hold60_hidden": [i for i, c in enumerate(cf) if c["sides"] == "long" and c["rr"] == 4.0 and c["max_hold"] == 60 and c["kind"] == "hidden"],
}
out = {}
for name, pool in pools.items():
    na, ne = [], []
    for _ in range(5000):
        ra, re_ = [], []
        for y in TEST:
            rs = res[pool[int(rng.integers(len(pool)))]][y]
            ra += rs
            if y != 2017:
                re_ += rs
        na.append(np.mean(ra) if ra else np.nan); ne.append(np.mean(re_) if re_ else np.nan)
    na, ne = np.array(na), np.array(ne)
    na, ne = na[np.isfinite(na)], ne[np.isfinite(ne)]
    # every config of the pool traded 2017-2026 in full (fixed config, no tuning)
    fixed = np.array([np.mean([r for y in TEST for r in res[i][y]]) for i in pool if sum(len(res[i][y]) for y in TEST) >= 30])
    fixed_ex = np.array([np.mean([r for y in TEST if y != 2017 for r in res[i][y]]) for i in pool if sum(len(res[i][y]) for y in TEST if y != 2017) >= 30])
    out[name] = {"pool_size": len(pool),
                 "random_pick_all": {"median": round(float(np.median(na)), 3), "pct_below_real": round(100 * float((na < real_all).mean()), 1)},
                 "random_pick_ex2017": {"median": round(float(np.median(ne)), 3), "pct_below_real": round(100 * float((ne < real_ex).mean()), 1)},
                 "fixed_config_2017_2026": {"n_configs": int(len(fixed)), "median_avg_r": round(float(np.median(fixed)), 3),
                                            "share_positive": round(float((fixed > 0).mean()), 3),
                                            "pct_below_real": round(100 * float((fixed < real_all).mean()), 1)},
                 "fixed_config_ex2017": {"n_configs": int(len(fixed_ex)), "median_avg_r": round(float(np.median(fixed_ex)), 3),
                                         "share_positive": round(float((fixed_ex > 0).mean()), 3),
                                         "pct_below_real": round(100 * float((fixed_ex < real_ex).mean()), 1)}}
    print(name, out[name])
(OUT / "c5_matched_null.json").write_text(json.dumps(out, indent=1))
