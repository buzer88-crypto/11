"""Check 5: how informative is the study's 'beats 99.7% of random picks' null?
Reproduce the study's random-pick null (a random config per test year) and compare its spread with
the spread of the actual pick sequence; then a matched random-pick null that each year draws a random
config with the SAME sides, rr and max_hold as the actual pick (so the same payoff shape and drift
exposure, only the divergence/confirmation details are random).
PYTHONPATH=. python .../c5_random_pick_nulls.py"""
import json

import numpy as np

from wf_lib import OUT, TEST_YEARS, arrays_from_results, load_cache, walk_forward

cache = load_cache()
configs, results = cache["configs"], cache["results"]
N, S, Q = arrays_from_results(results)
rng = np.random.default_rng(5)
B = 20000
out = {}
for w, m in ((0, "avg_r"), (3, "avg_r")):
    picks, (n, s, q) = walk_forward(N, S, Q, w, m)
    real = s / n
    real_ex = sum(S[p, yi] for p, yi in zip(picks, range(3, 13)) if yi != 3) / sum(N[p, yi] for p, yi in zip(picks, range(3, 13)) if yi != 3)
    pools = {"all": [np.arange(len(configs))] * 10,
             "long_only": [np.array([i for i, c in enumerate(configs) if c["sides"] == "long"])] * 10,
             "same_sides_rr_hold_as_pick": [np.array([i for i, c in enumerate(configs)
                                                      if (c["sides"], c["rr"], c["max_hold"]) == (configs[p]["sides"], configs[p]["rr"], configs[p]["max_hold"])])
                                            for p in picks]}
    res = {"real_avg_r": round(float(real), 3), "real_ex2017_avg_r": round(float(real_ex), 3), "real_n": int(n)}
    for name, pool in pools.items():
        ks = np.stack([pl[rng.integers(0, len(pl), B)] for pl in pool], 1)          # B x 10
        yi = np.arange(3, 13)[None, :]
        nn, ss = N[ks, yi], S[ks, yi]
        avg = ss.sum(1) / np.maximum(nn.sum(1), 1)
        avg_ex = ss[:, 1:].sum(1) / np.maximum(nn[:, 1:].sum(1), 1)
        res[name] = {"pool_sizes": sorted({len(p) for p in pool}), "median_avg_r": round(float(np.median(avg)), 3),
                     "sd_avg_r": round(float(avg.std()), 3), "median_n": int(np.median(nn.sum(1))),
                     "pct_below_real": round(100 * float((avg < real).mean()), 1),
                     "ex2017_median_avg_r": round(float(np.median(avg_ex)), 3),
                     "ex2017_pct_below_real": round(100 * float((avg_ex < real_ex).mean()), 1)}
    out[f"window_{w or 'expanding'}_{m}"] = res
print(json.dumps(out, indent=1))
(OUT / "c5_random_pick_nulls.json").write_text(json.dumps(out, indent=1))
