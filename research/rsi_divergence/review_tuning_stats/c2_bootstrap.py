"""Check 2: bootstrap 90% intervals for the expanding-window / avg-R walk-forward OOS average R,
by trade and by calendar year (block = all OOS trades of a year), with and without 2017.
Output: c2_bootstrap.json.  PYTHONPATH=. python .../c2_bootstrap.py"""
import json

import numpy as np

from wf_lib import OUT, TEST_YEARS, arrays_from_results, load_cache, walk_forward

B = 20000


def tstat(r):
    r = np.asarray(r, float)
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


def boot(blocks, rng, by):
    """blocks: list of arrays (one per year). by: 'trade' or 'year'. Returns bootstrap means."""
    if by == "trade":
        r = np.concatenate(blocks)
        idx = rng.integers(0, len(r), (B, len(r)))
        return r[idx].mean(1)
    sums = np.array([b.sum() for b in blocks])
    cnts = np.array([len(b) for b in blocks])
    idx = rng.integers(0, len(blocks), (B, len(blocks)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def describe(blocks, rng):
    r = np.concatenate(blocks)
    out = {"n": int(len(r)), "n_years": len(blocks), "avg_r": round(float(r.mean()), 3), "t": round(tstat(r), 2),
           "median_r": round(float(np.median(r)), 3), "win_rate_pct": round(100 * float((r > 0).mean()), 1)}
    for by in ("trade", "year"):
        m = boot(blocks, rng, by)
        out[f"by_{by}"] = {"ci90": [round(float(np.quantile(m, 0.05)), 3), round(float(np.quantile(m, 0.95)), 3)],
                           "ci80": [round(float(np.quantile(m, 0.10)), 3), round(float(np.quantile(m, 0.90)), 3)],
                           "P_boot_mean_le_0": round(float((m <= 0).mean()), 4)}
    # equal-weight-by-year average (each year counts once)
    ym = np.array([b.mean() for b in blocks])
    idx = rng.integers(0, len(ym), (B, len(ym)))
    em = ym[idx].mean(1)
    out["equal_weight_years"] = {"avg_of_year_avgs": round(float(ym.mean()), 3),
                                 "n_years_positive": int((ym > 0).sum()),
                                 "ci90": [round(float(np.quantile(em, 0.05)), 3), round(float(np.quantile(em, 0.95)), 3)]}
    return out


def main():
    cache = load_cache()
    results = cache["results"]
    N, S, Q = arrays_from_results(results)
    picks, _ = walk_forward(N, S, Q, 0, "avg_r")
    blocks = {int(y): np.asarray(results[p][int(y)], float) for p, y in zip(picks, TEST_YEARS)}
    rng = np.random.default_rng(4242)
    out = {"B": B, "per_year": {y: {"n": len(b), "avg_r": round(float(b.mean()), 3), "total_r": round(float(b.sum()), 2)}
                                for y, b in blocks.items()}}
    out["all_2017_2026"] = describe(list(blocks.values()), rng)
    out["ex2017"] = describe([b for y, b in blocks.items() if y != 2017], rng)
    r = np.concatenate(list(blocks.values()))
    tot = r.sum()
    srt = np.sort(r)[::-1]
    out["concentration"] = {"total_r": round(float(tot), 2), "share_of_total_from_2017": round(float(blocks[2017].sum() / tot), 3),
                            "total_r_without_best_5_trades": round(float(srt[5:].sum()), 2),
                            "avg_r_without_best_5_trades": round(float(srt[5:].mean()), 3),
                            "n_trades_hit_4R_target": int((r > 3.5).sum())}
    # What sample size would make +0.237R (ex-2017) significant at t=2 given its dispersion
    ex = np.concatenate([b for y, b in blocks.items() if y != 2017])
    sd = ex.std(ddof=1)
    out["power"] = {"ex2017_sd_r": round(float(sd), 3),
                    "trades_needed_for_t2_at_ex2017_mean": int(np.ceil((2 * sd / ex.mean()) ** 2)),
                    "years_needed_at_~6_trades_per_year": round(float((2 * sd / ex.mean()) ** 2 / 6), 1)}
    (OUT / "c2_bootstrap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
