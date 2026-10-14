"""Check every number quoted in research/README.md against the results.json files.

Run: cd /home/user/11 && python3 research/check_summary.py   (exit code 1 on any mismatch)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()
FAMILIES = ["trend_ts", "xs_momentum", "onchain", "regime_vol", "calendar", "mean_reversion"]
failures = []


def load(name):
    return json.loads((HERE / name / "results.json").read_text())


def check(label, quoted, actual, tol):
    ok = actual is not None and abs(quoted - actual) <= tol
    print(f"{'ok ' if ok else 'BAD'} {label}: README {quoted} vs results {actual}")
    if not ok:
        failures.append(label)


def n_configs(d):
    return d["n_configs_tested"] if "n_configs_tested" in d else d["meta"]["n_configs_tested"]


def test_sharpe(row):
    for path in (("test", "sharpe"), ("test_sharpe",), ("metrics", "test", "sharpe")):
        x = row
        for k in path:
            x = x.get(k) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            return float(x)
    return None


def grid_rows(d):
    g = d.get("grid") or []
    return list(g.values()) if isinstance(g, dict) else g


def num(s):
    return float(s.replace("−", "-").replace("+", "").replace("%", "").replace("*", "").strip())


# 1) Total number of variants.
counts = {k: n_configs(load(k)) for k in FAMILIES}
total = sum(counts.values())
print("variants per family:", counts, "total", total)
check("total variants (Ukratko)", int(re.search(r"ukupno (\d+) varijanti", README).group(1)), total, 0)
check("BTC trend variants", 32, load("btc_trend")["n_configs"], 0)

# 2) Results table, one row per family in README order.
rows = re.findall(r"^\| [^|]+\(`(\w+)`\) \| (\d+) \| [^|]+ \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$",
                  README, re.M)
assert [r[0] for r in rows] == FAMILIES, [r[0] for r in rows]
for key, n, tr, te, tot, med, beat in rows:
    d = load(key)
    c = d["chosen"]
    ts = [s for s in (test_sharpe(r) for r in grid_rows(d) if isinstance(r, dict)) if s is not None]
    check(f"{key} variants", int(n), n_configs(d), 0)
    check(f"{key} grid rows found", int(n), len(ts), 0)
    check(f"{key} train Sharpe", num(tr), c["train"]["sharpe"], 0.006)
    check(f"{key} test Sharpe", num(te), c["test"]["sharpe"], 0.006)
    check(f"{key} test total %", num(tot), c["test"]["total_pct"], 0.6)
    check(f"{key} grid median test Sharpe", num(med), float(np.median(ts)), 0.006)
    check(f"{key} configs > 0.48", num(beat), sum(s > 0.48 for s in ts), 0)

# 3) Benchmark and BTC trend section.
b = load("btc_trend")
bh, ch = b["benchmarks"]["btc"], b["chosen"]["btc"]
check("BTC B&H test Sharpe", 0.48, bh["test"]["sharpe"], 0.006)
check("BTC B&H test total %", 65, bh["test"]["total_pct"], 0.6)
check("BTC B&H test max DD %", -67, bh["test"]["max_dd_pct"], 0.6)
check("BTC B&H train Sharpe", 0.78, bh["train"]["sharpe"], 0.006)
check("BTC B&H pre Sharpe", 1.37, bh["pre"]["sharpe"], 0.006)
check("BTC B&H pre total %", 1808, bh["pre"]["total_pct"], 0.6)
check("BTC B&H pre max DD %", -81, bh["pre"]["max_dd_pct"], 0.6)
check("trend train Sharpe", 1.46, ch["train"]["sharpe"], 0.006)
check("trend test Sharpe", 0.74, ch["test"]["sharpe"], 0.006)
check("trend test total %", 109, ch["test"]["total_pct"], 0.6)
check("trend test max DD %", -40, ch["test"]["max_dd_pct"], 0.6)
check("trend pre Sharpe", 1.83, ch["pre"]["sharpe"], 0.006)
check("trend pre total %", 825, ch["pre"]["total_pct"], 0.6)
check("trend pre max DD %", -28, ch["pre"]["max_dd_pct"], 0.6)
check("trend grid beat B&H test", 29, b["grid_btc"]["n_beat_bh_sharpe"]["test"], 0)
check("trend grid beat B&H pre", 31, b["grid_btc"]["n_beat_bh_sharpe"]["pre"], 0)
check("trend grid better DD test", 32, b["grid_btc"]["n_better_max_dd"]["test"], 0)
check("trend grid better DD pre", 32, b["grid_btc"]["n_better_max_dd"]["pre"], 0)
check("trend placebo test pct", 88, b["chosen"]["placebo_percentile_test"], 0.6)
check("trend placebo pre pct", 99, b["chosen"]["placebo_percentile_pre"], 0.6)
check("trend grid median test Sharpe (rerun table)", 0.63, b["grid_btc"]["median_sharpe"]["test"], 0.006)

# 4) "86 of 105" drawdown claim (regime_vol).
g = load("regime_vol")["grid_summary"]["all"]
check("regime configs", 105, g["n"], 0)
check("regime: static same exposure DD as good (86 of 105)", 86, g["n"] - g["n_dd_better_than_static_same_exposure"], 0)

# 5) Rerun table: the "after" column must match the current results.
after = re.findall(r"^\| ([^|]+) \| [^|]+ \| (−?[\d.]+) / (−?[\d.]+)", README.split("Usporedba prije")[1], re.M)
names = {"Trend po tokenu": "trend_ts", "Rotacija tokena": "xs_momentum", "On-chain": "onchain",
         "Rizik i režimi": "regime_vol", "Kalendar": "calendar", "Povratak cijene": "mean_reversion"}
for label, te, med in after:
    label = label.strip()
    if label == "BTC trend":
        check("rerun BTC trend test", num(te), ch["test"]["sharpe"], 0.006)
        continue
    d = load(names[label])
    ts = [s for s in (test_sharpe(r) for r in grid_rows(d) if isinstance(r, dict)) if s is not None]
    check(f"rerun {label} test", num(te), d["chosen"]["test"]["sharpe"], 0.006)
    check(f"rerun {label} grid median", num(med), float(np.median(ts)), 0.006)

# 6) SMC table.
smc = load("smc_eurusd")["patterns"]
smc_rows = re.findall(r"^\| (Liquidity sweep|Fair value gap|Order block) \| [^|]+ \| ([^ ]+) \((\d+)\) \| \*\*([^*]+)\*\* \((\d+)\) \| ([^|]+) \|$",
                      README, re.M)
key = {"Liquidity sweep": "sweep", "Fair value gap": "fvg", "Order block": "order_block"}
assert len(smc_rows) == 3, smc_rows
for label, tr, ntr, te, nte, med in smc_rows:
    p = smc[key[label]]
    check(f"SMC {label} train avg R", num(tr), p["chosen_train"]["avg_r"], 0.006)
    check(f"SMC {label} train n", int(ntr), p["chosen_train"]["n"], 0)
    check(f"SMC {label} test avg R", num(te), p["chosen_test"]["avg_r"], 0.006)
    check(f"SMC {label} test n", int(nte), p["chosen_test"]["n"], 0)
    check(f"SMC {label} grid median test", num(med), p["grid_median_test_avg_r"], 0.006)
ob = smc["order_block"]
check("SMC order block train-negative configs (7 of 9)", 7, sum(r["train"]["avg_r"] < 0 for r in ob["grid"]), 0)
ns = [r["test"]["n"] for r in ob["grid"]]
check("SMC order block min test trades", 9, min(ns), 0)
check("SMC order block max test trades", 33, max(ns), 0)
check("SMC order block placebo (same bars)", 100, ob["placebo_same_bars_random_side_pct_below_real"], 0.6)
check("SMC order block placebo (random bars)", 99, ob["placebo_random_bars_pct_below_real"], 1.4)

print(f"\n{len(failures)} mismatches" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
