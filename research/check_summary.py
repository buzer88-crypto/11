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
    return float(s.replace("−", "-").replace("+", "").replace("%", "").replace("*", "").replace("R", "").strip())


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

# 7) RSI divergence section.
div = HERE / "rsi_divergence"
base = json.loads((div / "results.json").read_text())
combo = json.loads((div / "combo_results.json").read_text())
tune = json.loads((div / "tune_results.json").read_text())
sec = README.split("## RSI divergencija")[1].split("## Ispravci")[0]
check("divergence data corr", 0.99, base["data_check"]["daily_return_corr_vs_coinmetrics"], 0.006)
check("divergence median close diff %", 0.12, base["data_check"]["median_abs_close_diff_pct"], 0.006)
check("divergence missing 4h bars", 0, base["data_check"]["missing_4h_bars"], 0)
drows = re.findall(r"^\| [^|]+\(`(run|combo)\.py`\) \| (\d+) \| [^|]+ \| ([^|]+) \| ([^|]+) \| \*\*([^*]+)\*\* \((\d+)\) \| ([^|]+) \|$", sec, re.M)
assert len(drows) == 2, drows
for name, n, pre, train, test, ntest, med in drows:
    d = base if name == "run" else combo
    c = d["chosen"]["cost_0.1"]
    check(f"{name} configs", int(n), len(d["grid_rows"]), 0)
    check(f"{name} pre avg R", num(pre), c["pre"]["avg_r"], 0.006)
    check(f"{name} train avg R", num(train), c["train"]["avg_r"], 0.006)
    check(f"{name} test avg R", num(test), c["test"]["avg_r"], 0.006)
    check(f"{name} test n", int(ntest), c["test"]["n"], 0)
    check(f"{name} grid median test", num(med), d["grid_summary"]["test"]["median_avg_r"], 0.006)
check("base: all test years negative", 5, sum(v["avg_r"] < 0 for v in base["chosen"]["test_years"].values()), 0)
check("base: random timing better (~90%)", 90, 100 - base["chosen"]["placebo_test"]["random_bars_same_side"]["pct_below_real"], 1.0)
check("BTC buy-hold 2022-26 %", 80, base["btc_buy_hold_test_pct"], 0.6)
rules = combo["by_confirmation"]
ema = [rules[k] for k in ("ema50", "ema_9_21", "ema20_reclaim")]
better = [int(e[p]["better_than_plain"].split("/")[0]) for e in ema for p in ("pre", "train", "test")]
check("EMA confirmations better pairs min", 17, min(better), 0)
check("EMA confirmations better pairs max", 24, max(better), 0)
check("EMA confirmations median test min", -0.13, min(e["test"]["median_avg_r"] for e in ema), 0.006)
check("EMA confirmations median test max", -0.04, max(e["test"]["median_avg_r"] for e in ema), 0.006)
check("combo positive in all three", 10, len(combo["positive_in_all_three_periods"]), 0)
gs = combo["grid_summary"]
check("combo expected by chance", 15, combo["n_configs"] * np.prod([gs[k]["n_positive"] / combo["n_configs"] for k in ("pre", "train", "test")]), 0.6)
check("tune space", 76800, tune["space_size"], 0)
check("tune sampled", 2400, tune["n_sampled"], 0)
wrows = re.findall(r"^\| (prošireni prozor|3 godine), (prosjek R|t-statistika) \| ([+−\d.]+)R \((\d+)[^)]*\) \| ([+−\d.]+)R", sec, re.M)
keymap = {("prošireni prozor", "prosjek R"): "window_expanding_avg_r", ("3 godine", "prosjek R"): "window_3_avg_r",
          ("3 godine", "t-statistika"): "window_3_t", ("prošireni prozor", "t-statistika"): "window_expanding_t"}
assert len(wrows) == 4, wrows
for win, metric, allr, n, ex in wrows:
    w = tune["walk_forward"][keymap[(win, metric)]]
    check(f"tune {win}/{metric} OOS avg R", num(allr), w["oos"]["avg_r"], 0.006)
    check(f"tune {win}/{metric} OOS n", int(n), w["oos"]["n"], 0)
    check(f"tune {win}/{metric} without 2017", num(ex), w["excluding_2017"]["avg_r"], 0.006)
check("tune headline t", 2.38, tune["walk_forward"]["window_expanding_avg_r"]["oos"]["t"], 0.006)
check("tune headline n without 2017", 49, tune["walk_forward"]["window_expanding_avg_r"]["excluding_2017"]["n"], 0)
check("TradingView default avg R", -0.21, tune["tradingview_default"]["oos"]["avg_r"], 0.006)
check("TradingView default n", 258, tune["tradingview_default"]["oos"]["n"], 0)
# Numbers quoted from the reviews.
sig = json.loads((div / "review_implementation" / "check1_signals.json").read_text())["signals"]
jac = [v["jaccard_pct"] for k, v in sig.items() if k.startswith("lb5")]
check("review: TradingView signal agreement min %", 97, min(jac), 0.6)
check("review: TradingView signal agreement max %", 100, max(jac), 0.6)
top = json.loads((div / "review_intrabar" / "results_intrabar.json").read_text())["top_configs"]["hidden_lb3_rr3_trend"]
check("review: 1-minute replay trades", 360, sum(top[p]["4h"]["n"] for p in ("pre", "train", "test")), 0)
check("review: 1-minute replay changed outcomes", 1, sum(len(top[p]["changed"]) for p in ("pre", "train", "test")), 0)
rp = json.loads((div / "review_tuning_repro_placebo" / "v2_placebo.json").read_text())
check("review: random long entries avg R", 0.33, rp["P1_uniform"]["pooled_all"]["null_mean"], 0.006)
check("review: real vs random entries percentile", 86, rp["P1_uniform"]["pooled_all"]["pct_null_below"], 0.6)
check("review: percentile without 2017", 62, rp["P1_uniform"]["pooled_ex2017"]["pct_null_below"], 0.6)
check("review: BTC held on the same bars avg R", 0.60, rp["exposure"]["btc_same_bars_R"]["all"], 0.006)
dt = json.loads((div / "review_tuning_repro_placebo" / "v2_detrend.json").read_text())
check("review: detrended picks without 2017", -0.07, dt["a_picks_on_detrended"]["ex2017"]["avg_r"], 0.006)
sel = json.loads((div / "review_tuning_robustness" / "check1_selection.json").read_text())["summary"]
check("review: headline rank among 100 variants", 2, sel["rank_of_headline_among_variants"], 0)
check("review: variants checked", 100, sel["n_variants"], 0)
check("review: median of variants", 0.05, sel["variants_avg_r"]["median"], 0.006)
power = json.loads((div / "review_tuning_stats" / "c2_bootstrap.json").read_text())["power"]
check("review: trades needed for t=2 without 2017 (~200)", 200, power["trades_needed_for_t2_at_ex2017_mean"], 3)
check("review: years needed (~30)", 30, power["years_needed_at_~6_trades_per_year"], 4)

print(f"\n{len(failures)} mismatches" + (": " + ", ".join(failures) if failures else ""))
sys.exit(1 if failures else 0)
