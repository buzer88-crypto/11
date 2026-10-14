"""Cross-sectional momentum / reversal across coins: full grid, TRAIN-only selection, placebos.

Run:  cd /home/user/11 && python3 research/xs_momentum/run.py

Hypotheses (all weekly, decided at the Sunday close with data <= that close, traded by the
shared harness at the next close, lag=1):
  A  long_mom   long the top k coins by past L-day return (optionally skipping the last 7 days)
  B  ls_mom     long top k / short bottom k, dollar neutral, gross 1 (+0.5 / -0.5)
  C1 long_rev   short-term reversal: long the bottom k coins by past L-day return
  C2 ls_rev     reversal long/short: long bottom k / short top k, gross 1
  D  rel_btc    long the top k coins whose past L-day return beats BTC's; empty slots go to BTC
                (k = "all": equal weight every coin beating BTC, 100% BTC if none does)

Selection of the "best" config uses TRAIN Sharpe only. Everything is seeded.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinetest.crypto_data import load_panel, universe  # noqa: E402
from pinetest.portfolio import (TEST, TRAIN, backtest, buy_and_hold, circular_shift,  # noqa: E402
                                equal_weight, permute_assets, rebalance_on)

OUT = Path(__file__).resolve().parent
TMP = Path("/tmp/claude-0/research-xs_momentum")
SEED = 20260524
FREQ = "W-SUN"
# Trading starts at the last weekly rebalance before TRAIN. Signals still use the full history
# before it. Starting in 2010-2017 lets a thin, spike-prone universe kill a long/short book
# before TRAIN even begins (the harness then reports Sharpe 0 for every later period).
START = "2017-12-31"
BTC_TEST_SHARPE = 0.48
EW20_TEST_SHARPE = 0.15

# ---- parameter grid, fixed up front ----------------------------------------------------------
TOP_NS = (10, 20, 30)
LOOKBACKS = (7, 14, 30, 60, 90)
SKIPS = (0, 7)
KS = (3, 5)
REV_LOOKBACKS = (3, 7, 14)
RELBTC_KS = (3, 5, "all")

N_PLACEBO_MAIN = 500     # permute_assets draws for the TRAIN-chosen config
N_PLACEBO_TIMING = 200   # circular_shift draws for the TRAIN-chosen config (supplementary)
N_PLACEBO_FAMILY = 200   # permute_assets draws for the best config of each hypothesis
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)


def build_grid():
    grid = []
    for n in TOP_NS:
        for L in LOOKBACKS:
            for s in SKIPS:
                for k in KS:
                    grid.append(dict(hyp="long_mom", top_n=n, L=L, skip=s, k=k))
                    grid.append(dict(hyp="ls_mom", top_n=n, L=L, skip=s, k=k))
                for k in RELBTC_KS:
                    grid.append(dict(hyp="rel_btc", top_n=n, L=L, skip=s, k=k))
        for L in REV_LOOKBACKS:
            for k in KS:
                grid.append(dict(hyp="long_rev", top_n=n, L=L, skip=0, k=k))
                grid.append(dict(hyp="ls_rev", top_n=n, L=L, skip=0, k=k))
    for i, c in enumerate(grid):
        c["id"] = f"{c['hyp']}_n{c['top_n']}_L{c['L']}_s{c['skip']}_k{c['k']}"
    return grid


# ---- signals: panel -> daily target weights (uses data <= t only) ----------------------------

def past_return(px: pd.DataFrame, L: int, skip: int) -> pd.DataFrame:
    """Return from close t-skip-L to close t-skip. Only past closes; NaN if either is missing."""
    return px.shift(skip) / px.shift(skip + L) - 1


def rank_select(score: pd.DataFrame, eligible: pd.DataFrame, k: int, top: bool) -> pd.DataFrame:
    s = score.where(eligible & np.isfinite(score))
    r = s.rank(axis=1, ascending=not top, method="first")
    return r <= k


def daily_weights(panel, members: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    px = panel["PriceUSD"]
    score = past_return(px, cfg["L"], cfg["skip"])
    valid = members & np.isfinite(score)
    n_valid = valid.sum(axis=1)
    hyp, k = cfg["hyp"], cfg["k"]
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    if hyp in ("long_mom", "long_rev"):
        sel = rank_select(score, valid, k, top=(hyp == "long_mom"))
        cnt = sel.sum(axis=1).replace(0, np.nan)
        w = sel.astype(float).div(cnt, axis=0).fillna(0.0)
    elif hyp in ("ls_mom", "ls_rev"):
        top = rank_select(score, valid, k, top=True)
        bot = rank_select(score, valid, k, top=False)
        enough = (n_valid >= 2 * k).astype(float)
        longs, shorts = (top, bot) if hyp == "ls_mom" else (bot, top)
        w = (0.5 / k) * longs.astype(float) - (0.5 / k) * shorts.astype(float)
        w = w.mul(enough, axis=0)
    elif hyp == "rel_btc":
        beat = valid & score.gt(score["btc"], axis=0)
        beat["btc"] = False
        btc_ok = valid["btc"].to_numpy()
        if k == "all":
            cnt = beat.sum(axis=1)
            w = beat.astype(float).div(cnt.replace(0, np.nan), axis=0).fillna(0.0)
            w["btc"] = np.where(btc_ok & (cnt.to_numpy() == 0), 1.0, 0.0)
        else:
            sel = rank_select(score, beat, k, top=True)
            cnt = sel.sum(axis=1).to_numpy()
            w = sel.astype(float) / k
            w["btc"] = np.where(btc_ok, (k - cnt) / k, 0.0)
    else:
        raise ValueError(hyp)
    return w


def strategy(panel, members, cfg) -> pd.DataFrame:
    """Weekly target weights from START on (members may be the full or the trimmed universe)."""
    return rebalance_on(daily_weights(panel, members, cfg).loc[START:], FREQ)


# ---- helpers -----------------------------------------------------------------------------------

def clean(d: dict) -> dict:
    return {k: (float(v) if isinstance(v, (np.floating, float, np.integer)) and not isinstance(v, bool)
                else v) for k, v in d.items()}


def yearly_returns(ret: pd.Series, years=TEST_YEARS) -> dict:
    out = {}
    for y in years:
        r = ret.loc[str(y)]
        out[str(y)] = float(100 * ((1 + r).prod() - 1)) if len(r) else None
    return out


def beta_alpha(ret: pd.Series, bench: pd.Series, start, end) -> dict:
    r, b = ret.loc[start:end].to_numpy(), bench.loc[start:end].to_numpy()
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1]
    resid = r - beta * b
    alpha_ann = resid.mean() * 365
    ir = resid.mean() / resid.std() * np.sqrt(365) if resid.std() > 0 else 0.0
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return {"beta": float(beta), "alpha_ann_pct": float(100 * alpha_ann),
            "resid_sharpe": float(ir), "corr": float(corr)}


def percentile(real: float, draws) -> float:
    draws = np.asarray(draws)
    return float(100 * np.mean(draws < real))


def permute_alts(weights: pd.DataFrame, members: pd.DataFrame, rng: np.random.Generator,
                 keep: str = "btc") -> pd.DataFrame:
    """Stricter selection placebo for rel_btc: BTC keeps its own weight on every rebalance row,
    only the altcoin weights go to random non-BTC universe members. permute_assets would also
    move the BTC fallback weight to a random coin, which in a BTC-dominated TEST period makes
    any BTC tilt look like selection skill."""
    j = weights.columns.get_loc(keep)
    m = members.reindex_like(weights).fillna(False).to_numpy().copy()
    m[:, j] = False
    w = weights.to_numpy()
    res = w.copy()
    for t in np.flatnonzero(~np.isnan(w).all(axis=1)):
        vals = np.nan_to_num(w[t])
        alt = vals.copy()
        alt[j] = 0.0
        nz = alt[alt != 0]
        pool = np.flatnonzero(m[t])
        row = np.zeros(len(vals))
        row[j] = vals[j]
        if len(nz) and len(pool) >= len(nz):
            row[rng.choice(pool, len(nz), replace=False)] = rng.permutation(nz)
        res[t] = row
    return pd.DataFrame(res, index=weights.index, columns=weights.columns)


def placebo(weights, members, px, n, rng, kind="permute"):
    test_s, train_s = [], []
    for _ in range(n):
        if kind == "permute":
            pw = permute_assets(weights, members, rng)
        elif kind == "permute_alts":
            pw = permute_alts(weights, members, rng)
        else:
            pw = circular_shift(weights, px, rng)
        m = backtest(pw, px).split_metrics()
        test_s.append(m["test"]["sharpe"])
        train_s.append(m["train"]["sharpe"])
    return np.array(train_s), np.array(test_s)


def check_no_lookahead(panel, cfgs, cut="2023-06-18"):
    """Weights up to `cut` must not change when every field is truncated at `cut`."""
    cut = pd.Timestamp(cut)
    trunc = {k: v.loc[:cut] for k, v in panel.items()}
    for cfg in cfgs:
        full = daily_weights(panel, universe(panel, top_n=cfg["top_n"]), cfg).loc[:cut]
        part = daily_weights(trunc, universe(trunc, top_n=cfg["top_n"]), cfg)
        pd.testing.assert_frame_equal(full, part, check_freq=False)


def contributions(res, px, start, end, top=5):
    rets = px.pct_change(fill_method=None).fillna(0.0)
    contrib = (res.held * rets).loc[start:end].sum()
    contrib = contrib[contrib != 0].sort_values()
    return {"best": {k: round(float(100 * v), 2) for k, v in contrib.iloc[::-1].head(top).items()},
            "worst": {k: round(float(100 * v), 2) for k, v in contrib.head(top).items()}}


def robust_checks(test_sharpe, pct, cost30, years, med_family, med_grid):
    pos_years = sum(1 for v in years.values() if v is not None and v > 0)
    checks = {
        "test_sharpe_gt_btc": bool(test_sharpe > BTC_TEST_SHARPE),
        "placebo_pct_ge_95": bool(pct >= 95),
        "cost30_sharpe_gt_btc": bool(cost30 > BTC_TEST_SHARPE),
        "positive_years_ge_3": bool(pos_years >= 3),
        "family_grid_median_gt_0": bool(med_family > 0),
        "full_grid_median_gt_0": bool(med_grid > 0),
    }
    return checks, pos_years


def deep_dive(cfg, panel, members, px, rng, n_perm, n_timing, ew_rets, btc_rets, grid_df):
    w = strategy(panel, members, cfg)
    res = backtest(w, px)
    sm = res.split_metrics()
    t0 = time.time()
    p_train, p_test = placebo(w, members, px, n_perm, rng, "permute")
    out = {
        "config": cfg,
        "train": clean(sm["train"]), "test": clean(sm["test"]), "full": clean(sm["full"]),
        "placebo_permute": {
            "n": n_perm,
            "test_percentile": percentile(sm["test"]["sharpe"], p_test),
            "train_percentile": percentile(sm["train"]["sharpe"], p_train),
            "test_placebo_median": float(np.median(p_test)),
            "test_placebo_p95": float(np.quantile(p_test, 0.95)),
            "train_placebo_median": float(np.median(p_train)),
        },
    }
    if n_timing:
        c_train, c_test = placebo(w, members, px, n_timing, rng, "circular")
        out["placebo_circular_shift"] = {
            "n": n_timing,
            "test_percentile": percentile(sm["test"]["sharpe"], c_test),
            "train_percentile": percentile(sm["train"]["sharpe"], c_train),
            "test_placebo_median": float(np.median(c_test)),
            "test_placebo_p95": float(np.quantile(c_test, 0.95)),
        }
    if n_timing and cfg["hyp"] == "rel_btc":
        a_train, a_test = placebo(w, members, px, n_perm, rng, "permute_alts")
        out["placebo_permute_alts_keep_btc"] = {
            "n": n_perm,
            "test_percentile": percentile(sm["test"]["sharpe"], a_test),
            "train_percentile": percentile(sm["train"]["sharpe"], a_train),
            "test_placebo_median": float(np.median(a_test)),
            "test_placebo_p95": float(np.quantile(a_test, 0.95)),
        }
    out["placebo_seconds"] = round(time.time() - t0, 1)
    rows = w.dropna(how="all")
    for name, (a, b) in (("train", TRAIN), ("test", TEST)):
        r = rows.loc[a:b]
        out[f"positions_{name}"] = {
            "avg_n_long": float((r > 0).sum(axis=1).mean()),
            "avg_n_short": float((r < 0).sum(axis=1).mean()),
            "avg_btc_weight": float(r["btc"].mean()),
            "share_rows_all_btc": float((r["btc"] >= 0.999).mean()),
        }
    out["test_sharpe_cost30"] = float(backtest(w, px, cost_bps=30).metrics(*TEST)["sharpe"])
    out["test_sharpe_lag0"] = float(backtest(w, px, lag=0).metrics(*TEST)["sharpe"])
    out["test_sharpe_lag2"] = float(backtest(w, px, lag=2).metrics(*TEST)["sharpe"])
    out["test_yearly_returns_pct"] = yearly_returns(res.returns)
    out["train_yearly_returns_pct"] = yearly_returns(res.returns, (2018, 2019, 2020, 2021))
    out["vs_matched_ew_test"] = beta_alpha(res.returns, ew_rets[cfg["top_n"]], *TEST)
    out["vs_btc_test"] = beta_alpha(res.returns, btc_rets, *TEST)
    out["contrib_train_pct"] = contributions(res, px, *TRAIN)
    out["contrib_test_pct"] = contributions(res, px, *TEST)
    fam = grid_df[grid_df.hyp == cfg["hyp"]]
    checks, pos_years = robust_checks(sm["test"]["sharpe"], out["placebo_permute"]["test_percentile"],
                                      out["test_sharpe_cost30"], out["test_yearly_returns_pct"],
                                      fam.test_sharpe.median(), grid_df.test_sharpe.median())
    out["positive_test_years"] = pos_years
    out["robust_checks"] = checks
    out["robust_candidate"] = bool(all(checks.values()))
    return out


def main():
    t_start = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    px = panel["PriceUSD"].loc[START:]   # backtest window; signals use the full panel
    members = {n: universe(panel, top_n=n) for n in TOP_NS}
    grid = build_grid()

    # Sanity: no lookahead in signal construction (one config per hypothesis/variant).
    probe = [c for c in grid if c["top_n"] == 20 and c["k"] in (3, "all") and c["L"] in (7, 30)]
    check_no_lookahead(panel, probe)
    print(f"lookahead check passed on {len(probe)} configs")

    # Benchmarks (same harness, default costs, lag=1).
    bench = {"btc_buy_hold": backtest(buy_and_hold(px, "btc"), px)}
    for n in TOP_NS:
        bench[f"ew_top{n}_weekly"] = backtest(equal_weight(members[n], FREQ), px)
    ew_rets = {n: bench[f"ew_top{n}_weekly"].returns for n in TOP_NS}
    btc_rets = bench["btc_buy_hold"].returns
    bench_out = {k: {s: clean(v) for s, v in r.split_metrics().items()} for k, r in bench.items()}
    for k, v in bench_out.items():
        print(f"{k:18s} train SR {v['train']['sharpe']:.2f}  test SR {v['test']['sharpe']:.2f}"
              f"  test tot {v['test']['total_pct']:.0f}%")

    # Full grid.
    rows = []
    for i, cfg in enumerate(grid):
        w = strategy(panel, members[cfg["top_n"]], cfg)
        res = backtest(w, px)
        sm = res.split_metrics()
        row = dict(cfg)
        for split in ("train", "test", "full"):
            for key in ("sharpe", "cagr_pct", "total_pct", "max_dd_pct", "vol_pct",
                        "turnover_per_year", "exposure"):
                row[f"{split}_{key.replace('_pct', '')}" if key != "sharpe" else f"{split}_sharpe"] = \
                    float(sm[split][key])
        row["died"] = bool((res.returns <= -1).any())
        ba = beta_alpha(res.returns, ew_rets[cfg["top_n"]], *TEST)
        row["test_beta_ew"], row["test_alpha_ew_pct"] = ba["beta"], ba["alpha_ann_pct"]
        row["test_resid_sharpe_ew"] = ba["resid_sharpe"]
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i + 1}/{len(grid)} {cfg['id']}: train SR {row['train_sharpe']:.2f}"
                  f" test SR {row['test_sharpe']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grid.csv", index=False)
    n_configs = len(df)

    # Grid summary on TEST.
    def summarize(d):
        return {
            "n": int(len(d)),
            "median_train_sharpe": float(d.train_sharpe.median()),
            "median_test_sharpe": float(d.test_sharpe.median()),
            "mean_test_sharpe": float(d.test_sharpe.mean()),
            "n_test_sharpe_gt_btc_0.48": int((d.test_sharpe > BTC_TEST_SHARPE).sum()),
            "n_test_sharpe_gt_ew20_0.15": int((d.test_sharpe > EW20_TEST_SHARPE).sum()),
            "n_test_sharpe_gt_0": int((d.test_sharpe > 0).sum()),
            "n_died": int(d.died.sum()),
            "median_test_total_pct": float(d.test_total.median()),
            "spearman_train_test_sharpe": float(d.train_sharpe.rank().corr(d.test_sharpe.rank()))
            if len(d) > 2 else None,
            "best_test_sharpe": float(d.test_sharpe.max()),
            "best_test_id": d.loc[d.test_sharpe.idxmax(), "id"],
            "median_test_resid_sharpe_vs_ew": float(d.test_resid_sharpe_ew.median()),
        }
    summary = {"all": summarize(df)}
    for h, d in df.groupby("hyp"):
        summary[h] = summarize(d)
    for h in ("long_mom", "ls_mom", "rel_btc"):
        for s in SKIPS:
            summary[f"{h}_skip{s}"] = summarize(df[(df.hyp == h) & (df.skip == s)])
        for L in LOOKBACKS:
            summary[f"{h}_L{L}"] = summarize(df[(df.hyp == h) & (df.L == L)])
    for n in TOP_NS:
        summary[f"top{n}"] = summarize(df[df.top_n == n])
    # Long-only momentum vs the equal-weight basket of the same universe.
    lm = df[df.hyp == "long_mom"].copy()
    lm["ew_test_sharpe"] = lm.top_n.map(lambda n: bench_out[f"ew_top{n}_weekly"]["test"]["sharpe"])
    summary["long_mom_vs_matched_ew"] = {
        "n_beat_matched_ew_test_sharpe": int((lm.test_sharpe > lm.ew_test_sharpe).sum()),
        "n": int(len(lm)),
        "median_test_sharpe_minus_ew": float((lm.test_sharpe - lm.ew_test_sharpe).median()),
        "median_test_beta_vs_ew": float(lm.test_beta_ew.median()),
        "median_test_alpha_vs_ew_pct": float(lm.test_alpha_ew_pct.median()),
    }

    # TRAIN-only selection.
    best_row = df.loc[df.train_sharpe.idxmax()]
    best_cfg = {k: best_row[k] for k in ("id", "hyp", "top_n", "L", "skip", "k")}
    best_cfg = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_cfg.items()}
    print(f"\nTRAIN-chosen: {best_cfg['id']} train SR {best_row.train_sharpe:.2f}"
          f" test SR {best_row.test_sharpe:.2f}")

    rng = np.random.default_rng(SEED)
    chosen = deep_dive(best_cfg, panel, members[best_cfg["top_n"]], px, rng, N_PLACEBO_MAIN,
                       N_PLACEBO_TIMING, ew_rets, btc_rets, df)
    print(json.dumps({k: chosen[k] for k in ("test", "placebo_permute", "test_sharpe_cost30",
                                             "test_sharpe_lag0", "test_yearly_returns_pct",
                                             "robust_checks")}, indent=1, default=str))

    # Supplementary: TRAIN-best config within each hypothesis (still TRAIN-only selection).
    per_hyp = {}
    for h, d in df.groupby("hyp"):
        r = d.loc[d.train_sharpe.idxmax()]
        cfg = {k: r[k] for k in ("id", "hyp", "top_n", "L", "skip", "k")}
        cfg = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in cfg.items()}
        if cfg["id"] == best_cfg["id"]:
            per_hyp[h] = {"same_as_chosen": True, "id": cfg["id"]}
            continue
        hrng = np.random.default_rng([SEED, len(per_hyp) + 1])
        per_hyp[h] = deep_dive(cfg, panel, members[cfg["top_n"]], px, hrng, N_PLACEBO_FAMILY, 0,
                               ew_rets, btc_rets, df)
        print(f"  best-in-{h}: {cfg['id']} test SR {per_hyp[h]['test']['sharpe']:.2f}"
              f" placebo pct {per_hyp[h]['placebo_permute']['test_percentile']:.0f}"
              f" robust {per_hyp[h]['robust_candidate']}")

    results = {
        "family": "cross-sectional momentum and reversal across coins",
        "protocol": {
            "train": TRAIN, "test": TEST, "rebalance": FREQ, "cost_bps": 10, "lag": 1,
            "selection": "max TRAIN Sharpe over the full grid", "seed": SEED,
            "robust_candidate_rule": "test SR > 0.48, permute placebo pct >= 95, cost30 test SR > 0.48, "
                                     ">= 3 positive TEST calendar years, median TEST SR > 0 in both the "
                                     "chosen hypothesis' sub-grid and the full grid",
            "grid": {"top_n": TOP_NS, "lookback": LOOKBACKS, "skip": SKIPS, "k": KS,
                     "reversal_lookback": REV_LOOKBACKS, "rel_btc_k": RELBTC_KS},
        },
        "n_configs_tested": n_configs,
        "benchmarks": bench_out,
        "grid_summary": summary,
        "chosen": chosen,
        "best_per_hypothesis": per_hyp,
        "grid": df.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(OUT / "results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nwrote {OUT / 'results.json'} ({n_configs} configs, {results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
