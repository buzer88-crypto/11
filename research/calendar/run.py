"""Calendar effects on daily closes: BTC, ETH and the point-in-time top-20 equal-weight basket.

Run:  cd /home/user/11 && python3 research/calendar/run.py

Hypotheses (one "slice" = a set of UTC calendar days; the strategy is long the target on the
days inside the slice and in cash otherwise):
  dow        hold only on weekday d (side "in") / every day except weekday d (side "out")
  weekend    hold only Sat+Sun (in) / only Mon-Fri (out)
  tom        turn of month: last L days of a month + first F days of the next (in) / the rest (out)
  moy        hold only in month m (in) / every month except m (out)
  halloween  hold Nov-Apr (in) / May-Oct (out)
  dow_adapt  hold on the k weekdays with the best mean return so far (expanding or 730-day
             window, data <= t only), re-estimated every day
  moy_adapt  same with calendar months (expanding or 1461-day window)

Timing.  Coin Metrics PriceUSD for date D is the reference rate at 00:00 UTC of D+1 (end of the
UTC day D), so the return labelled D covers exactly the UTC calendar day D.  A calendar is known
in advance, so a rule needs no data to know it wants to be long on, say, Monday.  The harness
trades weights row t at close t+lag and the position then earns return day t+lag+1, so row t
holds the slice indicator of day t+lag+1 (calendar arithmetic, not a data shift).  An audit
checks that the positions the harness actually held line up with the intended slice days.

The grid is fixed up front and run in full.  The chosen config is the best TRAIN Sharpe
(2018-2021); TEST (2022-) is only reported.  Effects are also tested directly with a Welch
t-test of mean daily return inside vs outside each slice, per period, with Bonferroni / Holm /
Benjamini-Hochberg corrections over all distinct slices examined.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for `pinetest`

from pinetest.crypto_data import load_panel, universe
from pinetest.portfolio import TEST, TRAIN, backtest, circular_shift

OUT_DIR = Path(__file__).resolve().parent
TMP_DIR = Path("/tmp/claude-0/research-calendar")
ROOT = Path(__file__).resolve().parents[2]

SEED = 20261014
LAG = 1
COST = 10.0
TARGETS = ("btc", "eth", "top20")
DOW_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
TOM_LAST = (1, 2, 3)            # last L days of the month
TOM_FIRST = (1, 2, 3, 4)        # first F days of the next month
DOW_ADAPT_K = (1, 2, 3, 4, 5, 6)
DOW_ADAPT_WIN = (None, 730)     # None = expanding
MOY_ADAPT_K = (3, 6, 9)
MOY_ADAPT_WIN = (None, 1461)
DOW_MIN_OBS = 52                # one year of each weekday before an adaptive rule may trade
MOY_MIN_OBS = 56                # about two instances of each month
PRE = ("2014-01-01", "2017-12-31")   # extra, untouched pre-TRAIN period (information only)
PERIODS = {"train": TRAIN, "test": TEST, "pre": PRE}
TEST_YEARS = (2022, 2023, 2024, 2025, 2026)
N_PLACEBO_CHOSEN = 500
N_PLACEBO_INFO = 200
AUDIT_CUT = "2021-06-15"
ALPHA = 0.05
BH_Q = 0.10


# --------------------------------------------------------------------------------------------
# Statistics helpers (no scipy available)
# --------------------------------------------------------------------------------------------

def _betacf(a, b, x, itmax=500, eps=3e-15):
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > fpmin else fpmin)
        c = 1.0 + aa / c
        c = c if abs(c) > fpmin else fpmin
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def t_pvalue_2s(t, df):
    if not np.isfinite(t) or not np.isfinite(df) or df <= 0:
        return float("nan")
    return float(betainc(df / 2.0, 0.5, df / (df + t * t)))


def welch(inside: np.ndarray, outside: np.ndarray) -> dict:
    """Welch t-test of mean daily return inside vs outside a slice."""
    a, b = inside[np.isfinite(inside)], outside[np.isfinite(outside)]
    n1, n2 = len(a), len(b)
    out = {"n_in": n1, "n_out": n2}
    if n1 < 3 or n2 < 3:
        return {**out, "mean_in_bps": None, "mean_out_bps": None, "t": None, "df": None, "p": None}
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = math.sqrt(v1 + v2)
    t = (m1 - m2) / se if se > 0 else float("nan")
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return {**out, "mean_in_bps": 1e4 * m1, "mean_out_bps": 1e4 * m2, "t": t, "df": df,
            "p": t_pvalue_2s(t, df)}


def holm(p: np.ndarray, alpha=ALPHA) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    rej = np.zeros(m, bool)
    for i, idx in enumerate(order):
        if p[idx] <= alpha / (m - i):
            rej[idx] = True
        else:
            break
    return rej


def bh(p: np.ndarray, q=BH_Q) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    thr = q * np.arange(1, m + 1) / m
    passed = p[order] <= thr
    rej = np.zeros(m, bool)
    if passed.any():
        kmax = np.flatnonzero(passed).max()
        rej[order[:kmax + 1]] = True
    return rej


def spearman(x, y) -> float:
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    return float(s["x"].rank().corr(s["y"].rank()))


def rnd(d: dict, nd=4) -> dict:
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}


def metric_subset(m: dict) -> dict:
    keys = ("sharpe", "cagr_pct", "total_pct", "max_dd_pct", "vol_pct", "exposure", "turnover_per_year")
    return rnd({k: m[k] for k in keys if k in m})


def yearly_returns(r: pd.Series, years=TEST_YEARS) -> dict:
    return {str(y): round(100 * float((1 + r.loc[str(y)]).prod() - 1), 2) for y in years}


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------

def build_grid():
    grid = []
    for tg in TARGETS:
        for side in ("in", "out"):
            for d in range(7):
                grid.append(dict(target=tg, family="dow", side=side, day=d))
        for side in ("in", "out"):
            grid.append(dict(target=tg, family="weekend", side=side))
        for side in ("in", "out"):
            for L in TOM_LAST:
                for F in TOM_FIRST:
                    grid.append(dict(target=tg, family="tom", side=side, last=L, first=F))
        for side in ("in", "out"):
            for m in range(1, 13):
                grid.append(dict(target=tg, family="moy", side=side, month=m))
        for side in ("in", "out"):
            grid.append(dict(target=tg, family="halloween", side=side))
        for win in DOW_ADAPT_WIN:
            for k in DOW_ADAPT_K:
                grid.append(dict(target=tg, family="dow_adapt", side="in", k=k, window=win))
        for win in MOY_ADAPT_WIN:
            for k in MOY_ADAPT_K:
                grid.append(dict(target=tg, family="moy_adapt", side="in", k=k, window=win))
    for i, g in enumerate(grid):
        g["id"] = i
        g["name"], g["test_id"], g["description"] = describe(g)
    return grid


def describe(g):
    tg, fam, side = g["target"], g["family"], g["side"]
    what = {"btc": "BTC", "eth": "ETH", "top20": "the top-20 equal-weight basket"}[tg]
    if fam == "dow":
        dn = DOW_NAMES[g["day"]]
        key = f"{tg}|dow|{dn}"
        desc = (f"Hold {what} only on UTC {dn}days" if side == "in"
                else f"Hold {what} every UTC day except {dn}day")
    elif fam == "weekend":
        key = f"{tg}|weekend"
        desc = (f"Hold {what} only on UTC Saturday+Sunday" if side == "in"
                else f"Hold {what} only on UTC Monday-Friday")
    elif fam == "tom":
        key = f"{tg}|tom|L{g['last']}F{g['first']}"
        win = f"the last {g['last']} and first {g['first']} days of each month"
        desc = f"Hold {what} only on {win}" if side == "in" else f"Hold {what} except on {win}"
    elif fam == "moy":
        mn = MONTH_NAMES[g["month"] - 1]
        key = f"{tg}|moy|{mn}"
        desc = f"Hold {what} only in {mn}" if side == "in" else f"Hold {what} in every month except {mn}"
    elif fam == "halloween":
        key = f"{tg}|halloween"
        desc = f"Hold {what} Nov-Apr only" if side == "in" else f"Hold {what} May-Oct only"
    elif fam == "dow_adapt":
        w = "expanding" if g["window"] is None else f"{g['window']}d"
        key = f"{tg}|dow_adapt|k{g['k']}|{w}"
        desc = (f"Hold {what} on the {g['k']} UTC weekdays with the highest mean return so far "
                f"({w} window, data <= t, re-estimated daily)")
    elif fam == "moy_adapt":
        w = "expanding" if g["window"] is None else f"{g['window']}d"
        key = f"{tg}|moy_adapt|k{g['k']}|{w}"
        desc = (f"Hold {what} in the {g['k']} calendar months with the highest mean daily return "
                f"so far ({w} window, data <= t, re-estimated daily)")
    else:
        raise ValueError(fam)
    name = f"{key}|{side}" if fam not in ("dow_adapt", "moy_adapt") else key
    return name, key, desc + "; cash otherwise."


def fixed_slice(g, days: pd.DatetimeIndex) -> np.ndarray:
    """True if return day `days` (UTC calendar day) is inside the slice."""
    fam = g["family"]
    if fam == "dow":
        m = days.dayofweek == g["day"]
    elif fam == "weekend":
        m = days.dayofweek >= 5
    elif fam == "tom":
        m = (days.day <= g["first"]) | (days.day > days.days_in_month - g["last"])
    elif fam == "moy":
        m = days.month == g["month"]
    elif fam == "halloween":
        m = days.month.isin([11, 12, 1, 2, 3, 4])
    else:
        raise ValueError(fam)
    m = np.asarray(m, bool)
    return m if g["side"] == "in" else ~m


class Context:
    """Everything derived from one panel; all of it is causal (data <= t at row t)."""

    def __init__(self, panel):
        self.px = panel["PriceUSD"]
        self.dates = self.px.index
        self.members = universe(panel, top_n=20, min_history=180, min_volume=1e6)
        self.base = {}
        for a in ("btc", "eth"):
            w = pd.DataFrame(0.0, index=self.dates, columns=self.px.columns)
            w[a] = 1.0
            self.base[a] = w
        m = self.members.astype(float)
        self.base["top20"] = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        # Raw daily return of each target, labelled by return day D (= UTC day D).
        self.tret = {a: self.px[a].pct_change(fill_method=None) for a in ("btc", "eth")}
        bres = backtest(self.base["top20"], self.px, cost_bps=0.0, lag=0)
        self.tret["top20"] = bres.returns.where(bres.gross_exposure > 0)

    def rows(self, g, lag=LAG) -> pd.Series:
        """Bool by row date t: be exposed during return day t+lag+1."""
        ahead = self.dates + pd.Timedelta(days=lag + 1)
        if g["family"] in ("dow_adapt", "moy_adapt"):
            return pd.Series(self._adaptive(g, ahead), index=self.dates)
        return pd.Series(fixed_slice(g, ahead), index=self.dates)

    def _adaptive(self, g, ahead) -> np.ndarray:
        r = self.tret[g["target"]].to_numpy()
        n = len(r)
        if g["family"] == "dow_adapt":
            key_ret, key_row, nk, min_obs = self.dates.dayofweek, ahead.dayofweek, 7, DOW_MIN_OBS
        else:
            key_ret, key_row, nk, min_obs = self.dates.month - 1, ahead.month - 1, 12, MOY_MIN_OBS
        M = np.full((n, nk), np.nan)
        M[np.arange(n), np.asarray(key_ret)] = r
        M = pd.DataFrame(M, index=self.dates)
        if g["window"] is None:
            S, C = M.fillna(0.0).cumsum(), M.notna().cumsum()
        else:
            S = M.fillna(0.0).rolling(g["window"], min_periods=1).sum()
            C = M.notna().astype(float).rolling(g["window"], min_periods=1).sum()
        mean = S / C.replace(0, np.nan)
        valid = (C >= min_obs).all(axis=1).to_numpy()
        rank = mean.rank(axis=1, ascending=False, method="first").to_numpy()
        sel = rank[np.arange(n), np.asarray(key_row)] <= g["k"]
        return sel & valid

    def weights(self, g, lag=LAG) -> pd.DataFrame:
        """Strategy weights: target allocation when the slice is on, cash otherwise.
        Every row is set (daily), so entries and exits happen exactly on the slice edges."""
        on = self.rows(g, lag).astype(float)
        return self.base[g["target"]].mul(on, axis=0)

    def in_return_days(self, g, lag=LAG) -> pd.Series:
        """Which return days the rule is exposed on (row t -> day t+lag+1)."""
        return self.rows(g, lag).shift(lag + 1).fillna(False).astype(bool)


def strategy(panel, g, lag=LAG) -> pd.DataFrame:
    """panel -> weights for grid config g."""
    return Context(panel).weights(g, lag)


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

def ttests(ctx, g, lag=LAG) -> dict:
    r = ctx.tret[g["target"]]
    ins = ctx.in_return_days(g, lag)
    out = {}
    for pname, (s, e) in PERIODS.items():
        rr, ii = r.loc[s:e].to_numpy(), ins.loc[s:e].to_numpy()
        out[pname] = rnd(welch(rr[ii], rr[~ii]), 6)
    return out


def alignment_mismatch(ctx, g, res) -> int:
    """Days on which the held position disagrees with the intended slice (after warm-up)."""
    ins = ctx.in_return_days(g)
    if g["target"] == "top20":
        held_on = res.held.abs().sum(axis=1) > 1e-12
        avail = ctx.tret["top20"].notna()
    else:
        held_on = res.held[g["target"]].abs() > 1e-12
        avail = ctx.px[g["target"]].notna() & ctx.px[g["target"]].shift(1).notna()
    ok = avail & (ctx.dates >= pd.Timestamp("2014-01-01"))
    return int((held_on[ok] != ins[ok]).sum())


def evaluate(ctx, g):
    w = ctx.weights(g)
    res = backtest(w, ctx.px, cost_bps=COST, lag=LAG)
    res0 = backtest(w, ctx.px, cost_bps=0.0, lag=LAG)
    sm, sm0 = res.split_metrics(), res0.split_metrics()
    row = {k: v for k, v in g.items()}
    row.update({
        "train": metric_subset(sm["train"]), "test": metric_subset(sm["test"]),
        "full": metric_subset(sm["full"]), "pre": metric_subset(res.metrics(*PRE)),
        "train_sharpe_cost0": round(sm0["train"]["sharpe"], 4),
        "test_sharpe_cost0": round(sm0["test"]["sharpe"], 4),
        "ttest": ttests(ctx, g),
        "alignment_mismatch_days": alignment_mismatch(ctx, g, res),
    })
    return row, res


def placebo(ctx, w, n, rng):
    tr, te = np.empty(n), np.empty(n)
    for i in range(n):
        pw = circular_shift(w, ctx.px, rng)
        sm = backtest(pw, ctx.px, cost_bps=COST, lag=LAG).split_metrics()
        tr[i], te[i] = sm["train"]["sharpe"], sm["test"]["sharpe"]
    return tr, te


def placebo_summary(real_train, real_test, tr, te):
    return {
        "n": int(len(te)), "type": "circular_shift (timing)",
        "test_percentile": round(100 * float((te < real_test).mean()), 2),
        "test_percentile_ties_half": round(100 * float((te < real_test).mean() + 0.5 * (te == real_test).mean()), 2),
        "train_percentile": round(100 * float((tr < real_train).mean()), 2),
        "placebo_test_sharpe_median": round(float(np.median(te)), 4),
        "placebo_test_sharpe_p95": round(float(np.percentile(te, 95)), 4),
        "placebo_train_sharpe_median": round(float(np.median(tr)), 4),
        "placebo_train_sharpe_p95": round(float(np.percentile(tr, 95)), 4),
    }


def summarize(rows, btc_test):
    df = pd.DataFrame({"tr": [r["train"]["sharpe"] for r in rows],
                       "te": [r["test"]["sharpe"] for r in rows],
                       "te0": [r["test_sharpe_cost0"] for r in rows]})
    return {
        "n": len(rows),
        "median_test_sharpe": round(float(df["te"].median()), 4),
        "mean_test_sharpe": round(float(df["te"].mean()), 4),
        "n_beat_btc_test": int((df["te"] > btc_test).sum()),
        "median_test_sharpe_cost0": round(float(df["te0"].median()), 4),
        "n_beat_btc_test_cost0": int((df["te0"] > btc_test).sum()),
        "median_train_sharpe": round(float(df["tr"].median()), 4),
        "spearman_train_test": round(spearman(df["tr"], df["te"]), 4),
    }


def multiple_testing(rows):
    """One test per distinct partition (an 'in' rule and its 'out' complement share a test)."""
    tests = {}
    for r in rows:
        if r["test_id"] in tests:
            continue
        tests[r["test_id"]] = {"test_id": r["test_id"], "target": r["target"], "family": r["family"],
                               "side_reported": r["side"],
                               **{p: r["ttest"][p] for p in PERIODS}}
    T = list(tests.values())
    m = len(T)
    out = {"n_distinct_slices": m, "n_configs": len(rows),
           "bonferroni_alpha": ALPHA / m, "bh_q": BH_Q}
    for pname in ("train", "test"):
        p = np.array([t[pname]["p"] if t[pname]["p"] is not None else 1.0 for t in T])
        bon, hol, fdr = p <= ALPHA / m, holm(p), bh(p)
        out[pname] = {
            "n_raw_p_lt_0.05": int((p < ALPHA).sum()),
            "expected_false_positives_at_0.05": round(ALPHA * m, 1),
            "n_bonferroni": int(bon.sum()), "n_holm": int(hol.sum()), "n_bh": int(fdr.sum()),
            "bonferroni_survivors": [dict(test_id=T[i]["test_id"], t=T[i][pname]["t"], p=T[i][pname]["p"],
                                          mean_in_bps=T[i][pname]["mean_in_bps"],
                                          mean_out_bps=T[i][pname]["mean_out_bps"])
                                     for i in np.flatnonzero(bon)],
            "bh_survivors": [T[i]["test_id"] for i in np.flatnonzero(fdr)],
            "min_p": float(p.min()),
            "min_p_slice": T[int(p.argmin())]["test_id"],
        }
    # Replication: do TRAIN-nominal effects keep sign and significance in TEST?
    ttr = np.array([t["train"]["t"] if t["train"]["t"] is not None else np.nan for t in T])
    tte = np.array([t["test"]["t"] if t["test"]["t"] is not None else np.nan for t in T])
    ptr = np.array([t["train"]["p"] if t["train"]["p"] is not None else 1.0 for t in T])
    pte = np.array([t["test"]["p"] if t["test"]["p"] is not None else 1.0 for t in T])
    sig_tr = ptr < ALPHA
    out["replication"] = {
        "corr_train_t_vs_test_t": round(float(pd.Series(ttr).corr(pd.Series(tte))), 4),
        "n_train_nominal": int(sig_tr.sum()),
        "n_train_nominal_same_sign_in_test": int((sig_tr & (np.sign(ttr) == np.sign(tte))).sum()),
        "n_train_nominal_same_sign_and_p_lt_0.05_in_test":
            int((sig_tr & (np.sign(ttr) == np.sign(tte)) & (pte < ALPHA)).sum()),
        "share_same_sign_all_slices": round(float((np.sign(ttr) == np.sign(tte)).mean()), 4),
    }
    top = sorted(T, key=lambda t: -abs(t["train"]["t"] or 0))[:12]
    out["top_train_effects"] = [
        dict(test_id=t["test_id"], side=t["side_reported"],
             train_t=t["train"]["t"], train_p=t["train"]["p"], train_p_bonferroni=min(1.0, t["train"]["p"] * m),
             train_in_bps=t["train"]["mean_in_bps"], train_out_bps=t["train"]["mean_out_bps"],
             test_t=t["test"]["t"], test_p=t["test"]["p"],
             test_in_bps=t["test"]["mean_in_bps"], test_out_bps=t["test"]["mean_out_bps"],
             pre_t=t["pre"]["t"], pre_p=t["pre"]["p"])
        for t in top]
    return out, T


def effect_tables(ctx):
    """Mean daily return (bps) and t-stat (slice vs rest) per weekday / TOM day / month."""
    out = {}
    for tg in TARGETS:
        r = ctx.tret[tg]
        tab = {}
        for pname, (s, e) in PERIODS.items():
            rr = r.loc[s:e].dropna()
            dd = rr.index
            dow = {}
            for d in range(7):
                m = np.asarray(dd.dayofweek == d)
                w = welch(rr.to_numpy()[m], rr.to_numpy()[~m])
                dow[DOW_NAMES[d]] = {"bps": round(w["mean_in_bps"], 1), "t": round(w["t"], 2)}
            moy = {}
            for mo in range(1, 13):
                m = np.asarray(dd.month == mo)
                w = welch(rr.to_numpy()[m], rr.to_numpy()[~m])
                moy[MONTH_NAMES[mo - 1]] = {"bps": round(w["mean_in_bps"], 1), "t": round(w["t"], 2)}
            tom_m = np.asarray((dd.day <= 3) | (dd.day > dd.days_in_month - 2))
            wt = welch(rr.to_numpy()[tom_m], rr.to_numpy()[~tom_m])
            wk_m = np.asarray(dd.dayofweek >= 5)
            ww = welch(rr.to_numpy()[wk_m], rr.to_numpy()[~wk_m])
            tab[pname] = {"dow": dow, "moy": moy,
                          "tom_L2F3": rnd({k: wt[k] for k in ("mean_in_bps", "mean_out_bps", "t", "p")}, 3),
                          "weekend": rnd({k: ww[k] for k in ("mean_in_bps", "mean_out_bps", "t", "p")}, 3),
                          "all_days_bps": round(1e4 * float(rr.mean()), 1),
                          "daily_vol_bps": round(1e4 * float(rr.std()), 1)}
        out[tg] = tab
    return out


def best_per_family(rows, grid, ctx, results, rng, chosen_id, chosen_placebo):
    """TRAIN-best config inside each family (information only; not a second selection)."""
    out = {}
    for fam in ("dow", "weekend", "tom", "moy", "halloween", "dow_adapt", "moy_adapt"):
        br = max((r for r in rows if r["family"] == fam), key=lambda r: r["train"]["sharpe"])
        bg = next(x for x in grid if x["id"] == br["id"])
        if bg["id"] == chosen_id:
            ps = chosen_placebo
        else:
            ptr, pte = placebo(ctx, ctx.weights(bg), N_PLACEBO_INFO, rng)
            ps = placebo_summary(br["train"]["sharpe"], br["test"]["sharpe"], ptr, pte)
        out[fam] = {"name": br["name"], "train_sharpe": br["train"]["sharpe"],
                    "test_sharpe": br["test"]["sharpe"], "test_sharpe_cost0": br["test_sharpe_cost0"],
                    "test_sharpe_cost30": round(backtest(ctx.weights(bg), ctx.px, cost_bps=30.0)
                                                .split_metrics()["test"]["sharpe"], 4),
                    "test_total_pct": br["test"]["total_pct"], "test_max_dd_pct": br["test"]["max_dd_pct"],
                    "test_years_pct": yearly_returns(results[bg["id"]].returns),
                    "train_t": br["ttest"]["train"]["t"], "test_t": br["ttest"]["test"]["t"],
                    "test_p": br["ttest"]["test"]["p"], "placebo_test_percentile": ps["test_percentile"],
                    "placebo_n": ps["n"]}
    return out


def thursday_post_hoc(ctx):
    """POST HOC: UTC Thursdays were weak for all three targets in both TRAIN and TEST. This was
    noticed after looking at TEST, so it is reported for information only and cannot be a
    candidate. Thursday vs the other days, per period."""
    periods = {**PERIODS, "2018_to_end": (TRAIN[0], None)}
    out = {}
    for tg in TARGETS:
        r = ctx.tret[tg]
        out[tg] = {}
        for pname, (s, e) in periods.items():
            rr = r.loc[s:e].dropna()
            m = np.asarray(rr.index.dayofweek == 3)
            w = welch(rr.to_numpy()[m], rr.to_numpy()[~m])
            out[tg][pname] = rnd({k: w[k] for k in ("mean_in_bps", "mean_out_bps", "t", "p", "n_in")}, 4)
    return out


def lookahead_audit(panel, grid, ctx_full):
    cut = pd.Timestamp(AUDIT_CUT)
    trunc = {k: v.loc[:cut] for k, v in panel.items()}
    ctx_cut = Context(trunc)
    worst = 0.0
    for g in grid:
        a = ctx_full.weights(g).loc[:cut]
        b = ctx_cut.weights(g)
        worst = max(worst, float((a - b).abs().to_numpy().max()))
    return {"cut": AUDIT_CUT, "configs_checked": len(grid), "max_abs_diff": worst}


def self_test_stats():
    # Known values: t=2.0, df=10 -> two-sided p = 0.07339; t=1.959964, df=1e7 -> 0.05.
    p1, p2 = t_pvalue_2s(2.0, 10), t_pvalue_2s(1.959964, 1e7)
    assert abs(p1 - 0.0733880) < 1e-5, p1
    assert abs(p2 - 0.05) < 1e-5, p2


# --------------------------------------------------------------------------------------------

def main():
    t0 = time.time()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    self_test_stats()
    rng = np.random.default_rng(SEED)
    panel = load_panel()
    ctx = Context(panel)
    bench_file = json.loads((ROOT / "data" / "benchmarks.json").read_text())
    btc_test = round([v for k, v in bench_file.items() if k.startswith("BTC")][0]["test"]["sharpe"], 2)

    # Reference runs: the targets held every day.
    benchmarks = {}
    for tg in TARGETS:
        res = backtest(ctx.base[tg], ctx.px, cost_bps=COST, lag=LAG)
        sm = res.split_metrics()
        benchmarks[f"{tg}_always_in"] = {"train": metric_subset(sm["train"]), "test": metric_subset(sm["test"]),
                                         "pre": metric_subset(res.metrics(*PRE)),
                                         "test_years": yearly_returns(res.returns)}

    grid = build_grid()
    print(f"grid: {len(grid)} configs")
    rows, results = [], {}
    for g in grid:
        row, res = evaluate(ctx, g)
        rows.append(row)
        results[g["id"]] = res
    print(f"grid done in {time.time() - t0:.0f}s")

    # ---- choose on TRAIN only
    chosen_row = max(rows, key=lambda r: r["train"]["sharpe"])
    g = next(x for x in grid if x["id"] == chosen_row["id"])
    w = ctx.weights(g)
    res = results[g["id"]]
    real_tr, real_te = res.split_metrics()["train"]["sharpe"], res.split_metrics()["test"]["sharpe"]

    tr, te = placebo(ctx, w, N_PLACEBO_CHOSEN, rng)
    np.savez(TMP_DIR / "placebo_chosen.npz", train=tr, test=te)
    plc = placebo_summary(real_tr, real_te, tr, te)

    s30 = backtest(w, ctx.px, cost_bps=30.0, lag=LAG).split_metrics()
    s0 = backtest(w, ctx.px, cost_bps=0.0, lag=LAG).split_metrics()
    lag0_literal = backtest(w, ctx.px, cost_bps=COST, lag=0).split_metrics()
    lag2_literal = backtest(w, ctx.px, cost_bps=COST, lag=2).split_metrics()
    lag0_aligned = backtest(ctx.weights(g, lag=0), ctx.px, cost_bps=COST, lag=0).split_metrics()
    years = yearly_returns(res.returns)
    mt, tests = multiple_testing(rows)
    n_pos_years = sum(v > 0 for v in years.values())
    grid_sum = {"all": summarize(rows, btc_test)}
    for tg in TARGETS:
        grid_sum[f"target={tg}"] = summarize([r for r in rows if r["target"] == tg], btc_test)
    for fam in ("dow", "weekend", "tom", "moy", "halloween", "dow_adapt", "moy_adapt"):
        grid_sum[f"family={fam}"] = summarize([r for r in rows if r["family"] == fam], btc_test)

    checks = {
        "test_sharpe_gt_btc": real_te > btc_test,
        "placebo_pct_ge_95": plc["test_percentile"] >= 95,
        "cost30_test_sharpe_gt_btc": s30["test"]["sharpe"] > btc_test,
        "positive_years_ge_3_of_5": n_pos_years >= 3,
        "grid_median_test_sharpe_gt_0": grid_sum["all"]["median_test_sharpe"] > 0,
    }
    chosen = {
        **{k: v for k, v in chosen_row.items()},
        "selection": "max TRAIN Sharpe over the whole grid",
        "cost30": {"train_sharpe": round(s30["train"]["sharpe"], 4), "test_sharpe": round(s30["test"]["sharpe"], 4)},
        "cost0": {"train_sharpe": round(s0["train"]["sharpe"], 4), "test_sharpe": round(s0["test"]["sharpe"], 4)},
        "lag0_same_weights_test_sharpe": round(lag0_literal["test"]["sharpe"], 4),
        "lag0_realigned_test_sharpe": round(lag0_aligned["test"]["sharpe"], 4),
        "lag2_same_weights_test_sharpe": round(lag2_literal["test"]["sharpe"], 4),
        "lag_note": ("Calendar rules need no information, so weights are built for the execution lag. "
                     "'same_weights' re-runs the lag=1 weights with another lag, i.e. one day early (lag 0) "
                     "or late (lag 2) relative to the intended slice; 'realigned' rebuilds the weights for "
                     "lag 0 and is identical exposure by construction."),
        "test_years_pct": years,
        "n_positive_test_years": n_pos_years,
        "train_p_bonferroni": (min(1.0, chosen_row["ttest"]["train"]["p"] * mt["n_distinct_slices"])
                               if chosen_row["ttest"]["train"]["p"] is not None else None),
        "placebo": {**plc, "test_sharpes": [round(x, 4) for x in te]},
        "robust_checks": checks,
        "robust_candidate": all(checks.values()),
    }
    print(f"chosen: {chosen_row['name']} train {real_tr:.2f} test {real_te:.2f} placebo {plc['test_percentile']}")

    # ---- information only: best TRAIN config per target, smaller placebo
    best_per_target = {}
    for tg in TARGETS:
        br = max((r for r in rows if r["target"] == tg), key=lambda r: r["train"]["sharpe"])
        bg = next(x for x in grid if x["id"] == br["id"])
        bres = results[bg["id"]]
        sm = bres.split_metrics()
        if bg["id"] == g["id"]:
            ps = {k: v for k, v in plc.items()}
        else:
            ptr, pte = placebo(ctx, ctx.weights(bg), N_PLACEBO_INFO, rng)
            ps = placebo_summary(sm["train"]["sharpe"], sm["test"]["sharpe"], ptr, pte)
        best_per_target[tg] = {
            "id": br["id"], "name": br["name"], "description": br["description"],
            "train": br["train"], "test": br["test"], "test_sharpe_cost0": br["test_sharpe_cost0"],
            "test_sharpe_cost30": round(backtest(ctx.weights(bg), ctx.px, cost_bps=30.0).split_metrics()["test"]["sharpe"], 4),
            "test_years_pct": yearly_returns(bres.returns), "ttest": br["ttest"], "placebo": ps,
        }

    audit = lookahead_audit(panel, grid, ctx)
    align = {"max_mismatch_days": max(r["alignment_mismatch_days"] for r in rows),
             "n_configs_with_mismatch": sum(r["alignment_mismatch_days"] > 0 for r in rows)}
    print(f"lookahead audit: {audit}; alignment: {align}")

    out = {
        "meta": {
            "seed": SEED, "cost_bps": COST, "lag": LAG, "train": list(TRAIN), "test": list(TEST),
            "pre_extra_period": list(PRE), "targets": list(TARGETS),
            "universe": "top_n=20, min_history=180, min_volume=1e6 (point in time)",
            "price_timing": ("Coin Metrics PriceUSD for date D = reference rate at 00:00 UTC of D+1, "
                             "so return day D = UTC calendar day D"),
            "grid": {"tom_last": TOM_LAST, "tom_first": TOM_FIRST, "dow_adapt_k": DOW_ADAPT_K,
                     "dow_adapt_windows": ["expanding" if x is None else x for x in DOW_ADAPT_WIN],
                     "moy_adapt_k": MOY_ADAPT_K,
                     "moy_adapt_windows": ["expanding" if x is None else x for x in MOY_ADAPT_WIN],
                     "dow_min_obs": DOW_MIN_OBS, "moy_min_obs": MOY_MIN_OBS},
            "btc_bh_test_sharpe_threshold": btc_test,
            "n_configs_tested": len(rows),
            "runtime_s": round(time.time() - t0, 1),
        },
        "benchmarks": benchmarks,
        "lookahead_audit": audit,
        "alignment_audit": align,
        "grid_summary": grid_sum,
        "multiple_testing": mt,
        "effect_tables": effect_tables(ctx),
        "chosen": chosen,
        "best_per_target": best_per_target,
        "best_per_family": best_per_family(rows, grid, ctx, results, rng, g["id"], plc),
        "post_hoc_thursday": thursday_post_hoc(ctx),
        "grid": rows,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1, default=float))
    flat = pd.DataFrame([{
        "id": r["id"], "name": r["name"], "target": r["target"], "family": r["family"], "side": r["side"],
        "train_sharpe": r["train"]["sharpe"], "test_sharpe": r["test"]["sharpe"],
        "test_sharpe_cost0": r["test_sharpe_cost0"], "test_total_pct": r["test"]["total_pct"],
        "test_max_dd_pct": r["test"]["max_dd_pct"], "exposure_test": r["test"]["exposure"],
        "turnover_test": r["test"]["turnover_per_year"],
        "train_t": r["ttest"]["train"]["t"], "train_p": r["ttest"]["train"]["p"],
        "test_t": r["ttest"]["test"]["t"], "test_p": r["ttest"]["test"]["p"],
        "pre_t": r["ttest"]["pre"]["t"]} for r in rows])
    flat.to_csv(OUT_DIR / "grid.csv", index=False, float_format="%.4f")
    print(json.dumps({"grid_summary": grid_sum["all"], "chosen_checks": checks,
                      "mt_train": {k: v for k, v in mt["train"].items() if k != "bonferroni_survivors"},
                      "mt_test": {k: v for k, v in mt["test"].items() if k != "bonferroni_survivors"}},
                     indent=1, default=float))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
