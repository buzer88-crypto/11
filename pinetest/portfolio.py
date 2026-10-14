"""Daily multi-asset portfolio backtester on close prices (no lookahead by construction).

Conventions:
- `weights.loc[t]` is the target allocation decided with data up to and including close t.
  It is traded at close t+lag. The default lag=1 (trade one full day after the signal) is
  deliberately conservative; lag=0 trades at the same close the signal used.
- A row that is entirely NaN means "no rebalance": the portfolio keeps its drifted weights.
  In a row with any value, NaN means 0. Use `rebalance_on` to trade only on chosen days.
- Weights are fractions of equity: positive long, negative short, the rest is cash (earns 0).
- Costs: `cost_bps` per side on traded notional, measured against the drifted weights.
  Shorts also pay `short_fee_annual` (borrow/funding) on their notional, accrued daily.
- An asset without a price on a trade date gets target weight 0. A held asset whose price
  series ends is valued at its last price and sold at the next rebalance.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # crypto trades every day
TRAIN = ("2018-01-01", "2021-12-31")
TEST = ("2022-01-01", None)


@dataclass
class PortfolioResult:
    returns: pd.Series          # daily net returns
    gross_exposure: pd.Series   # sum of |held weights| during each day
    turnover: pd.Series         # traded notional / equity at each close
    held: pd.DataFrame          # weights held during each day
    died_on: pd.Timestamp | None = None   # first day equity hit zero, if it did

    def equity(self, start=None, end=None) -> pd.Series:
        return (1 + self.returns.loc[start:end]).cumprod()

    def metrics(self, start=None, end=None) -> dict:
        r = self.returns.loc[start:end]
        if r.empty:
            return {}
        eq = (1 + r).cumprod()
        years = len(r) / DAYS_PER_YEAR
        std = r.std()
        return {
            "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "total_pct": 100 * (eq.iloc[-1] - 1),
            "cagr_pct": 100 * (eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -100.0,
            "vol_pct": 100 * std * np.sqrt(DAYS_PER_YEAR),
            "sharpe": float(r.mean() / std * np.sqrt(DAYS_PER_YEAR)) if std > 0 else 0.0,
            "max_dd_pct": 100 * float((eq / eq.cummax() - 1).min()),
            "exposure": float(self.gross_exposure.loc[start:end].mean()),
            "turnover_per_year": float(self.turnover.loc[start:end].sum() / years),
            "died": self.died_on is not None and self.died_on <= r.index[-1],
        }

    def split_metrics(self) -> dict:
        return {"train": self.metrics(*TRAIN), "test": self.metrics(*TEST),
                "full": self.metrics(TRAIN[0], None)}


def backtest(weights: pd.DataFrame, prices: pd.DataFrame, cost_bps: float = 10.0, lag: int = 1,
             short_fee_annual: float = 0.05) -> PortfolioResult:
    prices = prices.reindex(columns=weights.columns)
    weights = weights.reindex(index=prices.index)
    rebalance = weights.notna().any(axis=1).shift(lag, fill_value=False).to_numpy()
    target = weights.fillna(0.0).shift(lag, fill_value=0.0)
    target = target.where(prices.notna(), 0.0).to_numpy()
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()

    n_days, n_assets = rets.shape
    held = np.zeros(n_assets)
    out = np.zeros(n_days)
    gross = np.zeros(n_days)
    turnover = np.zeros(n_days)
    held_hist = np.zeros((n_days, n_assets))
    fee_daily = short_fee_annual / DAYS_PER_YEAR
    dead = False
    for t in range(n_days):
        if dead:
            break
        held_hist[t] = held
        gross[t] = np.abs(held).sum()
        port = held @ rets[t] - fee_daily * -held[held < 0].sum()
        if 1 + port <= 0:
            out[t], dead = -1.0, True
            continue
        drifted = held * (1 + rets[t]) / (1 + port)
        new = target[t] if rebalance[t] else drifted
        traded = np.abs(new - drifted).sum()
        turnover[t] = traded
        out[t] = (1 + port) * (1 - cost_bps / 1e4 * traded) - 1
        held = new
    idx = prices.index
    died_on = idx[int(np.flatnonzero(out == -1.0)[0])] if dead else None
    return PortfolioResult(pd.Series(out, idx), pd.Series(gross, idx), pd.Series(turnover, idx),
                           pd.DataFrame(held_hist, idx, prices.columns), died_on)


def rebalance_on(weights: pd.DataFrame, freq: str = "W-SUN") -> pd.DataFrame:
    """Keep only the rows on the last day of each `freq` period; other rows become NaN (hold).

    The last available day of a period counts (so gaps such as missing weekends are fine),
    except for a final period that has not ended yet: live, its last row is not a period end.
    """
    periods = weights.index.to_period(freq)
    keep = weights.index.to_series().groupby(periods).transform("max").to_numpy()
    is_end = weights.index == keep
    if len(weights) and periods[-1].end_time.normalize() > weights.index[-1]:
        is_end &= periods != periods[-1]
    out = weights.copy()
    out.loc[~is_end] = np.nan
    return out


def buy_and_hold(prices: pd.DataFrame, asset: str = "btc") -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[asset] = prices[asset].notna().astype(float)
    return w


def equal_weight(members: pd.DataFrame, freq: str = "W-SUN") -> pd.DataFrame:
    """Equal weight across the universe, rebalanced on `freq` period ends."""
    w = members.astype(float)
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return rebalance_on(w, freq)


# --- Placebo controls -------------------------------------------------------------------------

def circular_shift(weights: pd.DataFrame, prices: pd.DataFrame, rng: np.random.Generator,
                   min_shift: int = 90, start=None) -> pd.DataFrame:
    """Timing placebo: rotate each asset's position series by a random offset within that
    asset's own priced span. Keeps each asset's exposure profile and the rebalance days,
    destroys the timing. A real timing signal should beat most of these.

    With `start` (e.g. TRAIN[0]) only positions from that date on are rotated, among themselves,
    so early years with a tiny universe and large weights do not leak into the placebo.
    Positions before `start` are then set to 0."""
    rows = weights.notna().any(axis=1)
    held = weights.ffill().fillna(0.0)
    out = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    in_window = np.ones(len(weights), dtype=bool) if start is None else weights.index >= pd.Timestamp(start)
    for col in weights.columns:
        valid = prices[col].reindex(weights.index).notna().to_numpy() & in_window
        pos = np.flatnonzero(valid)
        if len(pos) <= 2 * min_shift:
            continue
        span = held[col].to_numpy()[pos[0]:pos[-1] + 1]
        k = rng.integers(min_shift, len(span) - min_shift)
        out.iloc[pos[0]:pos[-1] + 1, out.columns.get_loc(col)] = np.roll(span, k)
    out.loc[~rows.to_numpy()] = np.nan
    return out


def permute_assets(weights: pd.DataFrame, members: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Selection placebo: on each rebalance row, give the same weights to randomly chosen
    universe members. Keeps the number of positions and their sizes, destroys the selection.
    While the real row repeats unchanged, the random assignment is kept too (if those coins are
    still members), so the placebo does not trade more often than the real strategy."""
    out = weights.copy()
    m = members.reindex_like(weights).fillna(False).to_numpy()
    w = weights.to_numpy()
    res = out.to_numpy(copy=True)
    prev_real, prev_row = None, None
    for t in np.flatnonzero(~np.isnan(w).all(axis=1)):
        vals = np.nan_to_num(w[t])
        if (prev_real is not None and np.array_equal(vals, prev_real)
                and m[t][prev_row != 0].all()):
            res[t] = prev_row
            continue
        nz = vals[vals != 0]
        pool = np.flatnonzero(m[t])
        row = np.zeros(len(vals))
        if len(nz) and len(pool) >= len(nz):
            row[rng.choice(pool, len(nz), replace=False)] = rng.permutation(nz)
        res[t] = row
        prev_real, prev_row = vals, row
    return pd.DataFrame(res, index=weights.index, columns=weights.columns)
