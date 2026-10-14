# Time-series trend following (absolute momentum) on crypto

Reproduce: `cd /home/user/11 && python3 research/trend_ts/run.py` (about 8 minutes, fixed seeds).
All numbers below come from `results.json`, which holds every grid row plus the chosen config's details.

## Bottom line

**No robust edge.** The config with the best TRAIN Sharpe (the only legitimate pick) is a fast
alt-coin trend filter. It went from a TRAIN Sharpe of 1.56 to a TEST Sharpe of 0.03. That is
no better than random timing (placebo percentile 50), it loses money at 30 bps costs and it was
positive in only 2 of 5 TEST years.

One sub-result is worth remembering, but it is **not** a robust candidate under the protocol.
**BTC-only long/flat trend** (hold BTC above its moving average or after a positive past return,
otherwise cash) beat BTC buy-and-hold on Sharpe in 22 of 32 configs in TEST and in 31 of 32 in
the untouched 2014-2017 period, with roughly half the drawdown. In TEST it was still only at the
77th-87th percentile of random-timing placebos, below the 95 required. Read it as a tool that
cuts drawdowns, not as proven alpha.

## What was tested (176 configs, all run, grid fixed before looking)

| Hypothesis | Configs | Grid |
|---|---|---|
| (a) top-20 long/flat | 64 | signal {past L-day return > 0, close > L-day MA} x L {20,50,100,200} x rebalance {weekly, daily} x weights {equal, inverse 60d vol} x sizing {fixed 1/N slot with the rest in cash, fully invested over the "on" assets} |
| (b) top-20 long/short | 32 | same signals; long uptrends, short downtrends, 5%/yr short fee |
| (c) BTC only | 32 | signal x L x rebalance x {long/flat, long/short} |
| (d) vol-targeted 40% | 48 | (a) fully invested, plus BTC long/flat, scaled to 40% ex-ante vol (60-day), max gross 1 |

The universe is the point-in-time top 20 by market cap (`universe(top_n=20, min_history=180,
min_volume=1e6)`). Costs are 10 bps and the lag is 1 day (harness defaults). An automatic
lookahead audit rebuilds the weights from data cut at 2021-06-15 and finds them identical up to
that date.

## The chosen config (best TRAIN Sharpe over the whole grid)

`a_LF|top20|lf|ma20|D|iv|slot`: hold each top-20 coin whose close is above its 20-day MA,
inverse-vol sized slots, the rest in cash, rebalanced daily.

| | Sharpe | CAGR | Total | Max DD |
|---|---|---|---|---|
| TRAIN 2018-21 | 1.56 | +97% | +1414% | -40% |
| TEST 2022-26 | 0.03 | -5% | -21% | -57% |
| BTC buy&hold TEST | 0.48 | +12% | +65% | -67% |

- TEST Sharpe is 0.03, far below 0.48. With 30 bps costs it is -0.25, with lag 0 it is 0.12 and with lag 2 it is 0.12.
- TEST years: 2022 -42.7%, 2023 +62.8%, 2024 +64.0%, 2025 -37.6%, 2026 (to 24 May) -16.9%.
- Placebos (500 draws each):
  - Timing (`circular_shift`): 49.6th percentile. A variant that rotates only within 2018 onward reaches the 59.8th.
  - Selection (`permute_assets`): 100th percentile, and 99.4th with zero costs on both sides.

  Choosing coins in an uptrend beat picking random alts. Random alts were simply terrible in
  2022-26 (placebo median Sharpe -0.39), so this is relative, not absolute, value.
- Robust checks: TEST > BTC no, placebo >= 95 no, cost30 > BTC no, 3 of 5 positive years no.
  Grid median > 0 yes (0.15). **Not a candidate.**

## The whole grid on TEST

- Median TEST Sharpe 0.15. 25 of 176 configs beat 0.48. The rank correlation between TRAIN and TEST Sharpe is 0.08, so TRAIN ranking says almost nothing about TEST.
- By group (median TEST Sharpe / configs beating 0.48):
  - top-20 long/flat: 0.08 (0 of 64)
  - top-20 long/short: -0.16 (0 of 32)
  - top-20 vol-targeted: 0.21 (2 of 32). Its TRAIN-to-TEST rank correlation is -0.77, so the better it looked in TRAIN, the worse it did.
  - BTC long/flat: 0.59 (10 of 16)
  - BTC long/short: 0.25 (1 of 16)
  - BTC vol-targeted: 0.63 (12 of 16)
- By lookback: L=20 has the best TRAIN (median 1.10) and the worst TEST (median -0.04). L=50/100 hold up best (TEST median about 0.24).
- MA versus return signal and weekly versus daily rebalancing made little difference.

## Best of each hypothesis (TRAIN-chosen within it, information only)

| | TRAIN Sharpe | TEST Sharpe | TEST total | TEST DD | cost30 | lag0 | TEST placebo pct |
|---|---|---|---|---|---|---|---|
| (a) ma20 daily iv slot | 1.56 | 0.03 | -21% | -57% | -0.25 | 0.12 | timing 50, selection 100 |
| (b) L/S ma20 daily iv | 1.38 | -0.24 | -65% | -75% | -0.66 | -0.12 | timing 67, selection 100 |
| (c) BTC ma50 daily long/flat | 1.27 | 0.60 | +89% | -52% | 0.47 | 0.70 | timing 77.5 |
| (d) BTC ma50 daily vt40 | 1.46 | 0.74 | +109% | -40% | 0.61 | 0.89 | timing 87 |

For the two BTC picks, the extra out-of-sample period 2014-2017 (never used for selection) gives:

- (c): Sharpe 1.72, max DD -40%
- (d): Sharpe 1.83, max DD -28%
- BTC buy-and-hold: Sharpe 1.37, max DD -81%

Their timing-placebo percentile in that window is 97.5 for (c) and 98.5 for (d). The (d) pick
would pass every check except the TEST placebo (87 < 95). The (c) pick also fails cost30
(0.47 < 0.48).

## Why I believe this

- The alt-coin versions earned their TRAIN Sharpe from two regimes: sitting in cash through the 2018 crash and riding the 2021 alt season. From 2022 the alt market drifted down with sharp bear rallies. A fast filter (20-day) was whipsawed and paid about 48x turnover a year. The fast lookbacks that TRAIN prefers are exactly the ones that failed.
- Shorting downtrending alts lost money: short squeezes plus the 5% fee. Survivorship bias (LUNA missing, dead coins sold at their last price) favours long alts, so real-world alt results would likely be worse, not better.
- BTC trend following is consistent across three periods (2014-17, 2018-21, 2022-26) and across most parameters, which argues against a parameter fluke. However, 4.4 years of TEST is too short to separate it from random timing with the same ~50% exposure. Its 2022 still lost 40-52% to whipsaws (Feb, Apr, Aug and the FTX crash in Nov).

## Caveats on the harness (not edited; see `harness_concerns` in the structured report)

- `circular_shift` rotates each asset over its whole price history. In multi-asset portfolios, positions from 2013-17 (universe of 1-12 coins, so large weights; BTC alone at weight 1) land in TEST. The placebo's gross exposure then averages 0.61 versus the real 0.43 and can reach 2x. The 2018-onward variant changes the percentile from 50 to 60, which does not change the verdict.
- `permute_assets` re-draws random assets on every rebalance row. With daily rebalancing the placebo pays large turnover costs, so I also report a zero-cost comparison.
- The last panel row (2026-05-24) has no prices. Daily-rebalanced strategies are therefore liquidated on the last day at a cost of about 10 bps. This is negligible.
