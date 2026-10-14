# Risk management and regime timing on crypto

This folder tests vol targeting, a BTC regime filter, vol-compression breakouts and drawdown stops.

To reproduce, run `cd /home/user/11 && python3 research/regime_vol/run.py`. It takes about 12 minutes and uses fixed seeds.
It writes two files:

- `results.json`: benchmarks, the lookahead audit, the grid summary, the chosen config with its placebos and robustness checks, the best config of each hypothesis and of each group, a post-hoc BTC section and every grid row.
- `grid.csv`: one line per config.

## Bottom line

**No robust edge.** These tools did what risk tools do: they cut drawdowns, sharply in most
cases. But they cut them almost entirely by **holding less**, not by timing well.

- **Drawdowns.** 102 of 105 configs had a smaller TEST max drawdown than buy-and-hold of the same asset. Only 19 of 105 had a smaller drawdown than simply holding that asset at a constant fraction equal to the strategy's own average exposure.
- **Sharpe.** It improved over the matching buy-and-hold in 67 of 105 configs, but only by a median of +0.07.
- **The pick.** The TRAIN-chosen config fails 4 of the 5 robustness checks.

## What was tested (105 configs, grid fixed up front, all run, none dropped)

| Hypothesis | Configs | Grid |
|---|---|---|
| (a) vol targeting | 24 | {BTC, top-20 EW basket} x target {20, 40, 60}% x trailing realized vol {20, 60}d x rebalance {weekly, daily}; scale = min(target/vol, 1) |
| (b) BTC regime filter | 18 | hold {BTC, alt basket = top-20 minus BTC, top-20 basket} only while BTC > its N-day MA, N {50, 100, 200}, else cash; weekly or daily check |
| (c) vol-compression breakout | 36 | {BTC, each top-20 coin in a 1/N slot} x 20d vol (previous day) in its lowest x% {10, 20, 30} x percentile {expanding, rolling 365d} x close > prior N-day high, N {20, 50, 100}; exit below the prior N/2-day low |
| (d) drawdown stop | 27 | {BTC, top-20 basket (stop on its own index), each top-20 coin in a 1/N slot} x stop X% {10, 20, 30} below the highest close since entry; re-enter on a new N-day closing high, N {20, 50, 100} |

Setup:

- **Universe.** The point-in-time `universe(top_n=20, min_history=180, min_volume=1e6)`.
- **Execution.** Harness defaults: 10 bps per side, lag 1, cash earns 0.
- **Rebalancing.** "Daily" means the decision is made every day, but the strategy trades only on days the target changes, plus a weekly drift rebalance.
- **Lookahead audit.** Every config's targets were rebuilt from a panel cut at 2021-06-15. They are identical to the full-panel targets up to that date (max difference 0.0).

## The chosen config (best TRAIN Sharpe over the whole grid)

The chosen config is `d_DD|top20_each|stop10|hi50|D`. Each top-20 coin gets a 1/N slot. The
slot goes to cash after a 10% fall from the coin's highest close since entry. It re-enters when
the coin makes a new 50-day closing high.

| | Sharpe | CAGR | Total | **Max DD** | Avg exposure |
|---|---|---|---|---|---|
| TRAIN 2018-21 | 1.48 | +46% | +355% | **-18%** | 0.17 |
| TEST 2022-26 | 0.40 | +6.2% | +30% | **-25%** | 0.17 |
| BTC buy&hold TEST | 0.48 | +12% | +65% | **-67%** | 1.00 |
| top-20 EW buy&hold TEST | 0.15 | -12% | -44% | **-73%** | 1.00 |
| top-20 EW held at a constant 17% (same avg exposure), TEST | 0.16 | +1% | +5% | **-17%** | 0.17 |

**TEST by year:**

| Year | Return |
|---|---|
| 2022 | -20.6% |
| 2023 | +20.0% |
| 2024 | +47.5% |
| 2025 | -1.4% |
| 2026 (to 24 May) | -6.2% |

It made money in only 2 of the 5 years.

**Costs and lag.** The TEST Sharpe is 0.30 with 30 bps costs, 0.70 with lag 0 and 0.45 with lag 2.
A large part of the edge sits in reacting on the same close as the signal.

**Placebos (500 draws each):**

| Placebo | Percentile |
|---|---|
| Timing (`circular_shift`) | 86.2 |
| Timing, rotated within 2018 onward only | 93.6 |
| Selection (`permute_assets`) | 100 |

The selection result means that holding coins that keep making new highs beat holding random
coins. Random alts were very poor in 2022-26: the placebo median Sharpe is about 0. So this is
relative value against a bad alt market, not absolute value.

**Robust-candidate checks:**

| Check | Result |
|---|---|
| TEST Sharpe > 0.48 | no (0.40) |
| Placebo percentile >= 95 | no (86.2) |
| Sharpe at 30 bps > 0.48 | no (0.30) |
| At least 3 of 5 years positive | no (2) |
| Grid median TEST Sharpe > 0 | yes (0.30) |

**Not a candidate.**

**The pick was a near-tie.** The second-best TRAIN Sharpe, 1.480 against 1.483, was the BTC vol-compression breakout
`c_VB|btc|p10|hi100|expanding`. In TEST it was invested only 9% of the time and returned -11%
(Sharpe -0.17). TRAIN ranking says little about TEST here: the Spearman rank correlation across the grid is 0.17.

## Whole grid on TEST (the key question: better risk-adjusted than buy&hold?)

- Median TEST Sharpe is 0.30. 92 of 105 configs are above 0 and 31 of 105 beat BTC's 0.48. 22 of those 31 are BTC-only overlays, which are still mostly long BTC.
- The column "DD < own B&H" counts configs whose TEST max drawdown is smaller than buy-and-hold of the same asset.
- The column "DD < static same exposure" compares against a no-timing benchmark. That benchmark holds the same asset at a constant fraction equal to the strategy's average TEST exposure.

| Group | n | Median TEST Sharpe | > 0.48 | Median TEST max DD | Median exposure | Sharpe > own B&H | DD < own B&H | DD < static same exposure |
|---|---|---|---|---|---|---|---|---|
| (a) VT BTC | 12 | 0.49 | 8 | -56% | 0.82 | 8 | 11 | 7 |
| (a) VT top-20 | 12 | 0.21 | 0 | -55% | 0.66 | 9 | 12 | 6 |
| (b) RF -> BTC | 6 | 0.62 | 4 | -42% | 0.51 | 4 | 6 | 3 |
| (b) RF -> alts | 6 | 0.25 | 0 | -63% | 0.51 | 6 | 6 | 0 |
| (b) RF -> top-20 | 6 | 0.26 | 0 | -62% | 0.51 | 6 | 6 | 0 |
| (c) VB BTC | 18 | 0.26 | 5 | -31% | 0.24 | 5 | 18 | 0 |
| (c) VB each top-20 coin | 18 | 0.17 | 3 | -24% | 0.13 | 9 | 18 | 0 |
| (d) DD BTC | 9 | 0.50 | 5 | -53% | 0.58 | 5 | 7 | 2 |
| (d) DD top-20 basket | 9 | 0.26 | 2 | -50% | 0.36 | 6 | 9 | 0 |
| (d) DD each top-20 coin | 9 | 0.46 | 4 | -34% | 0.32 | 9 | 9 | 1 |
| **all** | **105** | **0.30** | **31** | **-37%** | **0.36** | **67** | **102** | **19** |

What this says:

- **(a) Vol targeting is de-leveraging, not timing.**
  - On BTC the TEST Sharpe stays at 0.43-0.57, against 0.48 for buy-and-hold. The drawdown falls with the target: about -30 to -36% at 20%, -51 to -62% at 40% and -65 to -68% at 60%.
  - A constant BTC weight with the same average exposure gets almost the same drawdown. The 60-day window was slightly better (-30% versus -35% static at 20%).
  - On the top-20 basket the Sharpe went from 0.15 to a median of 0.21, still far below BTC.
- **(b) The BTC regime filter helps BTC itself, not alts.**
  - BTC above its MA, checked daily, gave TEST Sharpes of 0.60, 0.78 and 0.72 with max drawdowns of -35 to -52%. This matches `research/trend_ts`.
  - Applied to alts or the top-20 basket, it raised the Sharpe from about 0.14 to about 0.25. It never beat BTC buy-and-hold, and its drawdowns (-51 to -65%) were worse than simply holding half the basket.
  - In 2022 the TRAIN-picked MA50 version still lost 49% on alts. The MA200 version sat out 2022 (0 to +5%) but still had a -63% drawdown later in TEST.
- **(c) Vol-compression breakouts gave the smallest drawdowns (-11 to -43%) only because they are out of the market about 80% of the time.**
  - No config beat the static same-exposure drawdown.
  - The rolling-365d percentile version had a median TEST Sharpe of 0.06. The expanding version had 0.35, but expanding percentiles drift because BTC vol has trended down since 2011.
  - N=100 failed: its median TEST Sharpe was 0.01.
- **(d) Drawdown stops with slow re-entry (N=100) did best in TEST** (median Sharpe 0.74, 8 of 9 > 0.48). That is a TEST-selected subgroup. Its median TRAIN Sharpe (0.99) was lower than N=50's (1.13), so the protocol could not have picked it. Its TEST edge is mostly one event: the BTC and basket versions were stopped out by the end of 2021 and did not re-enter until 2023, a 0% return in 2022.

## Post-hoc, information only: BTC-only regime filter and drawdown stop

I chose to look here after seeing TEST. I ran 200 timing placebos (`circular_shift`) for each of
the 15 BTC-only configs in (b) and (d), on TEST and on 2014-2017, a period never used for anything else.

- **TEST.** The best placebo percentiles are 91 (BTC > MA100 daily) and 90 (30% stop, N=100). None reaches 95.
- **2014-2017.** Several configs reach 96-100: the 10% stops and BTC > MA50 daily. But 2014-17 contains BTC's two biggest crashes and bubbles, and TEST does not confirm the result.
- **Reading.** A BTC trend or stop overlay is a defensible way to cut BTC's drawdown roughly in half with a similar Sharpe. It is not proven alpha.

## Why I believe the verdict

- The TRAIN period (2018 crash plus the 2020-21 bull run) rewards anything that sits out a crash and catches a boom, so TRAIN Sharpes were inflated across the board (median 0.95). In TEST alts fell 72% in 2022, rallied in 2023-24 and fell again in 2025-26 (top-20 EW total -44%). Overlays on alts limited the damage but could not turn a falling basket into a good investment.
- In 2022 the median config lost 27%. Only 19 of 105 avoided a loss (0% to +5%), all by sitting in cash nearly all year:
  - the six BTC > MA200 filters;
  - the six N=100 drawdown stops on BTC and the basket;
  - seven rarely-invested BTC breakout configs.

  The faster stops and filters fired, but late and repeatedly (whipsaw).
- The drawdown improvements are large and consistent, but nearly all of them are reproduced by a constant smaller position. That is the honest answer to "do these improve risk-adjusted results?": they lower risk, and mostly at the same price per unit of risk.
- Survivorship bias (LUNA/UST missing) flatters the alt-basket results, so real-world alt numbers would likely be worse.

## Harness concerns (not edited)

- **`circular_shift` and per-asset portfolios.** It rotates each asset over its whole priced span. In per-asset portfolios, slots from 2013-2017, when the universe had 1-12 coins and so large 1/N weights, land in TEST. For the chosen config the placebo's mean TEST exposure is 0.25 against the real 0.17. Rotating within 2018 onward only gives 93.6 instead of 86.2, still below 95.
- **`circular_shift` and basket overlays.** For a basket under one common on/off signal, it shifts each coin independently. That breaks the common timing and changes the null hypothesis. I also ran a common-overlay block shift over the full span and over 2018 onward. It gives similar percentiles: 76 and 79 against 71 for the regime-filter pick, and 60 and 60 against 49 for the vol-target pick.
- **Missing last row.** The last panel row (2026-05-24) has no prices, so strategies traded daily are liquidated on that day at about 10 bps. This is negligible.
- **Cash earns 0.** Cash earns nothing in the harness, which understates real-world returns for low-exposure overlays (stablecoin/T-bill yields of about 4-5% in 2023-25). Sharpe is computed on raw returns with no risk-free rate.
