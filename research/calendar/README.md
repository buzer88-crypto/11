# Calendar effects on crypto daily closes (BTC, ETH, top-20 basket)

Reproduce: `cd /home/user/11 && python3 research/calendar/run.py` (about 5 minutes, fixed seed).
All numbers below come from `results.json`, which holds every grid row plus the chosen config's
details. `grid.csv` is a flat copy of the grid.

## Bottom line

**No robust edge.** The rule with the best TRAIN Sharpe was "hold ETH only on the last 3 and the
first 3 days of each month". It had a TRAIN Sharpe of 1.90 and then a TEST Sharpe of -0.51
(-53% in TEST). On TEST it did worse than 92% of random-timing placebos. Turn-of-month won TRAIN
for all three targets (BTC, ETH and the basket) and lost money on TEST for all three. No calendar
slice was significant after a multiple-testing correction, in TRAIN or in TEST. In TEST only 5 of
153 slices reached raw p < 0.05, fewer than the 7.7 expected by pure chance.

## What was tested (252 configs, 153 distinct slices, all run, grid fixed up front)

Each rule is long the target on the days inside a slice and in cash otherwise. The targets are
BTC, ETH and the point-in-time top-20 equal-weight basket (`universe(top_n=20, min_history=180,
min_volume=1e6)`). Costs are 10 bps per side and the harness lag is 1.

| Family | Configs per target | Slices |
|---|---|---|
| day of week | 14 | only weekday d / every day except d (7 days) |
| weekend | 2 | Sat+Sun only / Mon-Fri only |
| turn of month | 24 | last L in {1,2,3} + first F in {1,2,3,4} days, inside / outside |
| month of year | 24 | only month m / all months except m |
| Halloween | 2 | Nov-Apr / May-Oct |
| adaptive weekday | 12 | the k in {1..6} weekdays with the best mean return so far, expanding or 730-day window, re-estimated daily from data <= t |
| adaptive month | 6 | the k in {3,6,9} best months so far, expanding or 1461-day window |

That is 84 configs per target and 252 in total. An "inside" rule and its "outside" complement
share one t-test, so there are 153 distinct slices. That count (153) is the multiple-testing
denominator: the Bonferroni threshold is p < 0.05/153 = 0.00033.

## What the fixed daily close implies

- Coin Metrics `PriceUSD` for date D is the reference rate at 00:00 UTC at the end of day D (on
  2020-03-12 it is 4,959, the post-crash close). So the return labelled D is exactly the UTC
  calendar day D, and every slice here is defined in UTC days.
- UTC days do not line up with any trading community's local day. "UTC Monday" runs from Sunday
  20:00 to Monday 20:00 in New York (EDT), and from Monday 09:00 to Tuesday 09:00 in Tokyo. The
  "UTC weekend" contains the Sunday-evening CME futures reopen and the Monday-morning Asian open.
  Effects that live in a local session, or in a particular hour, are blurred or split across two
  days. Intraday effects (US hours versus Asian hours, for example) cannot be seen at all.
- One fixing per day means a single minute's price sets both ends of every return. The rule is
  assumed to trade at that index level, which is not a tradable price. 00:00 UTC is also when
  perpetual funding settles and daily candles close on many venues, so real slippage there is
  probably worse than the flat 10 bps.
- A calendar is known in advance, so a rule needs no data to know it will be long on Monday.
  Weights row t therefore holds the slice flag of return day t+lag+1. That is calendar
  arithmetic, not a shift of data. An audit checks the positions the harness actually held
  against the intended slice days: 0 mismatched days across all 252 configs. A truncation audit
  (data cut at 2021-06-15) reproduces every weight exactly (max difference 0.0), so the adaptive
  rules use no future data.
- Turn-of-month uses calendar days, not business days, because crypto trades every day.

## The chosen config (best TRAIN Sharpe over the whole grid)

`eth|tom|L3F3|in`: hold ETH on the last 3 and the first 3 UTC days of every month, cash otherwise.

| | Sharpe | CAGR | Total | Max DD | Exposure |
|---|---|---|---|---|---|
| TRAIN 2018-21 | 1.90 | +101% | +1536% | -32% | 20% |
| TEST 2022-26 | **-0.51** | -16% | -53% | -65% | 20% |
| extra 2015-17 (ETH from Aug 2015) | 1.05 | +46% | +356% | -46% | 12% |
| BTC buy&hold TEST | 0.48 | +12% | +65% | -67% | 100% |
| ETH buy&hold TEST | 0.17 | -12% | -43% | -74% | 100% |

- **Effect (Welch t-test, mean daily return inside vs outside the slice):**
  - TRAIN: +112 vs +3 bps/day, t = 3.41, p = 0.0007. After Bonferroni over 153 slices, p = 0.11, so not significant.
  - TEST: -16 vs +8 bps/day, t = -1.14, p = 0.26. The sign flipped.
  - 2015-17: +116 vs +100 bps/day, t = 0.27. No effect even before TRAIN.
- **Placebo:** 500 `circular_shift` draws. The real TEST Sharpe sits at the **8th percentile**
  (placebo median 0.00, 95th percentile 0.54). The same rule sits at the 99.4th percentile on
  TRAIN, which is the in-sample fit.
- **Costs and lag:**
  - 30 bps costs: TEST Sharpe -0.69. With 0 bps: -0.42, so costs are not why it fails.
  - Lag 0: rebuilt for lag 0 the exposure is identical, so -0.51. With the lag-1 weights run at lag 0 (one day early, so the last 4 and first 2 days): 0.21. Run at lag 2 (one day late): -0.27.
- **TEST years:** 2022 -1.1%, 2023 +17.5%, 2024 -39.3%, 2025 -14.4%, 2026 (to 24 May) -22.2%.
  Only 1 of 5 years was positive.
- **Robust checks:**
  - TEST Sharpe > 0.48: no.
  - Placebo >= 95: no.
  - 30 bps > 0.48: no.
  - At least 3 positive years: no.
  - Grid median TEST Sharpe > 0: yes (0.05), but only because many rules are long BTC most of the time.

  **Not a candidate.**

## Turn of month: real in the past, gone now?

Turn of month (last 2 + first 3 days, mean bps/day inside vs outside the slice):

| | 2014-17 | TRAIN 2018-21 | TEST 2022-26 |
|---|---|---|---|
| BTC | 68 vs 20 (t 1.9) | 62 vs 8 (t 2.1) | -5 vs 9 (t -0.9) |
| ETH | 96 vs 104 (t -0.1) | 123 vs 5 (t 3.4) | -21 vs 8 (t -1.3) |
| top-20 | 96 vs 38 (t 1.8) | 81 vs 3 (t 2.6) | -5 vs 4 (t -0.4) |

For BTC and the basket the turn-of-month effect was positive in two separate periods. It never
cleared a multiple-testing bar, and it reversed in 2022-26. Within the turn-of-month family the
rank correlation between TRAIN and TEST Sharpe is -0.65: the windows that looked best in TRAIN
did worst in TEST. This is the classic pattern of an anomaly that was either noise or has been
traded away. Either way it gives no usable edge today.

## The whole grid and the t-tests

- **TEST Sharpe across the grid:**
  - Median 0.05 at 10 bps (0.18 at 0 bps). 18 of 252 configs beat 0.48 (42 at 0 bps).
  - 15 of the 18 are BTC rules that hold BTC most of the time (all months but one, all days but one) or a single good month. With 84 BTC configs scattered around BTC's own 0.48, that many winners is about what chance gives.
  - Rank correlation between TRAIN and TEST Sharpe: 0.07.
- **Costs matter:** a one-day-a-week rule trades about 104 times a year, about 10% a year at 10 bps
  and about 31% a year at 30 bps. The median TEST Sharpe of the weekday family falls from 0.17 at
  0 bps to -0.08 at 10 bps.
- **Multiple testing (153 slices):**
  - TRAIN: 24 slices at raw p < 0.05. That is more than the 7.7 expected, but they are almost all overlapping turn-of-month windows on three highly correlated targets, so it is really one cluster. 0 survive Bonferroni or Holm. Benjamini-Hochberg (q = 0.10) passes 4, all ETH turn-of-month windows, and all 4 had a negative t in TEST.
  - TEST: 5 at raw p < 0.05, 0 after any correction.
- **Replication:** of the 24 TRAIN-nominal slices, 5 kept their sign in TEST and none was
  significant there. Across all 153 slices the correlation of TRAIN t and TEST t is -0.17.

TRAIN-best config inside each family (information only, placebo 200 draws, 500 for the chosen
turn-of-month rule):

| Family | TRAIN-best | TRAIN Sharpe | TEST Sharpe | TEST t | Placebo pct |
|---|---|---|---|---|---|
| day of week | ETH only on Saturday | 1.14 | -0.38 | 0.27 | 51 |
| weekend | ETH only Sat+Sun | 1.02 | -0.24 | 0.04 | 27 |
| turn of month | ETH last 3 + first 3 | 1.90 | -0.51 | -1.14 | 8 |
| month of year | ETH only April | 1.42 | -0.23 | -0.60 | 24 |
| Halloween | ETH Nov-Apr | 0.93 | 0.04 | -0.24 | 38 |
| adaptive weekday | ETH best 6 of 7 weekdays (730 d) | 0.82 | 0.11 | 0.62 | 75 |
| adaptive month | BTC best 6 months (1461 d) | 0.94 | 0.62 | 0.97 | 86 |

The only family winner above 0.48 in TEST is the adaptive-month BTC rule. Its in/out difference
is not significant (t = 0.97), it is at the 86th placebo percentile and it had 2 of 5 positive
years. Picking it now would be a second selection made after seeing TEST.

## Post hoc: weak Thursdays (noticed after looking at TEST, information only)

UTC Thursday was the worst weekday for all three targets in both TRAIN and
TEST. For BTC it averaged -28 bps/day (t -1.4) in TRAIN and -32 bps/day (t -2.3) in TEST, and
-30 bps/day (t -2.4, p 0.017) over 2018-26. However:

- It was not significant in TRAIN, so it was never selected.
- It was absent in 2014-17 (+36 bps, t +0.3).
- The three targets are highly correlated, so they are not three independent confirmations.
- p = 0.017 is nowhere near the 0.00033 Bonferroni bar.

"BTC every day except Thursday" had a TEST Sharpe of 0.65 at 10 bps. Treat it as a curiosity to
re-check on future data, not as an edge.

## Caveats

- The top-20 basket is rebalanced daily so that entries and exits fall exactly on slice edges and
  so that `circular_shift` sees a position every day. This costs some drift turnover: the
  always-in daily basket has a TEST Sharpe of 0.11, against 0.15 for the weekly benchmark.
- The basket t-tests use the membership of the previous day, while the strategy uses the
  membership from two days earlier. The difference is negligible.
- For a day-of-week rule on a single asset, `circular_shift` can only map the rule onto another
  weekday (7 outcomes). For those rules the placebo is coarse. It is fine for the turn-of-month
  and month rules, including the chosen one.
- The archive is missing some dead coins (LUNA/UST), which flatters long-altcoin rules. That
  makes the negative basket results conservative.
