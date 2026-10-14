# On-chain valuation and activity signals: no robust edge

`python3 research/onchain/run.py` runs everything in about 4 minutes with fixed seeds. It writes
`results.json`, which holds every grid row, the chosen config, the best config of each hypothesis,
data coverage and the placebos, and `grid.csv`, which has one row per config.
`check_lookahead.py` rebuilds every config's weights from a panel cut off at 2019-06-15 and at
2022-06-15 and confirms that they match the full-panel weights (600 out of 600 match).

## Protocol

- The grid was fixed before running and run in full: **300 configs**.
  - (a) MVRV timing: 120 configs.
  - (b) Cross-sectional MVRV: 48 configs.
  - (c) Activity momentum: 48 configs.
  - (d) Exchange flows: 84 configs.
- The config was chosen by **TRAIN Sharpe only** (2018–2021) over the whole grid. TEST (2022 to
  2026-05) is only reported.
- The harness defaults apply throughout: costs of 10 bps, lag=1, gross exposure of at most 1.
  Signals use only expanding or rolling windows.

## Data coverage

| Metric | Coverage |
|---|---|
| CapMVRVCur | 86 assets; about 92% (TRAIN) and 95% (TEST) of top-20 members. XMR has no MVRV. BNB's MVRV ends in 2019. |
| AdrActCnt | 89 assets; about 95% of members (XMR missing). |
| TxCnt | All 92 assets. |
| FlowInExUSD / FlowOutExUSD | BTC (from 2011) and ETH (from 2015) only. |

## Result

The chosen config is `d_flows#287`. It holds BTC and ETH in 50/50 slots, each long only when its
30-day net exchange inflow is not positive and flat otherwise, rebalanced weekly.

| | Sharpe | CAGR | Total | Max DD |
|---|---|---|---|---|
| TRAIN | 1.54 | +87% | +1124% | −44% |
| TEST | **0.18** | −0.8% | −3% | −52% |
| BTC buy & hold, TEST | 0.48 | +12% | +65% | −67% |
| BTC+ETH 50/50 buy & hold, TEST | 0.31 | | +5% | |

- **Placebo, timing (circular_shift, 500 draws):** the real config's TEST Sharpe beats **43.6%**
  of the placebo draws. Its TRAIN Sharpe beat 99.8% of them.
- **Supplementary placebo** that keeps the TEST exposure fixed and rotates positions only inside
  TEST: 37.4%.
- **Costs, lag and yearly returns:**
  - TEST Sharpe is 0.13 at 30 bps, 0.11 at lag=0 and 0.14 at lag=2.
  - TEST years: 2022 −45%, 2023 +69%, 2024 +68%, 2025 −29%, 2026 (partial) −12%.
- **It is not a robust candidate.** It fails four of the five required checks. Only the grid's
  median TEST Sharpe > 0 passes.

## What the grid says

- **Whole grid:** median TEST Sharpe is 0.31, below BTC's 0.48. 55 of the 300 configs beat 0.48
  on TEST.
- **TRAIN does not predict TEST.** The Spearman rank correlation between TRAIN and TEST Sharpe is
  +0.14 across the grid. It is −0.40 within the flow configs and −0.32 within the MVRV timing
  configs.
- **Per slice:** within 53 parameter slices, the TRAIN-best config beats BTC on TEST in 3, and only
  barely (0.48–0.52).
- **(a) MVRV timing:** median TEST Sharpe 0.36; 10 of 120 configs beat BTC (best 0.55).
  - In TEST, BTC's MVRV never went above 2.8, so level rules sat at their default exposure.
  - The TRAIN pick was in effect ETH buy & hold: TEST Sharpe 0.17, total −43%.
- **(b) Cross-sectional MVRV:** median TEST Sharpe 0.05.
  - The TRAIN pick (raw MVRV, the 3 cheapest coins, monthly) had TEST Sharpe 0.44 and beat 94.5%
    of random-selection placebos. Random top-20 alt picks were very poor in TEST (median 0.09), so
    it only beat a weak baseline.
  - The long-short versions had TRAIN and TEST medians of about 0.
- **(c) Activity momentum:** median TEST Sharpe 0.30.
  - The TRAIN pick (AdrActCnt 30-day growth, the top 3 coins) had TEST Sharpe 0.35 and beat 96%
    of selection placebos. It turns over its book about 60 times a year, so at 30 bps its Sharpe
    falls to 0.20.
  - The long-short activity configs were the best on TEST (up to 1.05, positive every year), but
    their TRAIN median was 0.05. That is a sign flip between periods, not something TRAIN could
    have found.
- **(d) Exchange flows:** they had the strongest TRAIN Sharpes of the four (1.2–1.5) and the
  largest collapse on TEST.
  - In TRAIN, net inflows came with the 2018 bear market. In TEST the link reversed. The 2022
    crash came with net *outflows* (withdrawals around FTX), so the filter stayed about 70% long in
    BTC, and the 2023–24 rally came with inflows.
  - BTC-only flow configs look good on TEST (median 0.61). The TRAIN-best of them still scores only
    0.42, so that observation comes from looking at TEST, not from something tradable.

**Conclusion:** none of the four hypotheses gives a robust edge over BTC buy & hold on this data.
Selection placebos show that cheap-MVRV and rising-activity coins beat *random* top-20 coins in
TEST. They did not beat simply holding BTC, after costs or before them.

## Caveats about the harness and data

- **Exchange flows are not point-in-time.** Coin Metrics labels exchange addresses with hindsight
  and revises the history. TRAIN-era flows were probably cleaner than anything available in real
  time, which may explain part of the very strong TRAIN Sharpe.
- **On-chain data for day t is published after the UTC day closes,** so the lag=0 results are
  slightly optimistic. lag=1 is realistic.
- **The `universe()` top-20 silently excludes several large coins in TEST** because their market cap
  or volume is missing from the archive:
  - BNB: market cap ends in 2019.
  - TRX: no market cap.
  - AVAX and SHIB: no volume.
  - DOT: market cap ends in 2022-06.
  - SOL is not in the archive.

  Their slots go to coins such as XVG, GNO, QNT and HT, so the "top-20" in TEST is not the real
  top-20. Together with missing dead coins (LUNA), this distorts the cross-sectional tests (b) and
  (c).
- **`circular_shift` takes placebo positions from an asset's whole price history** (BTC since
  2010), so the placebos' TEST exposure differs from the real one. That is why `test_only_shift` is
  also reported. Both give the same verdict.
