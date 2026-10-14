# Mean reversion in crypto

Reproduce with `cd /home/user/11 && python3 research/mean_reversion/run.py`. It takes about 4 minutes
and uses fixed seeds. It writes:
- `results.json`: benchmarks, grid summaries, the chosen config with placebos and robustness checks,
  the TRAIN-best config of each hypothesis, an event study, and every grid row.
- `grid.csv`: the same grid rows.

## What was tested

Every rule decides at close t using data up to t. The shared harness trades at close t+1 (lag 1)
and charges 10 bps per side plus a 5%/yr fee on shorts. Trading starts on 2017-12-31.
TRAIN is 2018-2021 and TEST runs from 2022 to 2026-05-24. Vol always means the 60-day standard
deviation of daily log returns, measured **before** the move being scored.

| hypothesis | rule | grid | configs |
|---|---|---|---|
| (a) `ts_mr` | z = k-day return / (vol * sqrt k). Buy a top-N coin when z < -thr, short it when z > +thr, or do both. Each position is 1/N of equity and is held H days after the last trigger. | k {1,3,7} x H {1,3,7} x thr {1.5,2,2.5} x side {long, short, both} x N {10,20} | 162 |
| (b) `xs_rev` | Rank the top 20 by L-day return (raw or in vol units). Either buy the k worst, or buy the k worst and short the k best (dollar neutral). Rebalance daily or weekly. | L {1,2,3} x k {3,5} x side {long, ls} x freq {D,W} x score {raw, vol} | 48 |
| (c) `pair` | z of log(ETH/BTC) against its rolling W-day mean and std. When \|z\| > e, go long the cheap coin and short the expensive one at +-0.5, and exit when z crosses 0 (`pair`). Or hold 100% of the cheap coin only (`rotate`). | W {30,90,180} x e {1,1.5,2,2.5} x mode | 24 |
| (d) `alt_btc` | z of (alt k-day return - BTC k-day return) in units of relative vol. When z < -thr, long the alt and short BTC by the same amount, 1/(2(N-1)) per alt, held H days. | k {3,7,14} x thr {1.5,2,2.5} x H {3,7,14} x N {10,20} | 54 |

All 288 configs were run and all are kept. The "best" config was picked by TRAIN Sharpe only,
across the whole grid. A truncation test confirms that the weights up to a cut date do not change
when all later data is removed.

## Result: no edge, and the sign points the other way

**TRAIN-chosen config:** `xs_rev_L2_k3_long_W_raw_n20`. Every Sunday it buys the 3 top-20 coins with
the worst 2-day return, in equal weights, and holds them for a week.

| | Sharpe | CAGR | total | max DD |
|---|---|---|---|---|
| TRAIN | 0.40 | -15.6% | -49% | -90% |
| TEST | **-0.08** | -31.8% | **-81%** | -85% |
| BTC buy&hold TEST | 0.48 | 12% | +65% | -67% |

- The best TRAIN Sharpe in the entire grid (0.40) is already below simply holding BTC in TRAIN (0.78).
- TEST by year: 2022 -73%, 2023 +8%, 2024 -8%, 2025 -10%, 2026 (to May) -23%.
- Costs and timing:
  - It turns over 83x a year.
  - TEST Sharpe is 0.03 with no costs and -0.28 at 30 bps.
  - It is -0.26 with lag 0 and 0.11 with lag 2.
- Placebos (300 draws each): `permute_assets` 35th percentile (33rd gross of costs), `circular_shift` 41st.
- It is a long-only basket of beaten-down coins: beta to BTC is 1.15 and alpha is -34%/yr in TEST.
- **Not a robust candidate.** It fails all six checks.

**Whole grid on TEST:** the median TEST Sharpe is **-0.30**, -0.17 even with zero costs, and -0.57 at 30 bps.
Only 2 of 288 configs beat 0.48. Both were TRAIN losers (TRAIN Sharpe -0.09 and 0.01).

| hypothesis | median TRAIN SR | median TEST SR | TEST SR, no costs | TEST SR, 30 bps | > 0.48 | median turnover/yr |
|---|---|---|---|---|---|---|
| (a) ts_mr | -1.03 | -0.43 | -0.30 | -0.70 | 1 / 162 | 15 |
| (b) xs_rev | -0.55 | -0.82 | -0.34 | -1.72 | 0 / 48 | 203 |
| (c) pair | 0.01 | 0.05 | 0.10 | -0.06 | 0 / 24 | 13 |
| (d) alt_btc | -0.49 | 0.00 | 0.16 | -0.30 | 1 / 54 | 7 |

What the grid says:
- **Shorting pumps loses everywhere.** The median is -1.28 in TRAIN and -1.00 in TEST, and it is still
  -0.88 gross of costs. Every one of the 54 short-side configs is negative in TEST.
- **Buying dumps is about flat**, with a median TEST Sharpe of 0.09 and -0.30 in TRAIN. It holds
  only ~6% of equity on average, so it is a small, part-time long position, not a reversal edge.
- **Cross-sectional reversal is negative in all 48 configs.** The daily versions have a median
  TEST Sharpe of -1.54 net and -0.57 gross. Daily long/short alone is -2.74 net and -1.03 gross.
  Daily turnover of about 420x a year costs ~42%/yr at 10 bps.
- **The BTC/ETH relative-value trade loses as a true pair.** The dollar-neutral `pair` mode has a
  median of -0.38 in TRAIN and -0.22 in TEST. The `rotate` mode looks better (+0.21 TEST), but only
  because it is net long BTC or ETH most of the time: beta to BTC is 0.88 and alpha is negative.
  Its TRAIN-best version is at the 54th percentile of a timing placebo.
- **Alt vs BTC is the only area with a small hint.** It appears at k=14, where the median TEST
  Sharpe is 0.24 (0.38 gross). The TRAIN median at k=14 is -0.37, and at 30 bps the median falls to 0.02.
- **The event study agrees.** Top-20 coins are tracked after a +-2 vol move, measured against the
  equal-weight universe. After big drops they keep underperforming over 1-7 days: -6 to -83 bps,
  naive t mostly about -2 in TRAIN, and the same sign in every TEST cell. After big pumps they show no consistent reversal.
  At daily to weekly horizons crypto shows continuation, not mean reversion. The TRAIN/TEST rank
  correlation across the grid is 0.66, because the same rules lose in both periods.

### Best config of each hypothesis (chosen on TRAIN within the hypothesis; supplementary)

| config | TRAIN SR | TEST SR | TEST SR, 30 bps | TEST SR, lag 0 | placebo pct (min) | positive TEST years |
|---|---|---|---|---|---|---|
| `ts_mr_k1_H7_z1.5_long_n20` | 0.19 | 0.24 | 0.10 | 0.03 | 82 | 2 / 5 |
| `xs_rev_L2_k3_long_W_raw_n20` (overall pick) | 0.40 | -0.08 | -0.28 | -0.26 | 33 | 1 / 5 |
| `pair_W30_e1.0_rotate` | 0.33 | 0.21 | 0.10 | 0.10 | 54 | 2 / 5 |
| `alt_btc_k14_z2.5_H14_n20` | 0.01 | 0.49 | 0.35 | 0.19 | 86 | 3 / 5 |

`alt_btc_k14_z2.5_H14_n20` comes closest. It is market neutral, with a TEST beta to BTC of 0.00,
and passes 4 of the 6 checks. It still fails, for these reasons:
- **It made nothing in TRAIN.** TRAIN Sharpe is 0.01.
- **It fails the 30 bps cost test** (TEST Sharpe 0.35).
- **The full grid's median is negative.**
- **The placebo percentile is only 86 once costs are equalised.** Shuffling the alts lands at the
  100th percentile with costs, but the shuffled portfolios re-draw their coins daily and pay much
  higher costs. Gross of costs, the shuffle puts it at 86.5.
- **The returns are tiny.** Gross exposure averages about 5%, for +6.6% total over 4.4 years.

It is one cell picked from a 54-config sub-grid, and the TRAIN data gave no reason to pick it.

## Caveats

- **Survivorship.** LUNA and UST are missing, which flatters "buy the dump" in 2022. Even so, the long
  side does not work.
- **Short fees.** A flat 5%/yr is optimistic for altcoin shorts. Real fees would make the short-side
  and long/short results worse still.
- **Event-study t-stats** are naive: events overlap (a k=7 move triggers on several consecutive days),
  so the t values are overstated. They are a diagnostic only.
- **Placebo method.** The official placebo percentile is the minimum over all placebos run for that
  hypothesis. For the coin-shuffling placebos it also includes the gross-of-cost comparison, because
  the harness's `permute_assets` re-draws coins on every rebalance row and so overcharges placebos of
  multi-day holds.
