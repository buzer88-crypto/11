# Cross-sectional momentum and reversal across coins

Reproduce with `cd /home/user/11 && python3 research/xs_momentum/run.py` (about 2.5 min, fixed seeds).
Outputs: `results.json` (benchmarks, grid summary, chosen config with placebos and robustness,
best config per hypothesis, every grid row) and `grid.csv` (the same grid rows).

## What was tested

All strategies rebalance weekly. Each one decides at the Sunday close using data up to that close,
and the shared harness trades at the next close (lag 1, 10 bps per side, 5%/yr short fee).
Universe: point-in-time top-10, top-20 or top-30 coins by market cap. Trading starts at
2017-12-31. TRAIN is 2018-2021 and TEST is 2022 to 2026-05-24.

| hypothesis | rule | grid | configs |
|---|---|---|---|
| A `long_mom` | long the top k coins by past L-day return, equal weight | top_n {10,20,30} x L {7,14,30,60,90} x skip {0,7} x k {3,5} | 60 |
| B `ls_mom` | long top k at +0.5/k, short bottom k at -0.5/k (dollar neutral, gross 1) | same | 60 |
| C `long_rev` | long the k worst coins over L days | top_n x L {3,7,14} x k {3,5} | 18 |
| C `ls_rev` | long the k worst, short the k best | same | 18 |
| D `rel_btc` | long the top k coins whose L-day return beats BTC's; empty slots go to BTC (k=all: every coin beating BTC, equal weight) | top_n x L x skip x k {3,5,all} | 90 |

That is 246 configs, all run and all kept. The "best" config was picked by TRAIN Sharpe only.

## Result: no robust edge

**TRAIN-chosen config:** `rel_btc_n20_L7_s0_kall`. Each week it holds, in equal weights, every top-20
coin whose 7-day return beat BTC's (BTC only when no coin did, about 2% of weeks).

| | Sharpe | CAGR | total | max DD |
|---|---|---|---|---|
| TRAIN | 1.12 | 85% | +1074% | -87% |
| TEST | 0.35 | -1.4% | -6% | -82% |
| BTC buy&hold TEST | 0.48 | 12% | +65% | -67% |
| top-20 equal weight TEST | 0.15 | -12% | -44% | -73% |

- TEST by year: 2022 -76%, 2023 +69%, 2024 +147%, 2025 +35%, 2026 (to May) -30%.
  BTC over the same years: -64%, +155%, +121%, -6%, -13%.
- Selection placebo (`permute_assets`, 500 draws): real TEST Sharpe beats 97% of the draws. A stricter
  placebo that leaves BTC's weight on BTC and shuffles only the altcoins puts it at 99.2%.
  The timing placebo (`circular_shift`, 200 draws) puts it at 88.5%.
- Costs and lag: TEST Sharpe is 0.14 at 30 bps. It is 0.40 with lag 0 and 0.43 with lag 2.
  Turnover is about 76x a year, so every extra 10 bps per side costs about 7.6% a year.
- Against the top-20 equal-weight basket it has a beta of 0.99 and an alpha of about +16%/yr.
  Against BTC it has a beta of 1.08 and an alpha of about -1%/yr.
- Robust-candidate checks: it fails TEST Sharpe > 0.48, cost-30 Sharpe > 0.48 and grid median > 0.
  It passes the placebo (>= 95) and has 3 of 5 positive years. **Not a candidate.**

**Whole grid on TEST:** the median TEST Sharpe is -0.11, and only 2 of 246 configs beat 0.48.

| hypothesis | median TEST Sharpe | configs > 0.48 | configs > 0 |
|---|---|---|---|
| long_mom | -0.07 | 0 / 60 | 21 / 60 |
| ls_mom | -0.58 | 2 / 60 | 6 / 60 |
| long_rev | -0.08 | 0 / 18 | 6 / 18 |
| ls_rev | -0.45 | 0 / 18 | 0 / 18 |
| rel_btc | -0.04 | 0 / 90 | 37 / 90 |

- **Long-only momentum is just market beta.** Its median beta to the matched equal-weight basket is 0.98,
  with a median alpha of -11%/yr against it. Only 4 of 60 configs beat the matched EW basket's
  TEST Sharpe, and none beat BTC. The TRAIN-best long-only config (`long_mom_n30_L90_s7_k5`) lost 88% in TEST.
- **Long/short momentum:** the only two configs above 0.48 are `ls_mom_n20_L7_s0` with k=3 (0.48) and
  k=5 (0.50). The k=3 one is placebo-significant (97.5%), but at 30 bps its TEST Sharpe falls to -0.04.
  The same signal on top-10 and top-30 is negative, at -0.36 and -0.42.
- **Short-term reversal does not work:** every long/short reversal config is negative in TEST.
- **Skipping the last 7 days hurts every variant.** Whatever works here is a 1-2 week continuation
  effect, not classic 3-12 month momentum.
- **The weak positive pocket exists only in the top-20 universe.** Its median TEST Sharpe is +0.03, against
  -0.12 for top-10 and -0.21 for top-30. When the same signal flips sign between top-20 and top-30, it is
  a coincidence of which coins sat in ranks 11-20, not a broad effect.

## Why I believe this

The placebos show that weekly short-horizon relative strength among top-20 coins did pick better
altcoins than chance in 2022-2026. That small effect does not survive four things:
- realistic costs (it turns over 75-90x a year)
- a change of universe size
- comparison with simply holding BTC
- the rest of the grid, which is mostly negative

The TRAIN/TEST rank correlation across the grid is only 0.39. TRAIN Sharpes of 1.0-1.1 came with TEST
Sharpes from -0.3 to 0.5, so the TRAIN winners were largely the 2018-2021 altcoin bull market.

## Caveats

- **Survivorship.** LUNA is missing from the archive. It was a top-10 coin with strong momentum in
  March-April 2022 before going to zero in May 2022, so long-momentum TEST numbers are probably
  slightly optimistic.
- **Short costs.** The 5%/yr short fee is optimistic for altcoins, and many top-20 coins were hard to
  short in 2018-2019. Both points flatter the long/short results.
- **Trading start.** If trading starts before 2018, a long/short book can be wiped out in 2017. A short
  XRP position lost 100% of equity on 2017-04-02, when XRP rose 174% in one day. That is why trading
  starts at 2017-12-31.
