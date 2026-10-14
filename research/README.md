# Istraživanje kripto strategija (dnevni podaci, Coin Metrics)

## Ukratko

- Testirano je **6 obitelji strategija s ukupno 1367 varijanti**, plus unaprijed zadana provjera BTC trenda (32 varijante).
- **Nijedna nije prošla strogi test**: bolja od BTC kupi-i-drži na razdoblju koje nije vidjela, bolja od nasumičnih verzija same sebe, otporna na troškove i stabilna kroz godine i parametre.
- Najdosljedniji rezultat je **trend filter na samom BTC-u** (drži BTC dok je iznad SMA50, inače gotovina, veličina prema volatilnosti). On pouzdano **smanjuje najveći pad**, ali tri neovisne recenzije pokazuju da to uglavnom dolazi od manje prosječne izloženosti, a ne od tajminga. Prednost tajminga vidi se samo u 2014–2017.
- Obrazac koji se stalno ponavlja: **što je strategija bolja na 2018–2021, to je češće lošija na 2022–2026.**
- SMC obrasci (sweep, FVG, order block) na EURUSD 1H: sweep i FVG gube, a order block je na malom uzorku promijenio predznak između razdoblja. Za zaključak trebaju godine unutardnevnih podataka.

## Rezultati po obitelji

Parametri birani na **2018–2021**, ocjena na **2022-01-01 → 2026-05-23**. Trošak 0.1% po strani, trgovanje dan nakon signala.
Mjerilo: **BTC kupi-i-drži** na 2022–2026: Sharpe **0.48**, +65%, najveći pad −67%.

| Obitelj (mapa) | Varijanti | Odabrana na 2018–21 | Sharpe 2018–21 | Sharpe 2022–26 | Ukupno 2022–26 | Medijan mreže 2022–26 | Postavki > 0.48 |
|---|---|---|---|---|---|---|---|
| Trend po tokenu (`trend_ts`) | 176 | top 20 iznad SMA20, dnevno | 1.56 | **0.10** | −11% | 0.15 | 27 |
| Rotacija tokena (`xs_momentum`) | 246 | tokeni jači od BTC-a zadnjih 7 dana | 1.12 | **0.25** | −33% | −0.09 | 0 |
| On-chain (`onchain`) | 300 | BTC+ETH dok su neto tokovi na burze ≤ 0 | 1.55 | **0.18** | −3% | 0.30 | 51* |
| Rizik i režimi (`regime_vol`) | 105 | stop 10% od vrha, ulaz na 50-dnevni vrh | 1.59 | **−0.09** | −29% | 0.31 | 31 |
| Kalendar (`calendar`) | 252 | ETH samo zadnja 3 i prva 3 dana mjeseca | 1.91 | **−0.51** | −53% | 0.02 | 18 |
| Povratak cijene (`mean_reversion`) | 288 | kupi 3 najveća 1-dnevna pada, tjedno | 0.47 | **−0.30** | −91% | −0.37 | 2 |

\* Postavke koje su na 2022–26 bile iznad 0.48 nisu se mogle odabrati unaprijed: korelacija rangova između razdoblja unutar obitelji je negativna. Podaci o tokovima na burze se naknadno revidiraju, pa ti rezultati nisu ostvarivi uživo.

Detalji, cijele mreže i placebo rezultati su u `<mapa>/results.json` i `<mapa>/README.md`.

## Podaci

- Izvor: besplatna arhiva [Coin Metrics](https://github.com/coinmetrics/data), dnevno, 2010-07 → 2026-05-23. Burzovni API-ji nisu bili dostupni iz okruženja.
- 92 imovine nakon izbacivanja stablecoina, zlata, wrapped tokena i duplikata (`pinetest/crypto_data.py`).
- Cijena je dnevni referentni tečaj u 00:00 UTC (samo zatvaranje, bez OHLC).
- Ograničenja:
  - **Pristranost preživjelih:** neki propali tokeni (LUNA/UST) nedostaju. To uljepšava strategije koje drže altcoine.
  - **Nepotpun skup tokena:** SOL, TON i TRX nisu rangirani, pa nekoliko malih tokena popunjava njihova mjesta u "top 20".
  - **Tokovi na burze nisu point-in-time:** Coin Metrics naknadno revidira pripisivanje adresa burzama.
  - **Volumen** je prijavljeni volumen, koji je napuhan.

## Alat i pravila

- `pinetest/portfolio.py`:
  - Odluka se donosi na zatvaranju dana *t*, a trguje se na zatvaranju *t+1* (konzervativno).
  - Trošak je 0.1% po strani na promet, a short plaća 5% godišnje.
  - Univerzum je top N po kapitalizaciji, isključivo iz podataka poznatih tog dana (`universe()`).
- **Protokol:**
  - Mreža parametara zadaje se unaprijed i prijavljuje cijela.
  - "Najbolja" postavka bira se samo po Sharpeu na **2018–2021 (TRAIN)**.
  - Ocjena je na **2022–2026 (TEST)**, a gdje se može i na 2014–2017.
- **Kandidat mora:**
  - imati Sharpe na TEST-u > 0.48 (BTC kupi-i-drži);
  - biti iznad 95. percentila placeba (≥ 200 nasumičnih verzija);
  - imati Sharpe > 0.48 i uz trošak 0.3%;
  - biti u plusu u ≥ 3 od 5 godina;
  - imati medijan mreže > 0.
- Svaki kandidat ide kod recenzenta koji ga pokušava srušiti.

## BTC trend: dodatna provjera i recenzije (`btc_trend/`)

Unaprijed zadana hipoteza: drži BTC dok je iznad SMA *N* (ili je povrat *N* dana > 0), inače gotovina, opcionalno uz ciljanu volatilnost od 40%. Mreža ima 32 postavke. Na 2018–2021 odabrana je **SMA50, ciljana volatilnost 40%, dnevno**.

| | Strategija | BTC kupi-i-drži |
|---|---|---|
| 2018–2021 (odabir) | Sharpe 1.46 | 0.78 |
| 2022–2026 | Sharpe **0.74**, +109%, pad −40% | 0.48, +65%, pad −67% |
| 2014–2017 (neviđeno) | Sharpe **1.83**, +825%, pad −28% | 1.37, +1808%, pad −81% |
| Mreža (32) | Sharpe bolji u 29/32 (2022–26) i 31/32 (2014–17), pad manji u 32/32 | |
| Placebo (nasumični tajming) | 88. percentil (2022–26), 99. (2014–17) | |

**Tri neovisne recenzije** (`btc_trend/review_*`) reproducirale su svaki broj vlastitim kodom i nisu našle gledanje u budućnost. Ipak su tvrdnju o prednosti na 2022–2026 **srušile**:
- **Manji pad je posljedica manje izloženosti.** Fiksnih 43% BTC-a (ista prosječna izloženost) imalo je pad **−35.6%**, manji od strategije. U 2022. strategija je izgubila −39.8%, a fiksna pozicija −32%.
- **Prednost u Sharpeu nije značajna** (bootstrap p ≈ 0.27, 90% interval [−0.36, +0.87]). Ako razdoblje počne u srpnju 2022., prednost nestaje.
- **Ukupno 2014–2026 manje od kupi-i-drži:** 95.8× naspram 104.9×. Zaostaje u svakoj bull godini (2017.: +187% naspram +1337%).
- **Osjetljiva je na izvedbu:** tjedne verzije gube od kupi-i-drži. Pragovi troška su ~50 bps po strani, a uz kašnjenje od 3 dana Sharpe pada na 0.56.
- **Vrijednost tajminga potvrđena je samo za 2014–2017** (placebo 98–99%, a i pad je manji od fiksne pozicije).

**Zaključak:** ispravno izgrađen alat za smanjenje rizika, ali ne i dokazana prednost u trgovanju.

## Što se dosljedno pokazalo

1. **Odabir po prošlosti ne predviđa budućnost.** Korelacija rangova između TRAIN i TEST razdoblja je 0.07–0.17, a unutar nekih obitelji i negativna (kraj mjeseca −0.65).
2. **Manji pad = manje izloženosti.** Alati za upravljanje rizikom režu pad, ali u 86 od 105 slučajeva jednako to postiže i fiksna manja pozicija.
3. **Altcoin košarice gube od BTC-a** u gotovo svakoj varijanti, a stvarnost je zbog nedostajućih propalih tokena još gora.
4. **Troškovi ubijaju kratke horizonte.** Rotacija i povratak cijene imaju promet od 80 do 400 puta godišnje.
5. **"Kupi kad padne" ne radi na 1–14 dana.** Tokeni koji su naglo pali u prosjeku nastavljaju zaostajati.

## SMC obrasci na EURUSD 1H (`smc_eurusd/`)

Kripto unutardnevni podaci nisu bili dostupni, pa su "Smart Money Concepts" testirani na jedinom unutardnevnom uzorku koji imamo:
- **Podaci:** EURUSD 1H, 2017-04 → 2018-02, 5000 svijeća.
- **Definicije** (`pinetest/smc.py`): liquidity sweep, fair value gap i order block, uz SL i TP u omjeru R po trejdu.
- **Simulator** (`pinetest/events.py`): limit nalozi; kad su SL i TP dotaknuti u istoj svijeći, računa se da je SL prvi.
- **Trošak:** ~1.1 pip po trejdu.
- **Protokol:** postavka se bira na prvih 70%, a ocjenjuje na zadnjih 30% (~3 mjeseca).

| Obrazac | Odabrana postavka | Prvi dio: prosjek R (n) | Zadnji dio: prosjek R (n) | Medijan mreže, zadnji dio |
|---|---|---|---|---|
| Liquidity sweep | lookback 48, 1:3 | −0.02 (158) | **−0.33** (72) | −0.18 |
| Fair value gap | razmak 0.5 ATR, 1:1.5 | +0.46 (12) | **−0.43** (9) | −0.37 |
| Order block | lookback 24, 1:2 | +0.22 (35) | **+0.86** (14) | +0.63 |

- **Sweep** gubi i gori je od istih trejdova s nasumičnim smjerom. Nakon probijenog lowa cijena češće nastavlja nego se vraća, isto kao kod kripta.
- **FVG** ima premalo trejdova za ikakav zaključak.
- **Order block** je na zadnjem dijelu jako pozitivan (placebo 99–100%), ali je na prvom dijelu u **7 od 9 postavki bio negativan**, a zadnji dio ima samo 9–33 trejda po postavki. Takav preokret između razdoblja na malom uzorku najčešće je slučajnost.
- **Za pravi test** trebaju godine podataka na 1m–1h, za više tržišta (Binance povijest), s istim protokolom.

## Ispravci alata nakon prvog kruga

Agenti i recenzenti prijavili su slabosti koje su ispravljene (commit `2eabf84`), nakon čega su sve skripte ponovno pokrenute:
- **Univerzum:** kapitalizacija se računa iz cijene i zadnje poznate ponude kad nedostaje (BNB, DOT), a tokeni bez prijavljenog volumena (AVAX, SHIB) više nisu isključeni.
- **Zadnji datum bez cijena** je uklonjen.
- **`rebalance_on`** više ne tretira nedovršeni tjedan kao dan rebalansiranja.
- **`permute_assets`** ne bira nove nasumične tokene dok stvarna strategija drži istu poziciju.
- **`circular_shift`** može se ograničiti na razdoblje (`start=`).
- **Propast računa** se sada prijavljuje (`died_on`).

Usporedba prije i nakon ispravaka (Sharpe odabrane postavke na 2022–26 / medijan mreže):

| Obitelj | Prije | Poslije |
|---|---|---|
| Trend po tokenu | 0.03 / 0.15 | 0.10 / 0.15 |
| Rotacija tokena | 0.35 / −0.11 | 0.25 / −0.09 |
| On-chain | 0.18 / 0.31 | 0.18 / 0.30 |
| Rizik i režimi | 0.40 / 0.30 | −0.09 / 0.31 (odabrana je druga postavka) |
| Kalendar | −0.51 / 0.05 | −0.51 / 0.02 |
| Povratak cijene | −0.08 / −0.30 | −0.30 / −0.37 (odabrana je druga postavka) |
| BTC trend | 0.74 / 0.63 | 0.74 / 0.63 |

Nijedan zaključak se nije promijenio.

README-i pojedinih obitelji opisuju prvo pokretanje; `results.json` sadrži ponovljeno.

## Kako ponoviti

```bash
git clone --depth 1 https://github.com/coinmetrics/data /home/user/coinmetrics/data   # ili postavi COINMETRICS_DIR
cd /home/user/11
python -m pytest
PYTHONPATH=. python3 research/<obitelj>/run.py
```
