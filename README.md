# Pine Script strategije + Python testovi

- `pine/` – TradingView strategije (Pine Script v6). Kopiraj sadržaj u Pine Editor i klikni *Add to chart*.
  - `ema_cross.pine` – long kad brza EMA (9) presiječe sporu (21) prema gore, izlaz na suprotni presjek.
  - `rsi_reversion.pine` – long kad RSI(14) izađe iznad 30, izlaz kad prijeđe 70 ili na SL 5% / TP 10%.
  - `ema_breakeven.pine` – long/short na EMA 9/21 presjek, početni SL 3%, TP 9%; kad cijena ode 3%
    u plus, SL se pomiče na ulaznu cijenu (breakeven), a TP ostaje. Opcionalni filtar volumena
    (signal samo ako je volumen > 1.5× prosjeka zadnjih 20 svijeća), isključen po zadanom.
- `pinetest/` – Python replika istih strategija i backtester koji oponaša TradingView zadane postavke
  (signal na zatvaranju svijeće, izvršenje na otvaranju sljedeće, 100% kapitala, provizija 0.1%).

## Pokretanje

```bash
pip install -r requirements.txt
python -m pytest              # testovi
python -m pinetest ema_cross podaci.csv   # backtest na CSV-u (stupci open, high, low, close)
python -m pinetest.optimize podaci.csv     # grid search EMA duljina, 70% train / 30% test
```

CSV možeš izvesti iz TradingViewa (*Export chart data*). Rezultati bi trebali biti blizu onih u
Strategy Testeru, ali ne identični (TV računa količinu po closeu signalne svijeće, zaokružuje itd.).

Pine kod se ne može izvršiti izvan TradingViewa, pa `tests/test_pine_sync.py` provjerava da su
zadani parametri u `.pine` datotekama isti kao u Python verziji.

## Optimizacija

`pinetest.optimize` bira parametre samo na prvih 70% podataka i zatim ih mjeri na zadnjih 30%.
Gledaj `test_pct`, ne `train_pct`: ako najbolji parametri na train dijelu nisu dobri i na testu
(ili je korelacija train/test blizu 0), to je prenaučenost, a ne pravi edge.
