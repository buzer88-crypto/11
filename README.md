# Pine Script strategije + Python testovi

- `pine/` – TradingView strategije (Pine Script v6). Kopiraj sadržaj u Pine Editor i klikni *Add to chart*.
  - `ema_cross.pine` – long kad brza EMA (9) presiječe sporu (21) prema gore, izlaz na suprotni presjek.
  - `rsi_reversion.pine` – long kad RSI(14) izađe iznad 30, izlaz kad prijeđe 70 ili na SL 5% / TP 10%.
- `pinetest/` – Python replika istih strategija i backtester koji oponaša TradingView zadane postavke
  (signal na zatvaranju svijeće, izvršenje na otvaranju sljedeće, 100% kapitala, provizija 0.1%).

## Pokretanje

```bash
pip install -r requirements.txt
python -m pytest              # testovi
python -m pinetest ema_cross podaci.csv   # backtest na CSV-u (stupci open, high, low, close)
```

CSV možeš izvesti iz TradingViewa (*Export chart data*). Rezultati bi trebali biti blizu onih u
Strategy Testeru, ali ne identični (TV računa količinu po closeu signalne svijeće, zaokružuje itd.).

Pine kod se ne može izvršiti izvan TradingViewa, pa `tests/test_pine_sync.py` provjerava da su
zadani parametri u `.pine` datotekama isti kao u Python verziji.
