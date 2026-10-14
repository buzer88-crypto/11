"""Paired circular block bootstrap of Sharpe(strategy) - Sharpe(B&H) in PRE, TRAIN, TEST, and PRE+TEST pooled."""
import numpy as np
import pandas as pd
from pinetest.crypto_data import load_panel
from pinetest.portfolio import TEST, TRAIN, backtest, buy_and_hold

px = load_panel()["PriceUSD"]; p = px["btc"]
vol60 = p.pct_change(fill_method=None).rolling(60).std() * np.sqrt(365)
w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
w["btc"] = ((p > p.rolling(50).mean()).astype(float) * (0.40 / vol60).clip(upper=1.0)).where(p.notna(), 0.0).fillna(0.0)
s, b = backtest(w, px).returns, backtest(buy_and_hold(px, "btc"), px).returns


def boot(a, c, n=2000, block=60, seed=7):
    rng = np.random.default_rng(seed); a, c = a.to_numpy(), c.to_numpy(); T = len(a); k = int(np.ceil(T / block))
    d = np.empty(n)
    for i in range(n):
        idx = ((rng.integers(0, T, k)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T
        x, y = a[idx], c[idx]; d[i] = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(365)
    pt = (a.mean() / a.std() - c.mean() / c.std()) * np.sqrt(365)
    return round(pt, 3), [round(np.quantile(d, .05), 3), round(np.quantile(d, .95), 3)], round((d <= 0).mean(), 3)


for name, (a0, a1) in {"pre": ("2014-01-01", "2017-12-31"), "train": TRAIN, "test": TEST}.items():
    print(name, boot(s.loc[a0:a1], b.loc[a0:a1]))
oos = pd.concat([s.loc["2014":"2017"], s.loc["2022":]]); oob = pd.concat([b.loc["2014":"2017"], b.loc["2022":]])
print("pre+test pooled", boot(oos, oob))
