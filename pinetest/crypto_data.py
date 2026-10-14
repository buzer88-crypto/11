"""Daily crypto panel from the free Coin Metrics community archive (github.com/coinmetrics/data).

Prices are daily reference rates: one close per day, no open/high/low. The archive covers only
assets in Coin Metrics' free tier, so several dead coins (e.g. LUNA/UST) are missing and results
carry some survivorship bias; others that collapsed (FTT, SRM, PPT, WTC, ...) are included.
"""
import os
import pickle
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(os.environ.get("COINMETRICS_DIR", "/home/user/coinmetrics/data/csv"))
CACHE = Path(__file__).resolve().parent.parent / "data" / "crypto_panel.pkl"

STABLE = {
    "busd", "crvusd_eth", "dai", "eurc_eth", "fdusd_eth", "frax_eth", "gusd", "husd", "lusd_eth",
    "pax", "pyusd_eth", "sai", "sdai_eth", "susde_eth", "tusd", "tusd_eth", "tusd_trx", "usdc",
    "usdc_avaxc", "usdc_eth", "usdc_trx", "usdd_eth", "usde_eth", "usdk", "usdm_eth", "usdt",
    "usdt_avaxc", "usdt_eth", "usdt_omni", "usdt_trx", "buidl_eth",
}
COMMODITY = {"paxg", "xaut"}
WRAPPED = {"wbtc", "weth", "renbtc", "hbtc", "wnxm"}
# The same asset tracked on another chain or under a second id; the kept id is noted.
DUPLICATES = {
    "avaxp", "avaxx",            # avaxc
    "flow_evm", "flow_native",   # flow
    "leo_eos",                   # leo_eth
    "bnb_eth", "eos_eth", "trx_eth",
    "pol_eth",                   # matic_eth (renamed to POL)
}
EXCLUDE = STABLE | COMMODITY | WRAPPED | DUPLICATES

FIELDS = (
    "PriceUSD", "CapMrktCurUSD", "volume_reported_spot_usd_1d", "CapMVRVCur", "AdrActCnt",
    "TxCnt", "FlowInExUSD", "FlowOutExUSD", "SplyCur",
)


def load_panel(data_dir: Path = DEFAULT_DIR, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return {field: DataFrame(date x asset)} for FIELDS, all on the PriceUSD date index.

    Cached in data/crypto_panel.pkl; pass refresh=True after updating the archive.
    """
    if CACHE.exists() and not refresh:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    for path in sorted(Path(data_dir).glob("*.csv")):
        asset = path.stem
        if asset in EXCLUDE:
            continue
        d = pd.read_csv(path, usecols=lambda c: c == "time" or c in FIELDS)
        if "PriceUSD" not in d or d["PriceUSD"].notna().sum() < 30:
            continue
        d.index = pd.to_datetime(d.pop("time"))
        for f in FIELDS:
            if f in d:
                frames[f][asset] = d[f]
    prices = pd.DataFrame(frames["PriceUSD"]).sort_index()
    priced = prices.notna().any(axis=1)
    prices = prices.loc[priced.idxmax():priced[::-1].idxmax()]
    panel = {f: pd.DataFrame(cols).reindex(index=prices.index) for f, cols in frames.items() if cols}
    panel["PriceUSD"] = prices
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


def universe(panel: dict[str, pd.DataFrame], top_n: int = 30, min_history: int = 180,
             min_volume: float = 1e6, volume_window: int = 30) -> pd.DataFrame:
    """Point-in-time tradable universe (bool, date x asset), using only data up to each date.

    An asset qualifies on date t if it has a price on t, at least `min_history` days of prices,
    a median reported spot volume over the last `volume_window` days of at least `min_volume`
    USD (skipped when min_volume is 0, and for assets whose volume is not reported), and is
    among the `top_n` largest by market cap of the assets that pass those filters. Where the
    archive has no market cap, price x the last reported supply is used instead (BNB after
    2019-04, DOT after 2022-06); TRX has neither and never qualifies. The ranking is among
    the archive's assets only, so a few small coins fill slots of missing large ones (SOL, TON).
    """
    px = panel["PriceUSD"]
    ok = px.notna() & (px.notna().cumsum() >= min_history)
    if min_volume:
        vol = panel["volume_reported_spot_usd_1d"].reindex_like(px)
        median = vol.rolling(volume_window, min_periods=10).median()
        ok &= median.isna() | (median >= min_volume)
    mcap = panel["CapMrktCurUSD"].reindex_like(px)
    if "SplyCur" in panel:
        mcap = mcap.fillna(px * panel["SplyCur"].reindex_like(px).ffill())
    rank = mcap.where(ok).rank(axis=1, ascending=False, method="first")
    return ok & (rank <= top_n)
