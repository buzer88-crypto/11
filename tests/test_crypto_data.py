import pandas as pd
import pytest

from pinetest import crypto_data


def write(dir_, name, rows):
    pd.DataFrame(rows).to_csv(dir_ / f"{name}.csv", index=False)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    days = pd.date_range("2020-01-01", periods=60).strftime("%Y-%m-%d")
    base = {"time": days, "volume_reported_spot_usd_1d": [5e6] * 60}
    write(tmp_path, "big", {**base, "PriceUSD": range(1, 61), "CapMrktCurUSD": [3e9] * 60})
    write(tmp_path, "mid", {**base, "PriceUSD": [2.0] * 60, "CapMrktCurUSD": [2e9] * 60})
    young = [None] * 25 + [1.0] * 35
    write(tmp_path, "young", {**base, "PriceUSD": young, "CapMrktCurUSD": [5e9] * 60})
    thin = {**base, "volume_reported_spot_usd_1d": [1e3] * 60}
    write(tmp_path, "thin", {**thin, "PriceUSD": [1.0] * 60, "CapMrktCurUSD": [9e9] * 60})
    write(tmp_path, "usdt", {**base, "PriceUSD": [1.0] * 60, "CapMrktCurUSD": [9e10] * 60})
    monkeypatch.setattr(crypto_data, "CACHE", tmp_path / "cache.pkl")
    return tmp_path


def test_load_panel_excludes_stablecoins_and_caches(archive):
    panel = crypto_data.load_panel(archive)
    assert set(panel["PriceUSD"].columns) == {"big", "mid", "young", "thin"}
    assert panel["CapMrktCurUSD"].index.equals(panel["PriceUSD"].index)
    assert (archive / "cache.pkl").exists()


def test_universe_is_point_in_time(archive):
    panel = crypto_data.load_panel(archive)
    u = crypto_data.universe(panel, top_n=2, min_history=30, min_volume=1e6)
    day = u.index[50]
    assert not u.loc[day, "thin"]                  # too little volume
    assert not u.loc[day, "young"]                 # only 26 days of history
    assert u.loc[day, ["big", "mid"]].all()
    assert not u.iloc[20].any()                    # nobody has 30 days of history yet
    assert u.loc[u.index[-1]].sum() == 2
