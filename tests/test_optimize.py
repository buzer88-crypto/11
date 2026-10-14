from pinetest.optimize import EMA_GRID, ema_valid, grid_search
from pinetest.strategies import ema_cross


def test_grid_search_ranks_by_in_sample_and_respects_constraint(random_walk):
    grid = {"fast_len": [5, 9, 25], "slow_len": [10, 21]}
    res = grid_search(random_walk, ema_cross, grid, valid=ema_valid)
    assert len(res) == 4                       # (25,10) and (25,21) are filtered out
    assert (res["fast_len"] < res["slow_len"]).all()
    assert res["train_pct"].is_monotonic_decreasing
    assert res[["train_pct", "test_pct"]].notna().all().all()


def test_default_grid_is_valid():
    assert any(ema_valid({"fast_len": f, "slow_len": s})
               for f in EMA_GRID["fast_len"] for s in EMA_GRID["slow_len"])
