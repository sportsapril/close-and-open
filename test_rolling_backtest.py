"""Tests for rolling_backtest.py windowing and beats-SPX logic."""

import numpy as np
import pandas as pd

from backtest import TRADING_DAYS, annualize_return, build_returns, load_prices, summarize
from rolling_backtest import benchmark_windows, generate_windows, window_returns_table


def _trading_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Business-day index covering [Jan 1 start_year, Jan 1 end_year)."""
    return pd.bdate_range(f"{start_year}-01-01", f"{end_year}-01-01", inclusive="left")


def test_generate_windows_stride_and_bounds():
    idx = _trading_index(2010, 2015)  # five full years
    w2 = generate_windows(idx, 2)
    # 2y windows with annual stride: starts 2010..2013
    assert [y for y, *_ in w2] == [2010, 2011, 2012, 2013]
    year, start, end, n_days = w2[0]
    assert start == pd.Timestamp(2010, 1, 1)
    assert end == pd.Timestamp(2012, 1, 1)
    # business-day years have ~260 days, comfortably above 0.95 * 252
    assert n_days >= 0.95 * 2 * TRADING_DAYS

    w5 = generate_windows(idx, 5)
    assert [y for y, *_ in w5] == [2010]


def test_generate_windows_coverage_filter():
    # Data starts in July 2010 -> the 2010 window is only half-covered
    idx = pd.bdate_range("2010-07-01", "2013-01-01", inclusive="left")
    w1 = generate_windows(idx, 1)
    assert [y for y, *_ in w1] == [2011, 2012]


def test_generate_windows_empty_index():
    assert generate_windows(pd.DatetimeIndex([]), 1) == []


def _synthetic_prices(index: pd.DatetimeIndex, overnight_daily: float) -> pd.DataFrame:
    """Prices where each open gaps up overnight_daily vs prior close and the
    intraday leg is flat (close == open)."""
    opens, closes = [], []
    close = 100.0
    for _ in index:
        open_ = close * (1 + overnight_daily)
        opens.append(open_)
        closes.append(open_)  # flat intraday
        close = open_
    return pd.DataFrame({"Open": opens, "Close": closes}, index=index)


def test_beats_spx_known_answer():
    idx = _trading_index(2010, 2013)
    # Stock gaps +10bps overnight every day; SPX benchmark fixed at 5%/yr.
    prices = _synthetic_prices(idx, 0.0010)
    overnight = build_returns(prices)["overnight"]
    spx = {(w, y): 0.05 for w in [1, 2, 3, 4, 5] for y in [2010, 2011, 2012]}

    rows = pd.DataFrame(window_returns_table(overnight, spx, "TEST", 0.95))
    assert not rows.empty
    # Gross: 10bps/day annualizes to ~+28.6%/yr -> beats 5% everywhere
    gross = rows[rows["haircut_bps"] == 0]
    assert gross["beats_spx"].all()
    # 20bps/day haircut turns it into ~-2.5%/yr -> beats nowhere
    costly = rows[rows["haircut_bps"] == 20]
    assert not costly["beats_spx"].any()
    # Windows the benchmark doesn't cover are excluded
    spx_partial = {(1, 2011): 0.05}
    rows_partial = pd.DataFrame(window_returns_table(overnight, spx_partial, "TEST", 0.95))
    assert set(zip(rows_partial["window_years"], rows_partial["window_start"])) == {(1, 2011)}


def test_strategy_ann_return_matches_direct_computation():
    idx = _trading_index(2015, 2016)
    prices = _synthetic_prices(idx, 0.0005)
    overnight = build_returns(prices)["overnight"]
    spx = {(1, 2015): 0.0}
    rows = pd.DataFrame(window_returns_table(overnight, spx, "T", 0.95))
    row = rows[(rows["window_years"] == 1) & (rows["haircut_bps"] == 0)].iloc[0]
    seg = overnight[(overnight.index >= pd.Timestamp(2015, 1, 1))
                    & (overnight.index < pd.Timestamp(2016, 1, 1))]
    expected = annualize_return((1 + seg).cumprod())
    assert abs(row["strategy_ann_ret"] - expected) < 1e-12


def test_benchmark_windows_spy_sanity():
    """Full-sample SPY: a rolling window's b&h return should match summarize's
    math on the same slice, tying the new code back to the existing script."""
    spx = benchmark_windows("data/SPY.csv", 0.95)
    assert len(spx) > 50  # ~25 years x 5 window sizes
    bh = build_returns(load_prices("data/SPY.csv"))["buy_and_hold"]
    seg = bh[(bh.index >= pd.Timestamp(2010, 1, 1)) & (bh.index < pd.Timestamp(2011, 1, 1))]
    assert abs(spx[(1, 2010)] - summarize("x", seg)["ann_return"]) < 1e-12
    # SPY data runs 2000-01-03 .. 2026-03-20: 2026 has ~3 months of data,
    # so no window may start there, and a 5y window can start no later
    # than 2021 (needs data through the end of 2025).
    years_1y = [y for w, y in spx if w == 1]
    years_5y = [y for w, y in spx if w == 5]
    assert min(years_1y) == 2000 and max(years_1y) == 2025
    assert max(years_5y) == 2021
