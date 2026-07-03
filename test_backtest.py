"""Minimal regression tests for backtest.py.

Usage: python -m pytest test_backtest.py -q  (or: python test_backtest.py)
"""

import math
import os
import tempfile

import pandas as pd
import pytest

from backtest import annualize_return, build_returns, load_prices, summarize


def _write_csv(path, rows):
    df = pd.DataFrame(rows, columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df.to_csv(path, index=False)


def test_load_prices_sorts_unsorted_input():
    # Regression test for the real bug in data/SPY.csv: the source file is
    # fetched in newest-year-first chunks and is not globally sorted on
    # disk. load_prices must not depend on the CSV already being ordered.
    rows = [
        ["01/03/2024", 10, 11, 9, 10.5, 100],
        ["01/02/2024", 9, 10, 8, 9.5, 100],
        ["01/01/2024", 8, 9, 7, 8.5, 100],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prices.csv")
        _write_csv(path, rows)
        prices = load_prices(path)
        assert prices.index.is_monotonic_increasing
        assert list(prices["Open"]) == [8, 9, 10]


def test_load_prices_rejects_duplicate_dates():
    rows = [
        ["01/01/2024", 8, 9, 7, 8.5, 100],
        ["01/01/2024", 8, 9, 7, 8.5, 100],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prices.csv")
        _write_csv(path, rows)
        with pytest.raises(AssertionError):
            load_prices(path)


def test_load_prices_rejects_non_positive_price():
    rows = [
        ["01/01/2024", 8, 9, 7, 8.5, 100],
        ["01/02/2024", 0, 1, -1, 0.5, 100],
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prices.csv")
        _write_csv(path, rows)
        with pytest.raises(AssertionError):
            load_prices(path)


def test_load_prices_missing_file_raises_friendly_error():
    with pytest.raises(FileNotFoundError):
        load_prices("does/not/exist.csv")


def test_annualize_return_known_value():
    # +100% total return over exactly one year (252 trading days) should
    # annualize to +100%.
    equity = pd.Series([2.0] * 252)
    assert math.isclose(annualize_return(equity), 1.0, rel_tol=1e-9)


def test_annualize_return_handles_wipeout():
    # Equity that has gone to (or below) zero must not silently produce
    # nan/complex output via a fractional power of a non-positive number.
    equity = pd.Series([1.0, 0.5, 0.0])
    assert annualize_return(equity) == -1.0

    equity_negative = pd.Series([1.0, -0.5])
    assert annualize_return(equity_negative) == -1.0


def test_build_returns_reconstructs_buy_and_hold():
    prices = pd.DataFrame(
        {"Open": [10, 11, 9, 12], "Close": [10.5, 9.5, 11.5, 12.5]},
        index=pd.date_range("2024-01-01", periods=4),
    )
    rets = build_returns(prices)
    reconstructed = (1 + rets["overnight"]) * (1 + rets["intraday"]) - 1
    assert (reconstructed - rets["buy_and_hold"]).abs().max() < 1e-12


def test_summarize_zero_vol_gives_nan_sharpe():
    daily_ret = pd.Series([0.0, 0.0, 0.0])
    row = summarize("flat", daily_ret)
    assert row["ann_vol"] == 0.0
    assert math.isnan(row["sharpe"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
