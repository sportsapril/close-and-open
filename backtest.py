"""Backtest: buy at close, sell at next open ("overnight") vs. buy at open,
sell at close ("intraday") vs. plain buy-and-hold, on SPY daily OHLC data.

Data: data/SPY.csv (public daily OHLCV, sourced from a GitHub-hosted dataset
since this sandbox's network policy blocks direct access to market-data
providers such as Yahoo Finance).

Usage: python backtest.py
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/SPY.csv"
HAIRCUTS_BPS = [0, 1, 2, 5, 10, 20, 50, 100]
PLOT_HAIRCUT_BPS = 5  # which HAIRCUTS_BPS entry to draw as the "net" curve
TRADING_DAYS = 252


def load_prices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found (cwd={os.getcwd()!r}); "
            "run this script from the repo root or update DATA_PATH."
        )
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y")
    df = df.sort_values("DateTime").set_index("DateTime")
    df = df[["Open", "Close"]]

    # The source CSV is fetched in chunks (newest-year-first) and is not
    # globally sorted on disk; the sort above is load-bearing, not
    # cosmetic. These assertions guard against that assumption silently
    # breaking, and against corrupt rows (zero/negative prices).
    assert df.index.is_monotonic_increasing, "prices are not sorted by date"
    assert not df.index.duplicated().any(), "duplicate dates in price data"
    assert (df[["Open", "Close"]] > 0).all().all(), "non-positive price in data"
    return df


def annualize_return(equity: pd.Series) -> float:
    final = equity.iloc[-1]
    if final <= 0:
        return -1.0
    n_days = len(equity)
    return final ** (TRADING_DAYS / n_days) - 1


def annualize_vol(daily_ret: pd.Series) -> float:
    return daily_ret.std() * TRADING_DAYS ** 0.5


def summarize(name: str, daily_ret: pd.Series) -> dict:
    equity = (1 + daily_ret).cumprod()
    ann_ret = annualize_return(equity)
    ann_vol = annualize_vol(daily_ret)
    return {
        "strategy": name,
        "total_return": equity.iloc[-1] - 1,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol else float("nan"),
    }


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    overnight_ret = prices["Open"] / prices["Close"].shift(1) - 1
    intraday_ret = prices["Close"] / prices["Open"] - 1
    daily_ret = prices["Close"] / prices["Close"].shift(1) - 1
    return pd.DataFrame({
        "overnight": overnight_ret,
        "intraday": intraday_ret,
        "buy_and_hold": daily_ret,
    }).dropna()


def haircut_sweep(rets: pd.DataFrame, bh_ann_ret: float) -> list[dict]:
    # One round trip (buy + sell) per day, cost expressed in bps of
    # notional, subtracted from that day's overnight return. bps=0 is
    # deliberately included as the sweep's own gross baseline, so callers
    # shouldn't also print a separate "overnight, gross" row.
    rows = []
    for bps in HAIRCUTS_BPS:
        net_ret = rets["overnight"] - bps / 10000
        row = summarize(f"overnight net of {bps}bps/day", net_ret)
        row["beats_buy_and_hold"] = row["ann_return"] > bh_ann_ret
        rows.append(row)
    return rows


def print_summary(n_days: int, rows: list[dict]) -> None:
    print(f"\n=== SPY, {n_days} trading days ===")
    header = f"{'strategy':<28}{'total_ret':>12}{'ann_ret':>10}{'ann_vol':>10}{'sharpe':>8}{'beats B&H':>11}"
    print(header)
    for row in rows:
        beats = row.get("beats_buy_and_hold", "")
        beats_str = "" if beats == "" else ("yes" if beats else "no")
        print(
            f"{row['strategy']:<28}"
            f"{row['total_return']*100:>11.1f}%"
            f"{row['ann_return']*100:>9.2f}%"
            f"{row['ann_vol']*100:>9.2f}%"
            f"{row['sharpe']:>8.2f}"
            f"{beats_str:>11}"
        )


def plot_equity_curves(rets: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    (1 + rets["buy_and_hold"]).cumprod().plot(ax=ax, label="Buy & hold")
    (1 + rets["overnight"]).cumprod().plot(ax=ax, label="Overnight, gross (close->open)")
    (1 + rets["overnight"] - PLOT_HAIRCUT_BPS / 10000).cumprod().plot(
        ax=ax, label=f"Overnight, net of {PLOT_HAIRCUT_BPS}bps/day"
    )
    (1 + rets["intraday"]).cumprod().plot(ax=ax, label="Intraday (open->close)")
    ax.set_yscale("log")
    ax.set_title("SPY: growth of $1, overnight vs. intraday vs. buy & hold")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend()
    fig.tight_layout()
    path = "overnight_backtest_SPY.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    prices = load_prices(DATA_PATH)
    rets = build_returns(prices)

    bh_row = summarize("buy_and_hold", rets["buy_and_hold"])
    rows = [
        bh_row,
        summarize("intraday (open->close)", rets["intraday"]),
    ]
    rows += haircut_sweep(rets, bh_row["ann_return"])
    print_summary(len(rets), rows)

    path = plot_equity_curves(rets)
    print(f"\nSaved chart to {path}")


if __name__ == "__main__":
    main()
