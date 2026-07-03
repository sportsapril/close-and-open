"""Backtest: buy at close, sell at next open ("overnight") vs. buy at open,
sell at close ("intraday") vs. plain buy-and-hold, on SPY daily OHLC data.

Data: data/SPY.csv (public daily OHLCV, sourced from a GitHub-hosted dataset
since this sandbox's network policy blocks direct access to market-data
providers such as Yahoo Finance).

Usage: python backtest.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/SPY.csv"
HAIRCUTS_BPS = [0, 1, 2, 5, 10, 20, 50, 100]
TRADING_DAYS = 252


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%m/%d/%Y")
    df = df.sort_values("DateTime").set_index("DateTime")
    return df[["Open", "Close"]]


def annualize_return(equity: pd.Series) -> float:
    n_days = len(equity)
    return equity.iloc[-1] ** (TRADING_DAYS / n_days) - 1


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


def haircut_sweep(rets: pd.DataFrame) -> list[dict]:
    # One round trip (buy + sell) per day, cost expressed in bps of
    # notional, subtracted from that day's overnight return.
    bh_ann_ret = annualize_return((1 + rets["buy_and_hold"]).cumprod())
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
    (1 + rets["overnight"] - 0.0005).cumprod().plot(ax=ax, label="Overnight, net of 5bps/day")
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

    rows = [
        summarize("buy_and_hold", rets["buy_and_hold"]),
        summarize("overnight (close->open)", rets["overnight"]),
        summarize("intraday (open->close)", rets["intraday"]),
    ]
    rows += haircut_sweep(rets)
    print_summary(len(rets), rows)

    path = plot_equity_curves(rets)
    print(f"\nSaved chart to {path}")


if __name__ == "__main__":
    main()
