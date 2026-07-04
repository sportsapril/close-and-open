#!/usr/bin/env python3
"""Rolling-window overnight-strategy backtest across a stock universe vs SPX.

For every stock CSV in --prices-dir, every rolling calendar window (sizes
--windows years, annual stride), and every haircut in HAIRCUTS_BPS, this
computes the annualized return of the overnight strategy (buy close, sell
next open, net of the haircut) and compares it against SPY buy-and-hold
annualized return over the same window. A window only counts when both the
stock and SPY have at least --min-coverage of the expected trading days.

Outputs (under --out):
    rolling_results.csv.gz            one row per (ticker, window, haircut)
    summary_by_window_haircut.csv     % of stock-windows beating SPX
    pct_beating_heatmap.png           window size x haircut
    pct_beating_timeseries.png        % beating by window start year

Usage: python rolling_backtest.py [--prices-dir data/prices] [--out results]
"""

import argparse
import glob
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from backtest import HAIRCUTS_BPS, TRADING_DAYS, annualize_return, build_returns, load_prices

WINDOW_YEARS = [1, 2, 3, 4, 5]
MIN_COVERAGE = 0.95

# Validated categorical palette (fixed slot order) + single-hue sequential
# ramp for the heatmap; see the dataviz palette reference.
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)


def build_wait_overnight(prices: pd.DataFrame, x: float) -> pd.Series:
    """Overnight returns with a gap-down-wait exit.

    Buy at every close. Next day: if the open is at or above the buy price,
    sell at the open (the plain overnight strategy). If the stock gapped
    down, wait, and assume the exit captures fraction x of the day's
    recovery off the open: sell at Open + x*(High - Open).

    x=0 is exactly the plain overnight strategy. x uses the day's High,
    which is unknowable in real time — treat results as an upper-bound
    diagnostic, not a tradeable backtest.
    """
    prev_close = prices["Close"].shift(1)
    # A handful of vendored rows have High < Open; a negative "recovery"
    # is meaningless, so clip it to zero (sell at the open).
    recovery = (prices["High"] - prices["Open"]).clip(lower=0)
    sell = prices["Open"].where(prices["Open"] >= prev_close,
                                prices["Open"] + x * recovery)
    return (sell / prev_close - 1).dropna()


def generate_windows(index: pd.DatetimeIndex, window_years: int,
                     min_coverage: float = MIN_COVERAGE) -> list[tuple]:
    """Calendar windows [Jan 1 Y, Jan 1 Y+W) with annual stride.

    Returns (start_year, start_ts, end_ts, n_days) for every window where
    the index covers at least min_coverage of the expected trading days.
    """
    if len(index) == 0:
        return []
    windows = []
    expected = TRADING_DAYS * window_years
    for year in range(index.min().year, index.max().year - window_years + 2):
        start = pd.Timestamp(year, 1, 1)
        end = pd.Timestamp(year + window_years, 1, 1)
        n_days = int(((index >= start) & (index < end)).sum())
        if n_days >= min_coverage * expected:
            windows.append((year, start, end, n_days))
    return windows


def window_returns_table(overnight: pd.Series, spx_ann_by_window: dict,
                         ticker: str, min_coverage: float,
                         haircuts_bps=HAIRCUTS_BPS,
                         extra: dict | None = None) -> list[dict]:
    """All (window, haircut) rows for one ticker's overnight return series.

    `extra` fields (e.g. the exit-model parameter) are copied into every row.
    """
    rows = []
    for w in WINDOW_YEARS:
        for year, start, end, n_days in generate_windows(overnight.index, w, min_coverage):
            if (w, year) not in spx_ann_by_window:
                continue  # benchmark itself lacks coverage for this window
            spx_ann = spx_ann_by_window[(w, year)]
            seg = overnight[(overnight.index >= start) & (overnight.index < end)]
            for bps in haircuts_bps:
                net = seg - bps / 10000
                ann = annualize_return((1 + net).cumprod())
                rows.append({
                    "ticker": ticker,
                    "window_years": w,
                    "window_start": year,
                    "n_days": n_days,
                    "haircut_bps": bps,
                    "strategy_ann_ret": ann,
                    "spx_ann_ret": spx_ann,
                    "beats_spx": ann > spx_ann,
                    **(extra or {}),
                })
    return rows


def benchmark_windows(spy_path: str, min_coverage: float) -> dict:
    """SPY buy-and-hold annualized return for every valid (size, year) window."""
    bh = build_returns(load_prices(spy_path))["buy_and_hold"]
    out = {}
    for w in WINDOW_YEARS:
        for year, start, end, _ in generate_windows(bh.index, w, min_coverage):
            seg = bh[(bh.index >= start) & (bh.index < end)]
            out[(w, year)] = annualize_return((1 + seg).cumprod())
    return out


def plot_heatmap(pct: pd.DataFrame, path: str) -> None:
    """pct: index=window_years, columns=haircut_bps, values=% beating SPX."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(pct.values, cmap=SEQ_CMAP, vmin=0, vmax=max(50, pct.values.max()),
                   aspect="auto")
    ax.set_xticks(range(len(pct.columns)), [f"{c}bps" for c in pct.columns])
    ax.set_yticks(range(len(pct.index)), [f"{w}y" for w in pct.index])
    ax.set_xlabel("Haircut per round trip")
    ax.set_ylabel("Window size")
    ax.set_title("Overnight strategy: % of stock-windows beating SPX buy-and-hold")
    mid = (pct.values.max() + pct.values.min()) / 2
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color="#0b0b0b" if v < mid else "#fcfcfb")
    fig.colorbar(im, ax=ax, label="% beating SPX")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries(by_year: pd.DataFrame, plot_haircut: int, path: str) -> None:
    """by_year: index=window_start, columns=window_years, values=% beating."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, w in enumerate(by_year.columns):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(by_year.index, by_year[w], label=f"{w}y window",
                color=color, linewidth=2)
        last = by_year[w].dropna()
        if len(last):
            ax.annotate(f"{w}y", (last.index[-1], last.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center")
    ax.set_xlabel("Window start year")
    ax.set_ylabel("% of stocks beating SPX")
    ax.set_title(f"Overnight strategy net of {plot_haircut}bps/day: "
                 "% of stocks beating SPX, by window start")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_wait(pct: pd.DataFrame, window_years: int, path: str) -> None:
    """pct: index=haircut_bps, columns=exit_x (%), values=% beating SPX."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = range(len(pct.index))
    for i, x in enumerate(pct.columns):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        label = "sell at open (x=0)" if x == 0 else f"x={x:.0f}% of recovery"
        ax.plot(positions, pct[x], label=label, color=color,
                linewidth=2, marker="o", markersize=5)
        ax.annotate(f"{x:.0f}%", (positions[-1], pct[x].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=9, va="center")
    ax.set_xticks(list(positions), [f"{b}bps" for b in pct.index])
    ax.set_xlabel("Haircut per round trip")
    ax.set_ylabel("% of stock-windows beating SPX")
    ax.set_title(f"Gap-down-wait exit ({window_years}y windows): "
                 "% beating SPX by haircut and recovery fraction x")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices-dir", default=os.path.join("data", "prices"))
    parser.add_argument("--spy", default=os.path.join("data", "SPY.csv"))
    parser.add_argument("--out", default="results")
    parser.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    parser.add_argument("--plot-haircut", type=int, default=5,
                        help="haircut (bps) used for the time-series chart")
    parser.add_argument("--top", type=int, default=20,
                        help="how many top tickers to print")
    parser.add_argument("--exit-x", default="0",
                        help="comma-separated recovery fractions in percent "
                             "for the gap-down-wait exit (0 = plain sell-at-open)")
    parser.add_argument("--gap-plot-window", type=int, default=3,
                        help="window size (years) for the gap-wait chart")
    args = parser.parse_args()
    exit_xs = [float(v) for v in args.exit_x.split(",")]

    files = sorted(glob.glob(os.path.join(args.prices_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no price CSVs in {args.prices_dir!r}; run download_data.py first")
    os.makedirs(args.out, exist_ok=True)

    spx_ann = benchmark_windows(args.spy, args.min_coverage)
    print(f"Benchmark: {len(spx_ann)} valid SPY windows; universe: {len(files)} tickers; "
          f"exit x sweep: {exit_xs}")

    all_rows = []
    skipped = []
    for i, path in enumerate(files, 1):
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            prices = load_prices(path, columns=("Open", "High", "Close"))
        except (AssertionError, ValueError, KeyError) as e:
            skipped.append((ticker, str(e)))
            continue
        for x in exit_xs:
            overnight = build_wait_overnight(prices, x / 100)
            all_rows += window_returns_table(overnight, spx_ann, ticker,
                                             args.min_coverage,
                                             extra={"exit_x": x})
        if i % 100 == 0:
            print(f"  processed {i}/{len(files)} tickers...")

    if skipped:
        print(f"Skipped {len(skipped)} tickers with bad data: "
              f"{[t for t, _ in skipped][:10]}{'...' if len(skipped) > 10 else ''}")
    results = pd.DataFrame(all_rows)
    if results.empty:
        raise SystemExit("no valid (ticker, window) combinations — check data coverage")
    results_path = os.path.join(args.out, "rolling_results.csv.gz")
    results.to_csv(results_path, index=False, compression="gzip")

    # The legacy outputs (summary, heatmap, time series, top tickers)
    # describe the plain sell-at-open strategy: the x=0 slice.
    base = results[results["exit_x"] == 0]

    # % of stock-windows beating SPX, per (window size, haircut)
    pct = (base.groupby(["window_years", "haircut_bps"])["beats_spx"]
           .mean().mul(100).unstack())
    summary = pct.round(1)
    summary.to_csv(os.path.join(args.out, "summary_by_window_haircut.csv"))
    print("\n% of stock-windows where overnight strategy beats SPX buy-and-hold (x=0):")
    print(summary.to_string())

    plot_heatmap(pct, os.path.join(args.out, "pct_beating_heatmap.png"))

    if len(exit_xs) > 1:
        gap = (results[results["window_years"] == args.gap_plot_window]
               .groupby(["haircut_bps", "exit_x"])["beats_spx"]
               .mean().mul(100).unstack())
        gap_all = (results.groupby(["exit_x", "window_years", "haircut_bps"])["beats_spx"]
                   .mean().mul(100).round(1).unstack())
        gap_all.to_csv(os.path.join(args.out, "gap_wait_summary.csv"))
        print(f"\nGap-down-wait exit, {args.gap_plot_window}y windows, "
              "% of stock-windows beating SPX by recovery fraction x:")
        print(gap.round(1).to_string())
        plot_gap_wait(gap, args.gap_plot_window,
                      os.path.join(args.out, "gap_wait_pct_beating.png"))

    at_h = base[base["haircut_bps"] == args.plot_haircut]
    by_year = (at_h.groupby(["window_start", "window_years"])["beats_spx"]
               .mean().mul(100).unstack())
    plot_timeseries(by_year, args.plot_haircut, os.path.join(args.out, "pct_beating_timeseries.png"))

    top = (at_h.groupby("ticker")["beats_spx"]
           .agg(["mean", "count"])
           .query("count >= 5")
           .sort_values("mean", ascending=False)
           .head(args.top))
    print(f"\nTop tickers by fraction of windows beating SPX (net of {args.plot_haircut}bps/day, >=5 windows):")
    for t, row in top.iterrows():
        print(f"  {t:<8} {row['mean']*100:5.1f}% of {int(row['count'])} windows")

    print(f"\nWrote {results_path} ({len(results):,} rows), summary CSV, and 2 charts under {args.out}/")


if __name__ == "__main__":
    main()
