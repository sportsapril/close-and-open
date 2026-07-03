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
                         haircuts_bps=HAIRCUTS_BPS) -> list[dict]:
    """All (window, haircut) rows for one ticker's overnight return series."""
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
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.prices_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no price CSVs in {args.prices_dir!r}; run download_data.py first")
    os.makedirs(args.out, exist_ok=True)

    spx_ann = benchmark_windows(args.spy, args.min_coverage)
    print(f"Benchmark: {len(spx_ann)} valid SPY windows; universe: {len(files)} tickers")

    all_rows = []
    skipped = []
    for i, path in enumerate(files, 1):
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            overnight = build_returns(load_prices(path))["overnight"]
        except (AssertionError, ValueError, KeyError) as e:
            skipped.append((ticker, str(e)))
            continue
        all_rows += window_returns_table(overnight, spx_ann, ticker, args.min_coverage)
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

    # % of stock-windows beating SPX, per (window size, haircut)
    pct = (results.groupby(["window_years", "haircut_bps"])["beats_spx"]
           .mean().mul(100).unstack())
    summary = pct.round(1)
    summary.to_csv(os.path.join(args.out, "summary_by_window_haircut.csv"))
    print("\n% of stock-windows where overnight strategy beats SPX buy-and-hold:")
    print(summary.to_string())

    plot_heatmap(pct, os.path.join(args.out, "pct_beating_heatmap.png"))

    at_h = results[results["haircut_bps"] == args.plot_haircut]
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
