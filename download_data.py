#!/usr/bin/env python3
"""Download daily OHLC history for the Russell 1000 universe via yfinance.

Universe: current iShares IWB (Russell 1000 ETF) holdings, fetched from
ishares.com. Falls back to the Wikipedia S&P 500 constituent table if the
IWB download fails. Note this is the *current* membership either way —
point-in-time constituent history isn't freely available, so the backtest
built on top of this carries survivorship bias.

Prices are downloaded per ticker with auto_adjust=True (split- and
dividend-adjusted). Unadjusted opens/closes would put fake +/-50% overnight
returns at every split, and dividends accrue at the ex-date open, so the
adjusted series credits the overnight leg correctly.

Output layout:
    data/universe.csv          ticker, name, sector (the universe actually used)
    data/prices/{TICKER}.csv   Date, Open, Close (ISO dates, ascending)

Existing per-ticker files are skipped, so an interrupted run resumes where
it left off. Usage:
    python download_data.py             # full universe
    python download_data.py --test 10   # first 10 tickers only
"""

import argparse
import io
import logging
import os
import time

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

PRICES_DIR = os.path.join("data", "prices")
UNIVERSE_PATH = os.path.join("data", "universe.csv")


def _normalize_ticker(ticker: str) -> str:
    # Yahoo uses dashes where index vendors use dots (BRK.B -> BRK-B).
    return ticker.strip().replace(".", "-")


def fetch_russell_1000() -> pd.DataFrame:
    """Fetch current Russell 1000 members from the iShares IWB holdings CSV."""
    resp = requests.get(IWB_HOLDINGS_URL, headers=UA_HEADERS, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    # The file leads with fund metadata; the holdings table starts at the
    # header line beginning with "Ticker".
    try:
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("Ticker"))
    except StopIteration:
        raise ValueError("IWB holdings CSV: no 'Ticker' header line found")
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), on_bad_lines="skip")
    df = df[df["Asset Class"] == "Equity"]
    df = df[df["Ticker"].notna() & (df["Ticker"] != "--")]
    out = pd.DataFrame({
        "ticker": df["Ticker"].map(_normalize_ticker),
        "name": df["Name"].str.strip(),
        "sector": df.get("Sector", pd.Series(dtype=str)),
    }).drop_duplicates("ticker")
    if len(out) < 800:  # sanity: Russell 1000 should be ~1000 names
        raise ValueError(f"IWB holdings parse looks wrong: only {len(out)} equities")
    logger.info("Universe: %d Russell 1000 members from iShares IWB", len(out))
    return out


def fetch_sp500_fallback() -> pd.DataFrame:
    """Fallback universe: S&P 500 constituents from Wikipedia."""
    resp = requests.get(WIKI_SP500_URL, headers=UA_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = next(t for t in tables if "Symbol" in t.columns)
    out = pd.DataFrame({
        "ticker": table["Symbol"].map(_normalize_ticker),
        "name": table["Security"].str.strip(),
        "sector": table.get("GICS Sector", pd.Series(dtype=str)),
    }).drop_duplicates("ticker")
    logger.info("Universe: %d S&P 500 members from Wikipedia (fallback)", len(out))
    return out


def get_universe() -> pd.DataFrame:
    try:
        return fetch_russell_1000()
    except Exception as e:
        logger.warning("iShares IWB fetch failed (%s); falling back to S&P 500", e)
        return fetch_sp500_fallback()


def download_ticker(ticker: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame | None:
    """Full adjusted daily history for one ticker, or None if unavailable."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if hist.empty:
                logger.warning("%s: no data", ticker)
                return None
            out = hist.reset_index()[["Date", "Open", "Close"]].copy()
            out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.strftime("%Y-%m-%d")
            # Adjusted history can contain zero opens on ancient rows; the
            # backtest asserts positivity, so drop them here.
            out = out[(out["Open"] > 0) & (out["Close"] > 0)]
            return out
        except Exception as e:
            wait = backoff * 2 ** attempt
            logger.warning("%s: attempt %d failed (%s); retrying in %.0fs",
                           ticker, attempt + 1, e, wait)
            time.sleep(wait)
    logger.error("%s: giving up after %d attempts", ticker, retries)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=int, metavar="N", help="only first N tickers")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="seconds between requests (rate limiting)")
    args = parser.parse_args()

    os.makedirs(PRICES_DIR, exist_ok=True)
    universe = get_universe()
    universe.to_csv(UNIVERSE_PATH, index=False)
    logger.info("Wrote %s (%d tickers)", UNIVERSE_PATH, len(universe))

    tickers = universe["ticker"].tolist()
    if args.test:
        tickers = tickers[: args.test]
        logger.info("TEST MODE: %s", tickers)

    done = skipped = failed = 0
    for ticker in tickers:
        path = os.path.join(PRICES_DIR, f"{ticker}.csv")
        if os.path.exists(path):
            skipped += 1
            continue
        data = download_ticker(ticker)
        if data is None or len(data) < 60:  # under ~3 months of history is useless
            failed += 1
        else:
            data.to_csv(path, index=False)
            done += 1
            if done % 50 == 0:
                logger.info("progress: %d downloaded, %d skipped, %d failed",
                            done, skipped, failed)
        time.sleep(args.delay)

    logger.info("Finished: %d downloaded, %d already present, %d failed/empty",
                done, skipped, failed)


if __name__ == "__main__":
    main()
