#!/usr/bin/env python3
"""Download daily OHLC history for the Russell 1000 universe via yfinance.

Universe: the Russell 1000 components table on Wikipedia (~1000 rows;
the iShares IWB holdings CSV endpoint now serves bot-detection HTML, so
Wikipedia is the primary source). Falls back to the Wikipedia S&P 500
constituent table. Note this is the *current* membership either way —
point-in-time constituent history isn't freely available, so the backtest
built on top of this carries survivorship bias.

Prices come straight from Yahoo Finance's v8 chart API (yfinance's own
backend; the library itself is unusable behind this sandbox's TLS proxy
because its curl_cffi browser-impersonation handshake gets reset). Both
Open and Close are scaled by adjclose/close, i.e. split- AND
dividend-adjusted: unadjusted opens/closes would put fake +/-50% overnight
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

WIKI_RUSSELL_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

PRICES_DIR = os.path.join("data", "prices")
UNIVERSE_PATH = os.path.join("data", "universe.csv")


def _normalize_ticker(ticker: str) -> str:
    # Yahoo uses dashes where index vendors use dots (BRK.B -> BRK-B).
    return ticker.strip().replace(".", "-")


def fetch_russell_1000() -> pd.DataFrame:
    """Fetch current Russell 1000 members from the Wikipedia components table."""
    resp = requests.get(WIKI_RUSSELL_URL, headers=UA_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    table = next(t for t in tables if "Symbol" in t.columns and len(t) > 500)
    table = table.dropna(subset=["Symbol", "Company"])
    out = pd.DataFrame({
        "ticker": table["Symbol"].map(_normalize_ticker),
        "name": table["Company"].str.strip(),
        "sector": table.get("GICS Sector", pd.Series(dtype=str)),
    }).drop_duplicates("ticker")
    if len(out) < 800:  # sanity: Russell 1000 should be ~1000 names
        raise ValueError(f"components parse looks wrong: only {len(out)} rows")
    logger.info("Universe: %d Russell 1000 members from Wikipedia", len(out))
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
        logger.warning("Russell 1000 fetch failed (%s); falling back to S&P 500", e)
        return fetch_sp500_fallback()


CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def download_ticker(ticker: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame | None:
    """Full adjusted daily history for one ticker, or None if unavailable.

    Uses explicit period1/period2 epoch params: range=max silently degrades
    to 3-month bars, while explicit periods return true daily granularity.
    """
    params = {"period1": 0, "period2": int(time.time()), "interval": "1d"}
    for attempt in range(retries):
        try:
            resp = requests.get(CHART_URL.format(ticker=ticker), params=params,
                                headers=UA_HEADERS, timeout=30)
            if resp.status_code == 404:
                logger.warning("%s: unknown to Yahoo (404)", ticker)
                return None
            resp.raise_for_status()
            result = resp.json()["chart"]["result"][0]
            if result["meta"].get("dataGranularity") != "1d":
                raise ValueError(f"granularity {result['meta'].get('dataGranularity')!r} != 1d")
            quote = result["indicators"]["quote"][0]
            adjclose = result["indicators"]["adjclose"][0]["adjclose"]
            dates = (pd.to_datetime(result["timestamp"], unit="s", utc=True)
                     .tz_convert(result["meta"]["exchangeTimezoneName"])
                     .strftime("%Y-%m-%d"))
            out = pd.DataFrame({
                "Date": dates,
                "Open": quote["open"],
                "Close": quote["close"],
                "AdjClose": adjclose,
            }).dropna()
            # Scale Open by the same split+dividend factor as Close, then
            # keep only the adjusted columns under the standard names.
            out["Open"] = out["Open"] * out["AdjClose"] / out["Close"]
            out["Close"] = out["AdjClose"]
            out = out[(out["Open"] > 0) & (out["Close"] > 0)]
            out = out.drop_duplicates("Date")[["Date", "Open", "Close"]]
            if out.empty:
                logger.warning("%s: no usable rows", ticker)
                return None
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
