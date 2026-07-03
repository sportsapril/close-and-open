# close-and-open

Is "buy at today's close, sell at tomorrow's open" a strategy that beats the
market? This repo has a literature review and a real, runnable backtest.

**Short answer:** the overnight (close-to-open) return premium is real and
well documented — but it's not free money. Whether it "beats the market"
depends heavily on the sample period, and even a tiny per-trade cost erases
the edge. See the numbers below.

## What the literature says

1. **The pattern is real and large, gross of costs.** AQR's study of the
   S&P 500 from 1990–2020 found overnight returns averaging ~0.04%/day while
   intraday returns hovered near zero; from 1993–2013 close-to-open returns
   accounted for nearly all of the index's cumulative gain.

2. **[Lou, Polk & Skouras (2019), "A Tug of War: Overnight versus Intraday
   Expected Returns"](https://personal.lse.ac.uk/polk/research/TugOfWar.pdf)**
   (*Journal of Financial Economics*) is the leading academic account.
   Overnight returns predict future overnight returns and *reverse* future
   intraday returns (and vice versa). They attribute this to a "tug of war"
   between institutional order flow concentrated near the close and retail
   order flow concentrated near the open. Momentum profits accrue almost
   entirely overnight; several other factor premia (value, profitability,
   investment) accrue intraday instead.

3. **[Boyarchenko, Larsen & Whelan, "The Overnight Drift"](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf)**
   (NY Fed Staff Report) ties the overnight premium to dealer/intermediary
   inventory-risk management around the close.

4. **Fringe / non-consensus explanation:** Bruce Knuteson's
   ["Strikingly Suspicious Overnight and Intraday Returns" (2020)](https://arxiv.org/abs/2010.01727)
   and its follow-up
   ["They Still Haven't Told You" (2022)](https://arxiv.org/abs/2201.00223)
   argue the pattern is *too* clean to be a normal risk premium and
   speculate it reflects the daily rebalancing footprint of one or more
   large quant funds. This is provocative but **not** a mainstream-accepted
   explanation — treat it as a hypothesis, not settled fact.

5. **Why it's hard to actually capture:**
   [AlphaArchitect, "Trading Costs Wipe Out the Overnight Return Anomaly"](https://alphaarchitect.com/trading-costs-wipe-out-the-overnight-return-anomaly/)
   and [Elm Wealth, "Night Moves"](https://elmwealth.com/night-moves-overnight-drift/)
   both find that once bid-ask spread, commissions, and price impact are
   included, the statistically significant edge shrinks dramatically —
   closer to a random walk than a reliably repeatable strategy. Price
   impact alone (trading ~1% of a stock's daily volume in a name with 2%
   daily volatility) can cost on the order of 40bps round-trip, which is
   large relative to the average overnight edge.

6. **Practical execution point:** you can't literally transact at the
   tape-printed close/open price for free. The realistic proxy is a
   Market-on-Close buy paired with a Market-on-Open sell (or the reverse),
   each of which carries its own spread/impact cost — hence the haircut
   sweep below, rather than assuming free execution at the printed price.

## Empirical backtest (this repo)

`backtest.py` loads `data/SPY.csv` (daily OHLC, January 2000 – March 2026) and
decomposes each day's return into an **overnight** leg (`Open_t /
Close_{t-1} - 1`) and an **intraday** leg (`Close_t / Open_t - 1`), which by
construction multiply back out to the buy-and-hold return. It then applies a
flat per-round-trip cost ("haircut," in bps of notional) to the overnight
leg and reports where the strategy stops beating buy-and-hold.

> Data note: this sandbox's network policy blocks direct access to market
> data providers (Yahoo Finance, Stooq, etc.), so the script uses a vendored,
> publicly available SPY OHLC dataset rather than a live `yfinance` pull.
> It's real historical data (you can see the dot-com crash, 2008 GFC,
> 9/11 reopening gap, and the March 2020 COVID crash in it), just not fetched
> live. It is **not** dividend-adjusted, so the buy-and-hold figure below is
> a pure price return and understates SPY's true total return by roughly
> SPY's dividend yield (~1.3%/year) compounded over 26 years — a bias that,
> if corrected, would make the overnight strategy's underperformance below
> even larger, not smaller.

Run it yourself: `pip install -r requirements.txt && python backtest.py`

Run the test suite: `pip install -r requirements-dev.txt && python -m pytest test_backtest.py -q`

Results, SPY, 2000-01-03 through 2026-03-20 (6,592 trading days):

| strategy | total return | annualized return | annualized vol | Sharpe | beats buy&hold? |
|---|---:|---:|---:|---:|:---:|
| buy & hold | 345.9% | 5.88% | 19.42% | 0.30 | — |
| overnight, gross (close→open) | 231.8% | 4.69% | 11.26% | 0.42 | no |
| intraday (open→close) | 34.4% | 1.14% | 15.58% | 0.07 | no |
| overnight, net of 1bp/day | 71.6% | 2.09% | 11.26% | 0.19 | no |
| overnight, net of 2bp/day | -11.2% | -0.45% | 11.26% | -0.04 | no |
| overnight, net of 5bp/day | -87.7% | -7.70% | 11.26% | -0.68 | no |
| overnight, net of 10bp/day | -99.5% | -18.64% | 11.26% | -1.66 | no |

![Overnight vs intraday vs buy & hold equity curves](overnight_backtest_SPY.png)

## Universe rolling-window backtest (Russell 1000)

Beyond the single-index test above, the repo can ask the broader question:
**how many individual stocks could the overnight strategy have used to beat
the SPX index return — in which periods, at which window sizes, and at
which cost levels?**

Pipeline:

1. `python download_data.py` — fetches the current Russell 1000 membership
   (iShares IWB holdings; falls back to the Wikipedia S&P 500 list) into
   `data/universe.csv`, then downloads each ticker's full daily history via
   yfinance into `data/prices/{TICKER}.csv` (split- and dividend-adjusted;
   resumable; `--test N` for a quick run). Requires internet access to
   ishares.com / finance.yahoo.com.
2. `python rolling_backtest.py` — for every stock, every rolling calendar
   window of 1–5 years (annual stride, ≥95% trading-day coverage required
   of both the stock and the benchmark), and every haircut in
   `HAIRCUTS_BPS`, compares the overnight strategy's annualized return
   against SPY buy-and-hold over the same window. Writes a long-format
   results table, a summary CSV, and two charts under `results/`.

Known limitations (read before quoting numbers):

- **Survivorship bias**: the universe is *today's* Russell 1000 members.
  Stocks that delisted or fell out of the index are absent, which inflates
  the fraction of "winners."
- The SPY benchmark here is a price return (the vendored CSV is not
  dividend-adjusted) while the stock data is total-return-adjusted; this
  slightly flatters the strategy vs. the real SPX total return.
- Current constituents' *history* includes years before they joined the
  index (fine for "could this stock have beaten SPX," but not an
  index-membership-aware simulation).

### Results

Run: 998 of 1,002 current Russell 1000 members downloaded (full available
history per ticker, some back to 1970; 4 tickers unavailable on Yahoo),
978 tickers with enough coverage for at least one window, 704,424
(ticker, window, haircut) combinations, benchmark windows 2000–2025.

**% of stock-windows where the overnight strategy beat SPX buy-and-hold:**

| window | 0bps | 1bps | 2bps | 5bps | 10bps | 20bps | 50bps | 100bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1y | 45.1 | 40.6 | 36.7 | 25.8 | 13.4 | 3.8 | 0.5 | 0.1 |
| 2y | 45.6 | 39.4 | 33.9 | 20.4 | 8.6 | 2.2 | 0.3 | 0.1 |
| 3y | 45.0 | 37.9 | 31.1 | 17.0 | 6.6 | 1.6 | 0.2 | 0.0 |
| 4y | 44.7 | 36.8 | 29.4 | 14.8 | 5.6 | 1.3 | 0.2 | 0.0 |
| 5y | 44.5 | 35.8 | 28.4 | 13.8 | 4.9 | 1.2 | 0.1 | 0.0 |

**Number of stocks (of 978) that beat SPX in a *majority* of their windows:**

| window | 0bps | 1bps | 2bps | 5bps | 10bps | 20bps |
|---|---:|---:|---:|---:|---:|---:|
| 1y | 295 | 203 | 151 | 52 | 13 | 2 |
| 2y | 377 | 266 | 172 | 56 | 15 | 7 |
| 3y | 348 | 258 | 165 | 54 | 11 | 4 |
| 4y | 364 | 272 | 183 | 57 | 13 | 4 |
| 5y | 349 | 247 | 178 | 56 | 17 | 4 |

![% of stock-windows beating SPX, by window size and haircut](results/pct_beating_heatmap.png)

![% of stocks beating SPX by window start year](results/pct_beating_timeseries.png)

What the numbers say:

1. **Gross of costs it's a worse-than-coin-flip bet.** Even with zero
   trading costs, only ~45% of stock-windows beat SPX, and only ~30–38%
   of stocks won a majority of their windows.
2. **Costs are the story, again.** At 5bps/day round trip, ~5% of stocks
   (52–57 of 978) still beat SPX in a majority of windows; at 20bps/day
   it is essentially zero (2–7 stocks). This is the single-index result
   from the section above, replicated a thousand times.
3. **Shorter windows look better only because they're noisier.** 1y
   windows beat SPX in 25.8% of cases at 5bps vs. 13.8% for 5y windows —
   dispersion, not persistence: hold the strategy longer and the cost
   drag reliably wins.
4. **The "winners" are momentum rockets, not overnight-anomaly stocks.**
   The top names by fraction of winning windows (RKLB, VKTX, ASTS, MU,
   AMD, APP...) are explosive stocks whose *any* long exposure —
   including plain buy-and-hold — trounced SPX. The overnight mechanic
   isn't the source of their edge; picking them in advance is the hard
   part.
5. **The edge is concentrated in the past and in bear markets.** Hit
   rates were 30–65% for windows starting 2000–2008 and mostly 5–20%
   after 2009, spiking only when SPX itself fell (2008, 2018, 2022) —
   a half-volatility strategy "beats the index" mainly when the index
   goes down. Restricting to windows starting 2015+ at 5bps leaves only
   ~40–60 majority-winners out of 978, most of them the same momentum
   names.

## Verdict

Over this ~26-year SPY sample, the overnight leg captured about two-thirds
of the market's total return (231.8% vs. 345.9%) with only about *half* the
annualized volatility of the intraday leg and a meaningfully better Sharpe
ratio (0.42 vs. 0.30 for buy-and-hold) — so the underlying pattern in the
literature clearly shows up in the data. **But it did not outright beat
buy-and-hold on raw return, even before any trading costs**, and a haircut
as small as ~2bps per round trip (a tighter cost than many retail traders
will realistically achieve on a stock, though achievable on SPY itself with
a good broker) is enough to erase the entire edge and turn it solidly
negative.

Your skepticism is well founded:
- The anomaly is real, replicated across decades and studies, and not a
  myth — but "captures nearly all the market's gains" is a period-specific
  headline stat (notably from the 1993–2013 AQR sample), not a law of
  nature. Extend the window and it can lag buy-and-hold outright.
- Whatever edge remains gross of costs is thin enough that ordinary
  transaction costs (spread + commission + price impact from an actual MOC
  buy / MOO sell) plausibly wipe it out, consistent with the AlphaArchitect
  and Elm Wealth findings cited above.
- It's a real, published academic phenomenon (Lou-Polk-Skouras, the NY Fed
  paper) with a debated cause, not a settled "free lunch" — do not size a
  live strategy off the gross numbers without your own costed backtest on
  the specific instrument and execution method you plan to use.
