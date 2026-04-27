"""
First-day price scraper — yfinance.

For each ticker in the IPO calendar, downloads daily OHLC data around the
IPO date and computes:
  - ``first_day_close``  — closing price on the IPO date
  - ``underpricing``     — (first_day_close - offer_price) / offer_price
  - ``first_week_return`` — cumulative return through end of week 1
  - ``first_month_return`` — cumulative return through end of 30 calendar days

Tickers that fail (delisted, name changes, API errors) are logged and skipped.
Results are saved to ``data/raw/prices.csv``.

Usage (CLI)::

    python -m src.scraper_prices
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils import setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
IPO_CSV = RAW_DIR / "ipo_calendar.csv"
OUTPUT_CSV = RAW_DIR / "prices.csv"

WINDOW_DAYS = 45          # calendar days around IPO to download


# ---------------------------------------------------------------------------
# Per-ticker price fetch
# ---------------------------------------------------------------------------

def _trading_close_on_or_after(hist: pd.DataFrame, target_date: pd.Timestamp) -> Optional[float]:
    """Return the closing price on the first trading day >= ``target_date``.

    Args:
        hist: yfinance OHLC DataFrame with a DatetimeIndex.
        target_date: The target date (IPO date).

    Returns:
        Closing price as float, or ``None`` if no data available.
    """
    if hist.empty:
        return None
    subset = hist[hist.index >= target_date]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[0])


def _compute_return(
    hist: pd.DataFrame,
    start_date: pd.Timestamp,
    days_ahead: int,
) -> Optional[float]:
    """Compute the return from the IPO close to the close ``days_ahead`` later.

    Args:
        hist: yfinance OHLC DataFrame.
        start_date: IPO date (first trading day).
        days_ahead: Target horizon in *calendar* days.

    Returns:
        Return as a decimal (e.g. 0.05 for 5 %), or ``None`` if data missing.
    """
    if hist.empty:
        return None
    ipo_rows = hist[hist.index >= start_date]
    if ipo_rows.empty:
        return None
    ipo_close = float(ipo_rows["Close"].iloc[0])

    target_date = start_date + pd.Timedelta(days=days_ahead)
    horizon_rows = hist[hist.index >= target_date]
    if horizon_rows.empty:
        return None
    horizon_close = float(horizon_rows["Close"].iloc[0])

    if ipo_close == 0:
        return None
    return (horizon_close - ipo_close) / ipo_close


def fetch_ticker_prices(
    ticker: str,
    ipo_date: pd.Timestamp,
    offer_price: float,
) -> dict:
    """Fetch price data for one IPO and compute underpricing metrics.

    Args:
        ticker: Stock ticker symbol.
        ipo_date: IPO date as a :class:`~pandas.Timestamp`.
        offer_price: IPO offer price in USD.

    Returns:
        Dict with keys ``ticker``, ``first_day_close``, ``underpricing``,
        ``first_week_return``, ``first_month_return``, and ``status``.
    """
    result: dict = {"ticker": ticker}

    start = ipo_date - pd.Timedelta(days=3)  # a few days of slack
    end = ipo_date + pd.Timedelta(days=WINDOW_DAYS)

    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            log.warning("%s: yfinance returned no data", ticker)
            result["status"] = "no_data"
            return result

        first_close = _trading_close_on_or_after(hist, ipo_date)
        if first_close is None:
            log.warning("%s: no close on/after IPO date", ticker)
            result["status"] = "no_ipo_date_close"
            return result

        result["first_day_close"] = first_close

        if offer_price and offer_price > 0:
            result["underpricing"] = (first_close - offer_price) / offer_price
        else:
            result["underpricing"] = np.nan

        result["first_week_return"] = _compute_return(hist, ipo_date, 7)
        result["first_month_return"] = _compute_return(hist, ipo_date, 30)
        result["status"] = "ok"

    except Exception as exc:
        log.error("%s: price fetch failed — %s", ticker, exc)
        result["status"] = f"error: {exc}"

    return result



# ---------------------------------------------------------------------------
# Market index downloads (VIX + NASDAQ)
# ---------------------------------------------------------------------------

def download_market_indices(
    start: str = "2018-10-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Download daily VIX and NASDAQ Composite closing prices.

    Args:
        start: Start date string ``YYYY-MM-DD``.
        end: End date string ``YYYY-MM-DD``.

    Returns:
        DataFrame indexed by date with columns ``vix_close`` and
        ``nasdaq_close``.
    """
    log.info("Downloading VIX and NASDAQ data …")
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
        nasdaq = yf.download("^IXIC", start=start, end=end, progress=False)["Close"]

        df = pd.DataFrame({"vix_close": vix, "nasdaq_close": nasdaq})
        df.index.name = "date"
        df = df.dropna(how="all")

        out_path = RAW_DIR / "market_indices.csv"
        df.to_csv(out_path)
        log.info("Saved market indices → %s (%d rows)", out_path, len(df))
        return df
    except Exception as exc:
        log.error("Failed to download market indices: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_price_scraper(
    ipo_csv: Path = IPO_CSV,
    max_tickers: Optional[int] = None,
    sleep_between: float = 0.25,
) -> pd.DataFrame:
    """Fetch first-day prices for all tickers in the IPO calendar.

    Args:
        ipo_csv: Path to IPO calendar CSV.
        max_tickers: Optional cap for test runs.
        sleep_between: Seconds to sleep between yfinance calls (courtesy
            throttle — yfinance is not as strict as EDGAR but we are polite).

    Returns:
        DataFrame with one row per ticker containing underpricing metrics.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df_ipos = pd.read_csv(ipo_csv, parse_dates=["ipo_date"])
    df_ipos = df_ipos.dropna(subset=["ticker", "ipo_date", "offer_price"])

    if max_tickers:
        df_ipos = df_ipos.head(max_tickers)

    results = []
    for i, (_, row) in enumerate(df_ipos.iterrows()):
        ticker = str(row["ticker"]).strip().upper()
        ipo_date = pd.Timestamp(row["ipo_date"])
        offer_price = float(row.get("offer_price", 0) or 0)

        if i > 0 and i % 50 == 0:
            log.info("Progress: %d / %d tickers processed", i, len(df_ipos))

        res = fetch_ticker_prices(ticker, ipo_date, offer_price)
        results.append(res)
        time.sleep(sleep_between)

    df_prices = pd.DataFrame(results)
    df_prices.to_csv(OUTPUT_CSV, index=False)

    ok_count = (df_prices["status"] == "ok").sum() if "status" in df_prices.columns else 0
    log.info(
        "Price scrape complete: %d / %d succeeded → %s",
        ok_count, len(df_prices), OUTPUT_CSV,
    )

    # Also download market indices for feature engineering
    download_market_indices()

    return df_prices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_price_scraper()
    print(df[["ticker", "first_day_close", "underpricing", "status"]].head(20).to_string())
    print(f"\nStatus counts:\n{df['status'].value_counts().to_string()}")
