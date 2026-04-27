"""
Feature engineering for IPO underpricing prediction.

Builds three feature groups:

  - **Calendar features** — year, quarter, month, day-of-week, quarter-end
    month indicator, and days from S-1 filing to pricing.
  - **Market-regime features** — VIX at pricing, NASDAQ rolling return/
    volatility, hot-market dummy.
  - **Deal features** — log offer size, price revision, underwriter rank,
    top-tier dummy.

All functions accept and return pandas DataFrames and are designed to be
called sequentially in the notebook pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import setup_logging
from src import text_features

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame, date_col: str = "ipo_date") -> pd.DataFrame:
    """Add calendar-derived features to *df*.

    Args:
        df: IPO DataFrame with a datetime column named *date_col*.
        date_col: Name of the column holding the IPO date.

    Returns:
        Copy of *df* with additional columns:
        ``ipo_year``, ``ipo_quarter``, ``ipo_month``, ``ipo_dayofweek``,
        ``is_quarter_end_month``, ``days_from_filing_to_pricing``.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")

    df["ipo_year"] = dates.dt.year
    df["ipo_quarter"] = dates.dt.quarter
    df["ipo_month"] = dates.dt.month
    df["ipo_dayofweek"] = dates.dt.dayofweek  # Monday=0, Friday=4

    # March, June, September, December = quarter-end months
    df["is_quarter_end_month"] = dates.dt.month.isin([3, 6, 9, 12]).astype(int)

    # Days from S-1 filing date to IPO pricing date
    if "filing_date" in df.columns:
        filing_dates = pd.to_datetime(df["filing_date"], errors="coerce")
        df["days_from_filing_to_pricing"] = (dates - filing_dates).dt.days
    else:
        df["days_from_filing_to_pricing"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Market-regime features
# ---------------------------------------------------------------------------

def add_market_features(
    df: pd.DataFrame,
    market_csv: Path = Path("data/raw/market_indices.csv"),
    date_col: str = "ipo_date",
    lookback_days: int = 30,
    hot_market_window: int = 90,
    hot_market_tercile_threshold: float = 2 / 3,
) -> pd.DataFrame:
    """Add market-regime features derived from VIX and NASDAQ data.

    Args:
        df: IPO DataFrame.
        market_csv: Path to the market indices CSV (from
            :func:`src.scraper_prices.download_market_indices`).
        date_col: Name of the IPO date column.
        lookback_days: Number of trading days for rolling NASDAQ statistics.
        hot_market_window: Calendar-day window for counting IPOs (hot market).
        hot_market_tercile_threshold: Fraction of the distribution that defines
            the boundary between cold and hot markets (default: top tercile).

    Returns:
        Copy of *df* with additional columns:
        ``vix_at_pricing``, ``nasdaq_30d_return``, ``nasdaq_30d_volatility``,
        ``hot_market_dummy``.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")

    if not market_csv.exists():
        log.warning("Market index CSV not found at %s; skipping market features.", market_csv)
        for col in ("vix_at_pricing", "nasdaq_30d_return", "nasdaq_30d_volatility", "hot_market_dummy"):
            df[col] = np.nan
        return df

    mkt = pd.read_csv(market_csv, parse_dates=["date"], index_col="date")
    mkt = mkt.sort_index()

    # Compute rolling NASDAQ log-returns and vol
    nasdaq_ret = np.log(mkt["nasdaq_close"] / mkt["nasdaq_close"].shift(1))
    nasdaq_roll_ret = nasdaq_ret.rolling(lookback_days).sum()
    nasdaq_roll_vol = nasdaq_ret.rolling(lookback_days).std() * np.sqrt(252)

    vix_list, nasdaq_ret_list, nasdaq_vol_list = [], [], []

    for ipo_date in dates:
        if pd.isna(ipo_date):
            vix_list.append(np.nan)
            nasdaq_ret_list.append(np.nan)
            nasdaq_vol_list.append(np.nan)
            continue

        # VIX on the IPO pricing date (or nearest prior trading day)
        vix_subset = mkt["vix_close"][mkt.index <= ipo_date]
        vix_list.append(float(vix_subset.iloc[-1]) if not vix_subset.empty else np.nan)

        ret_subset = nasdaq_roll_ret[nasdaq_roll_ret.index <= ipo_date]
        nasdaq_ret_list.append(float(ret_subset.iloc[-1]) if not ret_subset.empty else np.nan)

        vol_subset = nasdaq_roll_vol[nasdaq_roll_vol.index <= ipo_date]
        nasdaq_vol_list.append(float(vol_subset.iloc[-1]) if not vol_subset.empty else np.nan)

    df["vix_at_pricing"] = vix_list
    df["nasdaq_30d_return"] = nasdaq_ret_list
    df["nasdaq_30d_volatility"] = nasdaq_vol_list

    # Hot-market dummy: rolling 90-day IPO count in top tercile
    df_sorted = df.sort_values(date_col)
    rolling_counts = []
    for ipo_date in pd.to_datetime(df_sorted[date_col]):
        if pd.isna(ipo_date):
            rolling_counts.append(np.nan)
            continue
        window_start = ipo_date - pd.Timedelta(days=hot_market_window)
        count = ((pd.to_datetime(df_sorted[date_col]) >= window_start) &
                 (pd.to_datetime(df_sorted[date_col]) <= ipo_date)).sum()
        rolling_counts.append(count)

    df_sorted["_ipo_rolling_count"] = rolling_counts
    threshold = df_sorted["_ipo_rolling_count"].quantile(hot_market_tercile_threshold)
    df_sorted["hot_market_dummy"] = (
        df_sorted["_ipo_rolling_count"] >= threshold
    ).astype(int)
    df_sorted = df_sorted.drop(columns=["_ipo_rolling_count"])

    # Merge back (preserve original order)
    hot_col = df_sorted[["hot_market_dummy"]]
    df = df.drop(columns=["hot_market_dummy"], errors="ignore")
    df = df.join(hot_col)

    return df


# ---------------------------------------------------------------------------
# Deal features
# ---------------------------------------------------------------------------

def add_deal_features(
    df: pd.DataFrame,
    underwriter_ranks_csv: Path = Path("data/external/underwriter_ranks.csv"),
) -> pd.DataFrame:
    """Add deal-specific features to *df*.

    Args:
        df: IPO DataFrame with columns ``offer_price``, ``shares_offered``,
            ``offer_size_m``, and ``lead_underwriter``.
        underwriter_ranks_csv: Path to the underwriter-rank CSV containing
            ``underwriter`` and ``rank`` columns.

    Returns:
        Copy of *df* with additional columns:
        ``log_offer_size``, ``log_shares_offered``,
        ``underwriter_rank``, ``top_tier_underwriter``.
    """
    df = df.copy()

    # Log-transforms (offer_size_m in millions, shares_offered in units)
    if "offer_size_m" in df.columns:
        df["log_offer_size"] = np.log1p(pd.to_numeric(df["offer_size_m"], errors="coerce"))
    if "shares_offered" in df.columns:
        df["log_shares_offered"] = np.log1p(pd.to_numeric(df["shares_offered"], errors="coerce"))

    # Underwriter rank merge
    if underwriter_ranks_csv.exists():
        ranks = pd.read_csv(underwriter_ranks_csv)
        # Normalise underwriter name for fuzzy matching
        if "underwriter" in ranks.columns and "lead_underwriter" in df.columns:
            ranks["_uw_key"] = ranks["underwriter"].str.lower().str.strip()
            df["_uw_key"] = df["lead_underwriter"].str.lower().str.strip()
            df = df.merge(
                ranks[["_uw_key", "rank"]].rename(columns={"rank": "underwriter_rank"}),
                on="_uw_key",
                how="left",
            ).drop(columns=["_uw_key"])
    else:
        log.warning("Underwriter ranks file not found; underwriter_rank will be NaN.")
        df["underwriter_rank"] = np.nan

    if "underwriter_rank" not in df.columns:
        df["underwriter_rank"] = np.nan

    # Top-tier dummy: Carter-Manaster rank ≥ 8
    df["top_tier_underwriter"] = (
        pd.to_numeric(df["underwriter_rank"], errors="coerce") >= 8
    ).astype(float)  # float to carry NaN through

    # SPAC flag
    if "company_name" in df.columns:
        spac_re = r"acqui(?:sition|re)|blank\s+check|spac\b"
        df["is_spac"] = df["company_name"].str.lower().str.contains(
            spac_re, regex=True, na=False
        ).astype(int)
    else:
        df["is_spac"] = 0

    return df


# ---------------------------------------------------------------------------
# Textual features
# ---------------------------------------------------------------------------

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract textual features using src.text_features.

    Args:
        df: IPO DataFrame with path columns.

    Returns:
        DataFrame with textual features: lm_*, gunning_fog, uniqueness.
    """
    df = df.copy()

    # 1. Sentiment (LM) and Readability (Fog)
    lm_dict = text_features.load_lm_dictionary()
    
    sentiment_records = []
    for _, row in df.iterrows():
        rec = {}
        risk_path = row.get("risk_factors_path")
        mda_path = row.get("mda_path")
        
        if risk_path and Path(risk_path).exists():
            risk_text = Path(risk_path).read_text(encoding="utf-8")
            if len(risk_text) > 100:
                s = text_features.compute_lm_ratios(risk_text, lm_dict)
                rec.update(s)
        
        if mda_path and Path(mda_path).exists():
            mda_text = Path(mda_path).read_text(encoding="utf-8")
            if len(mda_text) > 100:
                rec["gunning_fog"] = text_features.gunning_fog_index(mda_text)
        
        sentiment_records.append(rec)
    
    df_sent = pd.DataFrame(sentiment_records)
    for col in df_sent.columns:
        df[col] = df_sent[col].values

    # 2. Prospectus Uniqueness (Hanley-Hoberg)
    full_texts = []
    sectors = []
    for _, row in df.iterrows():
        p = row.get("full_text_path")
        s = row.get("sector", "Other")
        if p and Path(p).exists():
            full_texts.append(Path(p).read_text(encoding="utf-8"))
            sectors.append(s if pd.notna(s) and s != "" else "Other")
        else:
            full_texts.append("")
            sectors.append("Other")
    
    if any(len(t) > 0 for t in full_texts):
        # Pass sectors for more accurate Hanley-Hoberg (sector-level boilerplate)
        uniqueness = text_features.compute_prospectus_uniqueness(full_texts, sectors)
        df["prospectus_uniqueness"] = uniqueness
    else:
        df["prospectus_uniqueness"] = np.nan

    return df



# ---------------------------------------------------------------------------
# Combined pipeline entry point
# ---------------------------------------------------------------------------

def build_all_features(
    df: pd.DataFrame,
    market_csv: Path = Path("data/raw/market_indices.csv"),
    underwriter_ranks_csv: Path = Path("data/external/underwriter_ranks.csv"),
) -> pd.DataFrame:
    """Apply all feature-engineering steps in sequence.

    Args:
        df: Cleaned IPO DataFrame (output of preprocessing).
        market_csv: Market indices CSV path.
        underwriter_ranks_csv: Underwriter rank CSV path.

    Returns:
        DataFrame with all engineered features appended.
    """
    log.info("Building calendar features …")
    df = add_calendar_features(df)

    log.info("Building market-regime features …")
    df = add_market_features(df, market_csv=market_csv)

    log.info("Building deal features …")
    df = add_deal_features(df, underwriter_ranks_csv=underwriter_ranks_csv)

    log.info("Building text features (NLP) …")
    df = add_text_features(df)

    log.info("Feature engineering complete. Shape: %s", df.shape)
    return df
