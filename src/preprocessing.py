"""
Data preprocessing: missing-value imputation, outlier handling, and
categorical encoding for the IPO underpricing dataset.

Design decisions are documented inline so that the notebook can reference
them as justification.  All functions are pure (return copies, do not mutate
the input) and are idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERIM_PATH = Path("data/interim/ipo_clean.parquet")
RAW_PATH = Path("data/raw/ipo_master_raw.csv")

# Columns where a null means we must drop the row
_MANDATORY_COLS = ["ticker", "ipo_date", "offer_price", "first_day_close"]

# Numeric columns with meaningful median-imputation candidates
_NUMERIC_IMPUTE_COLS = [
    "offer_size_m",
    "shares_offered",
    "underwriter_rank",
    "vix_at_pricing",
    "nasdaq_30d_return",
    "nasdaq_30d_volatility",
]

# Winsorisation tails
_WINSOR_LOWER = 0.01
_WINSOR_UPPER = 0.99


# ---------------------------------------------------------------------------
# Load / merge raw sources
# ---------------------------------------------------------------------------

def load_and_merge_raw() -> pd.DataFrame:
    """Merge the IPO calendar, price data, and text filing paths.

    Returns:
        Merged DataFrame saved to data/raw/ipo_master_raw.csv.
    """
    master_path = Path("data/raw/ipo_master_raw.csv")
    if not master_path.exists():
        # Fallback if master hasn't been created yet
        df = pd.read_csv(Path("data/raw/ipo_calendar.csv"), parse_dates=["ipo_date"])
        df["ticker"] = df["ticker"].str.upper().str.strip()
        # Ensure underpricing/close are there
        if "underpricing" not in df.columns and "first_day_return_pct" in df.columns:
            df["underpricing"] = df["first_day_return_pct"] / 100.0
        if "first_day_close" not in df.columns and "offer_price" in df.columns:
            df["first_day_close"] = df["offer_price"] * (1 + df.get("underpricing", 0))
    else:
        df = pd.read_csv(master_path, parse_dates=["ipo_date"])
        df["ticker"] = df["ticker"].str.upper().str.strip()

    # Find text filing paths
    s1_dir = Path("data/raw/s1_filings")
    text_files = {}
    if s1_dir.exists():
        for f in s1_dir.glob("*_risk_factors.txt"):
            ticker_date = f.name.replace("_risk_factors.txt", "")
            # Split into ticker and date (last part is date)
            parts = ticker_date.rsplit("_", 1)
            if len(parts) == 2:
                ticker, date_str = parts
                text_files[(ticker, date_str)] = {
                    "risk_factors_path": str(f),
                    "mda_path": str(f.parent / f"{ticker_date}_mda.txt"),
                    "full_text_path": str(f.parent / f"{ticker_date}.txt"),
                }

    # Map paths to dataframe
    def get_path(row, key):
        t = str(row["ticker"]).upper()
        d = str(row["ipo_date"])[:10]
        return text_files.get((t, d), {}).get(key, "")

    df["risk_factors_path"] = df.apply(lambda r: get_path(r, "risk_factors_path"), axis=1)
    df["mda_path"] = df.apply(lambda r: get_path(r, "mda_path"), axis=1)
    df["full_text_path"] = df.apply(lambda r: get_path(r, "full_text_path"), axis=1)

    df.to_csv(master_path, index=False)
    log.info("Master dataset updated with text paths → %s (%d rows)", master_path, len(df))
    return df



# ---------------------------------------------------------------------------
# Drop structurally incomplete rows
# ---------------------------------------------------------------------------

def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any mandatory column.

    Mandatory columns: ``ticker``, ``ipo_date``, ``offer_price``,
    ``first_day_close``.

    Args:
        df: Raw merged DataFrame.

    Returns:
        Filtered DataFrame with a reset index.  Logs the drop count.
    """
    before = len(df)
    cols_present = [c for c in _MANDATORY_COLS if c in df.columns]
    df = df.dropna(subset=cols_present).reset_index(drop=True)
    dropped = before - len(df)
    log.info("Dropped %d rows missing mandatory columns; %d remain.", dropped, len(df))
    return df


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with documented justifications.

    Imputation rules:

    - ``underwriter_rank`` → sector-median rank.
      *Why: missing rank typically indicates a boutique underwriter, which
      clusters near the sector median rather than being a top-tier bank.*
    - ``sector`` → "Unknown".
      *Why: missingness is not informative; encoding as its own category
      avoids data loss.*
    - Numeric columns (offer size, VIX, etc.) → column median.
      *Why: median is robust to the heavy-tailed distribution of these
      variables.*
    - ``lead_underwriter`` → "Unknown".

    Args:
        df: DataFrame after mandatory-column filtering.

    Returns:
        Copy of *df* with imputed values and a boolean flag column
        ``underwriter_rank_imputed``.
    """
    df = df.copy()

    # Sector
    if "sector" in df.columns:
        df["sector"] = df["sector"].fillna("Unknown").replace("", "Unknown")

    # Lead underwriter
    if "lead_underwriter" in df.columns:
        df["lead_underwriter"] = df["lead_underwriter"].fillna("Unknown")

    # Underwriter rank — sector-median imputation
    if "underwriter_rank" in df.columns:
        df["underwriter_rank_imputed"] = df["underwriter_rank"].isna().astype(int)
        if "sector" in df.columns:
            sector_medians = df.groupby("sector")["underwriter_rank"].transform("median")
            global_median = df["underwriter_rank"].median()
            df["underwriter_rank"] = df["underwriter_rank"].fillna(sector_medians)
            df["underwriter_rank"] = df["underwriter_rank"].fillna(global_median)
        else:
            df["underwriter_rank"] = df["underwriter_rank"].fillna(
                df["underwriter_rank"].median()
            )

    # Other numeric columns — global median
    for col in _NUMERIC_IMPUTE_COLS:
        if col in df.columns and col != "underwriter_rank":
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log.debug("Imputed %s with median=%.3f", col, median)

    return df


# ---------------------------------------------------------------------------
# Outlier handling — winsorisation
# ---------------------------------------------------------------------------

def winsorise_target(df: pd.DataFrame, col: str = "underpricing") -> pd.DataFrame:
    """Add a winsorised version of the target column.

    Raw underpricing is retained for descriptive statistics.
    ``winsorized_underpricing`` (at 1st/99th percentiles) is the column used
    in all modelling to prevent a handful of extreme observations (e.g. DWAC)
    from dominating gradient-based losses.

    Args:
        df: DataFrame with an ``underpricing`` column.
        col: Name of the column to winsorise.

    Returns:
        Copy of *df* with a ``winsorized_{col}`` column added.
    """
    df = df.copy()
    if col not in df.columns:
        log.warning("Column %s not found; skipping winsorisation.", col)
        return df

    lower = df[col].quantile(_WINSOR_LOWER)
    upper = df[col].quantile(_WINSOR_UPPER)

    df[f"winsorized_{col}"] = df[col].clip(lower, upper)
    log.info(
        "Winsorised %s at [%.3f, %.3f] → new column winsorized_%s",
        col, lower, upper, col,
    )
    return df


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns for modelling.

    Strategy:
    - **Target encoding** for high-cardinality columns (``lead_underwriter``,
      ``sector``) — mean of ``winsorized_underpricing`` per category.
      *Why: label/ordinal encoding imposes arbitrary order; one-hot explodes
      dimensionality for 100+ underwriters.*
    - **One-hot encoding** for low-cardinality columns (``exchange``,
      ``ipo_dayofweek``).

    Args:
        df: DataFrame with imputed values and ``winsorized_underpricing``.

    Returns:
        Copy of *df* with encoded columns.  Original categorical columns are
        retained (with ``_raw`` suffix) for interpretability.
    """
    df = df.copy()

    target_col = "winsorized_underpricing" if "winsorized_underpricing" in df.columns else "underpricing"

    # --- Target encoding (high cardinality) ---
    for col in ("lead_underwriter", "sector", "industry"):
        if col not in df.columns:
            continue
        df[f"{col}_raw"] = df[col]
        means = df.groupby(col)[target_col].transform("mean")
        global_mean = df[target_col].mean()
        df[f"{col}_encoded"] = means.fillna(global_mean)
        log.debug("Target-encoded %s (%d unique)", col, df[col].nunique())

    # --- One-hot encoding (low cardinality) ---
    ohe_cols = [c for c in ("exchange",) if c in df.columns and df[c].nunique() <= 20]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, drop_first=True)

    return df


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    output_path: Path = INTERIM_PATH,
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end-to-end.

    Args:
        output_path: Destination for the cleaned Parquet file.

    Returns:
        Cleaned, encoded DataFrame ready for feature engineering.
    """
    log.info("=== Preprocessing pipeline starting ===")

    df = load_and_merge_raw()
    df = drop_incomplete(df)
    df = impute_missing(df)
    df = winsorise_target(df)
    df = encode_categoricals(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Cleaned dataset saved → %s (%d rows, %d cols)", output_path, *df.shape)
    return df


# ---------------------------------------------------------------------------
# Summary utilities (used in the notebook)
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a missing-value summary table for the notebook.

    Args:
        df: Any DataFrame.

    Returns:
        DataFrame with columns ``column``, ``dtype``, ``n_missing``,
        ``pct_missing``.
    """
    report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean() * 100).round(2).values,
    })
    return report.sort_values("pct_missing", ascending=False).reset_index(drop=True)
