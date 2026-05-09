"""
Data preprocessing: missing-value imputation, outlier handling, and
categorical encoding for the IPO underpricing dataset.

Design decisions are documented inline so that the notebook can reference
them as justification.  All functions are pure (return copies, do not mutate
the input) and are idempotent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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
# Rule-based sector classification for companies missing sector data
# Uses GICS (Global Industry Classification Standard) sectors. The
# stockanalysis.com detail scrape only completed for ~8% of tickers before
# being rate-limited, so we infer sector from company name.
#
# Strategy (priority order):
#   1. Trust an existing non-empty sector label.
#   2. Manual ticker overrides for well-known IPOs (Pinterest, Zoom, Lyft, …).
#   3. SPAC pattern match (these dominate the dataset 2020-2021).
#   4. Word-boundary keyword matching against curated GICS keyword lists.
#   5. Suffix heuristics ("Bancorp" → Financials, "Inc" with no other signal
#      → Industrials).  We never emit "Unknown" or "Other/Diversified".
# ---------------------------------------------------------------------------

import re as _re

# Manual overrides for well-known names that wouldn't reliably match keywords.
# Keys are lowercased exact ticker symbols.
_TICKER_OVERRIDES: dict[str, str] = {
    # Tech / software / internet
    "pins": "Technology", "zm": "Technology", "lyft": "Consumer Discretionary",
    "uber": "Consumer Discretionary", "abnb": "Consumer Discretionary",
    "dash": "Consumer Discretionary", "rivn": "Consumer Discretionary",
    "pltr": "Technology", "snow": "Technology", "u": "Technology",
    "rblx": "Communication Services", "ds": "Communication Services",
    "dbx": "Technology", "twlo": "Technology", "net": "Technology",
    "crwd": "Technology", "zs": "Technology", "ddog": "Technology",
    "okta": "Technology", "mdb": "Technology", "estc": "Technology",
    "fsly": "Technology", "asan": "Technology", "team": "Technology",
    "frsh": "Technology", "path": "Technology", "ai": "Technology",
    "cflt": "Technology", "gtlb": "Technology", "rbrk": "Technology",
    "sprr": "Technology", "spt": "Technology", "rxt": "Technology",
    "futu": "Financials", "ms": "Financials", "kc": "Technology",
    "atec": "Healthcare", "medp": "Healthcare", "krtx": "Healthcare",
    "bhvn": "Healthcare", "swav": "Healthcare", "argx": "Healthcare",
    "rvmd": "Healthcare", "rxrx": "Healthcare", "alec": "Healthcare",
    "atra": "Healthcare", "trvi": "Healthcare", "atrc": "Healthcare",
    # Consumer
    "levi": "Consumer Discretionary", "rl": "Consumer Discretionary",
    "ctva": "Materials", "wmg": "Communication Services",
    "spot": "Communication Services", "siri": "Communication Services",
    # Energy / industrial
    "rrc": "Energy", "vist": "Energy",
    # Financials
    "vrt": "Industrials",
    # More known names that wouldn't match keywords
    "mcfe": "Technology", "inst": "Technology", "exfy": "Technology",
    "nvei": "Technology", "api": "Technology", "blnd": "Technology",
    "ncno": "Technology", "snwv": "Technology", "smar": "Technology",
    "ampl": "Technology", "iren": "Technology",
    "vrm": "Consumer Discretionary", "cook": "Consumer Discretionary",
    "weber": "Consumer Discretionary", "wbr": "Consumer Discretionary",
    "shco": "Consumer Discretionary", "onon": "Consumer Discretionary",
    "bird": "Consumer Discretionary", "mcw": "Consumer Discretionary",
    "rl": "Consumer Discretionary",
    "vaxx": "Healthcare", "ppd": "Healthcare", "pepg": "Healthcare",
    "soph": "Healthcare", "atrc": "Healthcare", "hcwb": "Healthcare",
    "lsdif": "Healthcare", "azt": "Healthcare", "azitra": "Healthcare",
    "lmnd": "Financials", "root": "Financials", "rely": "Financials",
    "tw": "Financials", "step": "Financials",
    "fubo": "Communication Services", "ds": "Communication Services",
    "kvue": "Consumer Staples", "ptve": "Consumer Staples",
    "tbbb": "Consumer Staples",
    "bnl": "Real Estate",
    "usgo": "Materials",
    "ctva": "Materials",
}

# GICS sectors with curated keywords. We use word-boundary regex so that
# "ai" doesn't match "captain", and "ev" doesn't match "every".
# Order matters — earlier entries take precedence for ambiguous names.
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Healthcare": [
        r"pharma", r"pharmaceutical", r"therapeutics?", r"biosciences?",
        r"biotech", r"biopharma", r"\bbiolog\w*", r"medical", r"medicines?",
        r"medi[ck]al", r"health", r"oncology", r"genomic", r"diagnostic",
        r"surgical", r"surgery", r"vaccine", r"immun[oa]", r"neuro",
        r"cardio", r"ophthalm", r"derma", r"clinical", r"life\s+sciences?",
        r"hospital", r"cannabis", r"\bcbd\b", r"\bhemp\b", r"wellness",
        r"nutrient", r"vitamin", r"\bbio\b", r"genom\w*", r"genetic\w*",
        r"therapy", r"therap\w+", r"dental", r"\bvision\b", r"prescript",
        r"\bdrug\b", r"vaccin\w*", r"\bvaxx\w*", r"epigen\w*", r"\bprotein",
        r"antibody", r"\brna\b", r"crispr", r"\bbiosci\w*", r"sophia",
    ],
    "Technology": [
        r"software", r"\bdigital", r"\btech\w*", r"\bcyber",
        r"\bcloud\w*", r"\bai\b", r"artificial\s+intelligence",
        r"machine\s+learn", r"\brobot", r"automation", r"semiconductor",
        r"\bchip\b", r"computing", r"computer", r"\bsaas\b", r"platform",
        r"\bapp\b", r"mobile", r"\binternet\b", r"e[\s\-]?commerce",
        r"fintech", r"blockchain", r"crypto", r"quantum", r"analytic",
        r"network", r"\biot\b", r"information\s+tech",
        r"\bsystems?\b", r"cybersec\w*", r"\bdata\w*",
    ],
    "Communication Services": [
        r"media", r"broadcast", r"publish", r"advertis\w*", r"social",
        r"content", r"\bnews\b", r"entertainment", r"music", r"podcast",
        r"studio", r"\bgame", r"gaming", r"telecom", r"wireless",
        r"satellite", r"streaming", r"\bstream\b", r"interactive",
    ],
    "Financials": [
        r"\bbank\b", r"bancorp", r"\bcapital\b", r"financial", r"insurance",
        r"asset\s+manag", r"investment", r"\bcredit\b", r"lending",
        r"mortgage", r"wealth", r"\btrust\b", r"securities", r"brokerage",
        r"\bfund\b", r"\breit\b", r"\bventure", r"private\s+equity",
        r"\bsavings\b", r"holdings?\s+corp", r"financ\w*",
        r"\bmarkets\b", r"\bexchange\b", r"\bpartners\b",
    ],
    "Energy": [
        r"\benergy\b", r"\bsolar\b", r"\boil\b", r"\bgas\b", r"petroleum",
        r"\bwind\b", r"renewable", r"clean\s+energy", r"green\s+energy",
        r"power\b", r"electric\s+util", r"\bfuel\b", r"drilling",
        r"\blithium", r"\bbattery", r"hydrogen", r"nuclear", r"\bev\b",
        r"electric\s+vehicle", r"charging", r"emission",
    ],
    "Materials": [
        r"\bgold\b", r"\bsilver\b", r"\bcopper\b", r"mineral", r"resources?",
        r"forest", r"\bpaper\b", r"packag\w*", r"container",
        r"\bmining\b", r"\bsteel\b", r"\bmetal", r"chemical", r"\bplastic",
    ],
    "Industrials": [
        r"aerospace", r"defen[cs]e", r"manufactur", r"industrial",
        r"engineer", r"construct", r"\bbuilding\b", r"transport",
        r"logistic", r"shipping", r"freight", r"\btruck", r"\brail",
        r"airline", r"aviation", r"\bdrone\b", r"infrastructure",
        r"environment\w*", r"\bwaste\b",
    ],
    "Consumer Discretionary": [
        r"retail", r"fashion", r"luxury", r"apparel", r"\bcloth\w*",
        r"restaurant", r"\bhotel\b", r"travel", r"leisure", r"casino",
        r"gambling", r"sport", r"fitness", r"beauty", r"cosmetic",
        r"\bauto\b", r"automotive", r"\bmotor\b", r"vehicle", r"\bride\b",
        r"delivery", r"furniture", r"home\s+decor", r"\btoy\b",
        r"e[\s\-]?commerce\s+retail", r"\bjeans?\b", r"footwear",
        r"\bbrand\b", r"\bcoupon",
    ],
    "Consumer Staples": [
        r"\bfood\b", r"beverage", r"grocery", r"\bsnack\b", r"organic",
        r"\bmeat\b", r"dairy", r"coffee", r"\btea\b", r"\bwater\b",
        r"household", r"cleaning", r"personal\s+care", r"\bbeer\b",
        r"\bwine\b", r"distill",
    ],
    "Real Estate": [
        r"real\s+estate", r"property", r"\breit\b", r"housing",
        r"\bland\b", r"developer",
    ],
    "Utilities": [
        r"\butilit", r"water\s+treat", r"sewage", r"electric\s+util",
    ],
}

# Compile regex per sector once, with word boundaries baked in.
_SECTOR_PATTERNS: dict[str, _re.Pattern[str]] = {
    sector: _re.compile("|".join(keywords), _re.IGNORECASE)
    for sector, keywords in _SECTOR_KEYWORDS.items()
}

# Strong SPAC indicators (run BEFORE operating keywords). A name with
# "Healthcare Merger Corp" is a SPAC, not an operating healthcare company.
_SPAC_STRONG_PATTERN = _re.compile(
    r"\b("
    r"acquisition\s+(corp|corporation|company)|"
    r"merger\s+(corp|corporation|sub)|"
    r"\bspac\b|"
    r"blank[\s\-]check|"
    r"\bsponsor\s+capital\b"
    r")\b",
    _re.IGNORECASE,
)

# Weaker SPAC indicators (run AFTER operating keywords).
_SPAC_WEAK_PATTERN = _re.compile(
    r"\b(acquisition|merger)\b",
    _re.IGNORECASE,
)

# Suffix-based fallback heuristics.
_SUFFIX_HEURISTICS: list[tuple[_re.Pattern[str], str]] = [
    (_re.compile(r"\bbancorp\b|\bbancshares\b", _re.IGNORECASE), "Financials"),
    (_re.compile(r"\bholdings?\b.*\bcorp\b", _re.IGNORECASE), "Financials"),
    (_re.compile(r"\btrust\b", _re.IGNORECASE), "Financials"),
    (_re.compile(r"\bgroup\b\s*(inc|ltd|holdings?)?$", _re.IGNORECASE), "Financials"),
]


def classify_sector(
    company_name: str,
    existing_sector: str | None = None,
    ticker: str | None = None,
) -> str:
    """Classify a company into a GICS sector.

    Priority:
      1. Trust an existing non-empty sector label.
      2. Manual ticker override (well-known IPOs).
      3. Operating-company keyword match (Healthcare → Tech → Comm → ...).
      4. SPAC if the name has SPAC patterns AND no operating keyword matched.
      5. Suffix heuristics (Bancorp → Financials, etc.).
      6. Final fallback: "Industrials" (broad GICS bucket; never "Unknown").

    Args:
        company_name: The company name string.
        existing_sector: Pre-existing sector label (from scrape).
        ticker: Optional ticker symbol for manual overrides.

    Returns:
        GICS sector string. Never "Unknown" or "Other/Diversified".
    """
    import pandas as _pd

    # 1. Existing label wins
    if (existing_sector
        and not _pd.isna(existing_sector)
        and str(existing_sector).strip() not in
            ("Unknown", "Other/Diversified", "", "nan", "None")):
        return str(existing_sector).strip()

    # 2. Ticker override
    if ticker:
        t = str(ticker).strip().lower()
        if t in _TICKER_OVERRIDES:
            return _TICKER_OVERRIDES[t]

    name = (company_name or "").strip()
    if not name:
        return "Industrials"

    # 3. Strong SPAC indicators dominate (e.g. "Healthcare Merger Corp" is a
    # SPAC, not a healthcare company)
    if _SPAC_STRONG_PATTERN.search(name):
        return "SPAC"

    # 4. Operating-company keyword match (first hit wins)
    for sector, pattern in _SECTOR_PATTERNS.items():
        if pattern.search(name):
            return sector

    # 5. Weak SPAC indicators (just "Acquisition" or "Merger" alone)
    if _SPAC_WEAK_PATTERN.search(name):
        return "SPAC"

    # 6. Suffix heuristics
    for pat, sector in _SUFFIX_HEURISTICS:
        if pat.search(name):
            return sector

    # 7. Final fallback. We pick "Industrials" as a broad GICS bucket
    # (rather than "Other/Diversified" or "Unknown") for names with no
    # discernible signal. This applies to roughly 5-10% of the dataset.
    return "Industrials"


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

    # Sector — rule-based classification with ticker overrides
    if "company_name" in df.columns:
        df["sector"] = df.apply(
            lambda row: classify_sector(
                row.get("company_name", ""),
                row.get("sector", None),
                row.get("ticker", None),
            ),
            axis=1,
        )
        # Sweep: any leftover legacy labels become a real GICS sector
        df["sector"] = df["sector"].replace(
            {"Other/Diversified": "Industrials", "Unknown": "Industrials"}
        )
        sector_counts = df["sector"].value_counts()
        log.info("Sector classification:\n%s", sector_counts.to_string())

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
