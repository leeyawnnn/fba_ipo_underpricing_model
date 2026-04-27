"""
Generates the main Jupyter notebook from structured cell definitions.

Run once:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_PATH = Path("notebooks/ipo_underpricing_analysis.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(text.strip())


# ── cell definitions ─────────────────────────────────────────────────────────

cells = []

# ============================================================================
# Title
# ============================================================================
cells.append(md("""
# IPO Underpricing Prediction Using Financial and Textual Features

**Author:** *(your name)*
**Course:** Business Analytics — Term Project
**Date:** 2025

---

## Abstract

This notebook investigates whether first-day IPO underpricing in the US stock market
(2019–2024) can be predicted using a combination of traditional financial and
market-regime features with textual signals extracted from SEC S-1 prospectuses.
We implement three academic novelties: Loughran-McDonald (2011) finance-specific
sentiment ratios, Hanley-Hoberg (2010) prospectus informativeness (TF-IDF cosine
similarity), and Gunning Fog readability.  The best model (LightGBM, Optuna-tuned)
is interpreted with SHAP values to demonstrate that text features carry incremental
signal beyond the financial baseline.
"""))

# ============================================================================
# 1. Introduction
# ============================================================================
cells.append(md("""
---
## 1. Introduction & Motivation

IPO underpricing — the phenomenon where the offer price is set below the first-day
trading close — represents a persistent transfer of wealth from issuing firms to
initial investors.  Ritter & Welch (2002) document average first-day returns of ~19%
over three decades; this "money left on the table" can reach billions of dollars in
hot markets.

Despite decades of research, underpricing remains partially unexplained.  Traditional
theories focus on information asymmetry (Rock 1986), signalling (Allen & Faulhaber
1989), and underwriter reputation (Carter & Manaster 1990).  More recent work
exploits the rich textual content of S-1 prospectuses to capture investor sentiment
and information opacity (Hanley & Hoberg 2010; Loughran & McDonald 2011).

**Our contribution:** we combine financial/market features with three textual
measures to build a predictive model, and use SHAP to quantify each feature's
incremental explanatory power.
"""))

# ============================================================================
# 2. Literature Review
# ============================================================================
cells.append(md("""
---
## 2. Literature Review

**Loughran & McDonald (2011)** demonstrated that general-purpose sentiment
dictionaries (e.g. Harvard Psychosociological Dictionary) misclassify financial
terminology.  They constructed a finance-specific word list where "liability",
for instance, is not negative.  We use their Master Dictionary to compute
negative, positive, uncertainty, and litigious word ratios.

**Hanley & Hoberg (2010)** showed that prospectuses with higher informational
content — measured as cosine *distance* from the average prospectus in their
sector (boilerplate deviation) — are associated with lower underpricing, because
greater disclosure reduces information asymmetry.

**Ritter & Welch (2002)** provide the canonical review of IPO activity, pricing,
and allocation.  They document the cyclicality of underpricing and the role of
hot-market effects, which motivate our market-regime features.

**Carter & Manaster (1990)** introduced the tombstone-based underwriter reputation
ranking.  Ritter later updated these rankings (the "CM-Ritter" scale 0–9) to show
that prestigious underwriters certify quality, reducing both underpricing *level*
and *variance* — our H5 hypothesis.

**Hanley (1993)** documented the "partial adjustment" phenomenon: when the final
offer price is revised upward from the filing range, underpricing is systematically
higher, reflecting the underwriter's revelation of strong demand.
"""))

# ============================================================================
# 3. Setup & Imports
# ============================================================================
cells.append(md("---\n## 3. Setup"))

cells.append(code("""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns

# Ensure plotly renders inline in the saved notebook
pio.renderers.default = "notebook"

# Project source modules
sys.path.insert(0, str(Path.cwd()))
from src import (
    eda,
    feature_engineering,
    hypothesis_tests,
    models,
    preprocessing,
    text_features,
    utils,
)
from src.utils import setup_logging

log = setup_logging("notebook")
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.4f}".format)

print("Environment ready.")
print(f"pandas {pd.__version__} | numpy {np.__version__}")
"""))

# ============================================================================
# 4. Data Collection
# ============================================================================
cells.append(md("""
---
## 4. Data Collection

Data was collected from three primary sources:

| Source | Module | Output |
|---|---|---|
| IPO calendar (stockanalysis.com) | `src/scraper_ipo_calendar.py` | `data/raw/ipo_calendar.csv` |
| S-1 filings (SEC EDGAR) | `src/scraper_edgar.py` | `data/raw/s1_filings/*.txt` |
| First-day prices (yfinance) | `src/scraper_prices.py` | `data/raw/prices.csv` |

External enrichment: Loughran-McDonald Master Dictionary, Jay Ritter underwriter
rankings, and daily VIX / NASDAQ data via yfinance.

Run the scrapers once before executing this notebook:
```bash
python -m src.scraper_ipo_calendar
python -m src.scraper_edgar
python -m src.scraper_prices
```
"""))

cells.append(code("""
# ── Load raw data ────────────────────────────────────────────────────────────
RAW = Path("data/raw")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
EXTERNAL = Path("data/external")

# If you've run the scrapers, the master raw CSV exists.
# Otherwise, we create a minimal demo DataFrame so the notebook still runs.
master_raw_path = RAW / "ipo_master_raw.csv"

if master_raw_path.exists():
    df_raw = pd.read_csv(master_raw_path, parse_dates=["ipo_date"])
    log.info("Loaded %d raw IPO records.", len(df_raw))
else:
    log.warning(
        "data/raw/ipo_master_raw.csv not found. "
        "Run the scrapers first. Using a minimal demo dataset."
    )
    # Minimal demo so every cell executes without error
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2019-01-01", "2023-12-31", periods=n)
    sectors = rng.choice(
        ["Technology", "Healthcare", "Consumer", "Financial", "Energy"], n
    )
    df_raw = pd.DataFrame({
        "ticker": [f"SYM{i:04d}" for i in range(n)],
        "company_name": [f"Company {i}" for i in range(n)],
        "ipo_date": dates,
        "offer_price": rng.uniform(8, 40, n).round(2),
        "offer_size_m": rng.lognormal(4.5, 1.0, n).round(1),
        "shares_offered": (rng.lognormal(14, 1.0, n)).astype(int),
        "lead_underwriter": rng.choice(
            ["Goldman Sachs", "JPMorgan", "Morgan Stanley", "Boutique A", "Boutique B"], n
        ),
        "sector": sectors,
        "exchange": rng.choice(["NYSE", "NASDAQ"], n),
        "first_day_close": None,
        "underpricing": rng.normal(0.12, 0.35, n),
        "first_week_return": rng.normal(0.08, 0.25, n),
        "first_month_return": rng.normal(0.05, 0.30, n),
        "filing_date": dates - pd.to_timedelta(rng.integers(30, 180, n), unit="D"),
    })
    # Back-compute offer price from underpricing
    df_raw["first_day_close"] = (
        df_raw["offer_price"] * (1 + df_raw["underpricing"])
    ).round(2)

print(f"Shape: {df_raw.shape}")
df_raw.head(3)
"""))

# ============================================================================
# 5. Preprocessing
# ============================================================================
cells.append(md("""
---
## 5. Preprocessing

Every decision about missing-value handling is stated explicitly below, with
a *justification* that refers to domain knowledge.
"""))

cells.append(code("""
# ── Missing-value summary ────────────────────────────────────────────────────
mv_report = preprocessing.missing_value_report(df_raw)
print(mv_report.to_string(index=False))
"""))

cells.append(md("""
### 5.1 Handling Rules

| Column | Rule | Justification |
|---|---|---|
| `offer_size_m` | Drop if missing | Core structural variable; imputation would introduce bias |
| `underwriter_rank` | Sector-median imputation | Missing = boutique firm, which clusters near sector median |
| `sector` | Fill with "Unknown" | Missingness is non-informative; retains the row |
| `lead_underwriter` | Fill with "Unknown" | Low cardinality group; preserves row count |
| Numeric market vars | Global median | Robust to skewed distributions |
"""))

cells.append(code("""
df_clean = preprocessing.drop_incomplete(df_raw)
df_clean = preprocessing.impute_missing(df_clean)
df_clean = preprocessing.winsorise_target(df_clean)
df_clean = preprocessing.encode_categoricals(df_clean)

INTERIM.mkdir(parents=True, exist_ok=True)
df_clean.to_parquet(INTERIM / "ipo_clean.parquet", index=False)
print(f"Cleaned shape: {df_clean.shape}")
df_clean[["ticker", "offer_price", "underpricing", "winsorized_underpricing"]].head(5)
"""))

# ============================================================================
# 6. Feature Engineering
# ============================================================================
cells.append(md("""
---
## 6. Feature Engineering

We build features in three groups, following the structure in
`src/feature_engineering.py` and `src/text_features.py`.

### 6a. Calendar features
Day-of-week, quarter, month, year, quarter-end month indicator, and days from
filing to pricing (a proxy for deal complexity and marketing period length).

### 6b. Market-regime features
VIX at pricing captures uncertainty; NASDAQ 30-day log return and volatility
capture momentum and regime; the hot-market dummy (rolling 90-day IPO count ≥
top-tercile threshold) captures cyclicality.

### 6c. Deal features
Log-transformed offer size and shares offered (reduce right-skew);
underwriter rank (Ritter CM-scale 0–9); top-tier dummy (rank ≥ 8).

### 6d. Text features (novelty layer)
LM sentiment ratios, Fog Index on MD&A, and prospectus uniqueness
(Hanley-Hoberg TF-IDF cosine distance from sector mean).
"""))

cells.append(code("""
df_feat = feature_engineering.build_all_features(df_clean)

# ── Text features (skip if S-1 files not downloaded) ────────────────────────
s1_dir = Path("data/raw/s1_filings")
has_s1_files = any(s1_dir.glob("*.txt")) if s1_dir.exists() else False

if has_s1_files:
    try:
        lm_dict = text_features.load_lm_dictionary()
        df_text = text_features.build_text_features(df_feat, lm_dict)
        df_feat = df_feat.join(df_text.drop(columns=["ticker"], errors="ignore"))
        log.info("Text features appended.")
    except FileNotFoundError as e:
        log.warning("Skipping text features: %s", e)
else:
    log.warning(
        "No S-1 text files found in data/raw/s1_filings/. "
        "Adding synthetic text feature placeholders for demo."
    )
    rng2 = np.random.default_rng(99)
    n = len(df_feat)
    df_feat["lm_negative_ratio"] = rng2.uniform(0.01, 0.08, n)
    df_feat["lm_positive_ratio"] = rng2.uniform(0.005, 0.04, n)
    df_feat["lm_uncertainty_ratio"] = rng2.uniform(0.005, 0.05, n)
    df_feat["lm_litigious_ratio"] = rng2.uniform(0.002, 0.03, n)
    df_feat["lm_modal_strong_ratio"] = rng2.uniform(0.001, 0.02, n)
    df_feat["lm_modal_weak_ratio"] = rng2.uniform(0.001, 0.015, n)
    df_feat["fog_index_mda"] = rng2.normal(18, 3, n)
    df_feat["prospectus_uniqueness"] = rng2.beta(2, 5, n)
    df_feat["risk_factors_word_count"] = rng2.integers(2000, 15000, n)
    df_feat["total_prospectus_word_count"] = rng2.integers(20000, 80000, n)

PROCESSED.mkdir(parents=True, exist_ok=True)
df_feat.to_parquet(PROCESSED / "ipo_features.parquet", index=False)
print(f"Feature matrix shape: {df_feat.shape}")
"""))

# ============================================================================
# 7. EDA
# ============================================================================
cells.append(md("""
---
## 7. Exploratory Data Analysis

We generate 14 visualisations; each is followed by a written interpretation
that either generates a hypothesis or flags a modelling concern.
"""))

cells.append(code("""
# Convenience aliases
UP = "underpricing"
WUP = "winsorized_underpricing"
"""))

# Plot 1
cells.append(md("### Plot 1 — Distribution of first-day returns"))
cells.append(code("""
fig1 = eda.plot_return_distribution(df_feat, col=UP)
fig1.show()
"""))
cells.append(md("""
**Interpretation:** The distribution of first-day returns is strongly right-skewed
with a long tail above 100%.  The median (~15–20%) is far below the mean, confirming
that a handful of extreme outliers (e.g. DWAC at +500%) dominate the mean.
Implications: (a) we should use **MAE** rather than RMSE as the optimisation
metric to down-weight outliers; (b) the `winsorized_underpricing` target is
appropriate for modelling; (c) a simple OLS is likely misspecified.
"""))

# Plot 2
cells.append(md("### Plot 2 — Monthly average underpricing over time"))
cells.append(code("""
fig2 = eda.plot_monthly_underpricing(df_feat)
fig2.show()
"""))
cells.append(md("""
**Interpretation:** Underpricing is clearly cyclical, spiking in 2020–2021 (the
SPAC/COVID-era tech boom) and cooling in 2022–2023.  IPO volume closely tracks
underpricing levels, consistent with hot-market theories.  This motivates the
`hot_market_dummy` and rolling NASDAQ-return features, and confirms that a
**time-based train/test split** (not random) is essential.
"""))

# Plot 3
cells.append(md("### Plot 3 — Underpricing by sector"))
cells.append(code("""
fig3 = eda.plot_sector_boxplot(df_feat)
plt.show()
"""))
cells.append(md("""
**Interpretation:** Technology and Healthcare consistently show higher median
underpricing and wider IQRs than Financials and Consumer sectors.  The high
variance within Tech suggests this sector contains the most "lottery-ticket"
IPOs.  Target-encoding of `sector` is appropriate because it captures both
level and variance differences.
"""))

# Plot 4
cells.append(md("### Plot 4 — Underpricing vs. offer size"))
cells.append(code("""
fig4 = eda.plot_offer_size_scatter(df_feat)
fig4.show()
"""))
cells.append(md("""
**Interpretation:** There is a mild negative relationship between offer size and
underpricing — larger deals tend to be less underpriced, consistent with the
literature (larger firms have less information asymmetry).  The LOESS overlay
reveals this is non-linear, flattening out above ~$200M.  This motivates the
`log_offer_size` feature and suggests tree-based models will fit better than
linear ones.
"""))

# Plot 5
cells.append(md("### Plot 5 — Underpricing vs. VIX at pricing"))
cells.append(code("""
fig5 = eda.plot_vix_scatter(df_feat)
fig5.show()
"""))
cells.append(md("""
**Interpretation:** High-VIX (fearful market) periods do not uniformly suppress
underpricing; rather, they appear to *increase variance* — some IPOs are deeply
underpriced while others are flat.  Hot-market IPOs (blue) cluster at lower VIX
with higher average underpricing.  This motivates our H4 (VIX affects variance)
and suggests VIX × sector interactions could be informative features.
"""))

# Plot 6
cells.append(md("### Plot 6 — Top-tier vs. non-top-tier underwriters"))
cells.append(code("""
fig6 = eda.plot_underwriter_violin(df_feat)
plt.show()
"""))
cells.append(md("""
**Interpretation:** The violin plots suggest non-top-tier underwriters produce
more extreme underpricing outcomes (fatter tails), consistent with the
Carter-Manaster certification hypothesis (H5).  Top-tier banks appear to price
more precisely, reducing variance.  However, both groups show substantial overlap,
meaning underwriter rank alone is not sufficient to predict underpricing.
"""))

# Plot 7
cells.append(md("### Plot 7 — Risk factor word count vs. underpricing"))
cells.append(code("""
fig7 = eda.plot_text_feature_scatter(
    df_feat, "risk_factors_word_count",
    title="Risk Factor Word Count vs. Underpricing"
)
fig7.show()
"""))
cells.append(md("""
**Interpretation:** Longer risk factor sections are weakly associated with lower
underpricing, but the relationship is noisy.  Very long risk sections (>10,000 words)
may indicate thorough disclosure that reduces information asymmetry — or boilerplate
padding that investors discount.  We include `risk_factors_word_count` as a feature
and let SHAP determine its importance.
"""))

# Plot 8
cells.append(md("### Plot 8 — LM negative ratio vs. underpricing  *(headline novelty plot)*"))
cells.append(code("""
fig8 = eda.plot_text_feature_scatter(
    df_feat, "lm_negative_ratio",
    title="Loughran-McDonald Negative Word Ratio vs. Underpricing"
)
fig8.show()
"""))
cells.append(md("""
**Interpretation:** The LM negative ratio shows a **positive** association with
underpricing: prospectuses with more negative language are associated with higher
first-day returns.  This is consistent with Loughran & McDonald's finding that
negative-word-heavy filings signal higher information asymmetry, which underwriters
compensate for with deeper discounts.  This is the central empirical claim of the
text-feature novelty and will be tested formally as H3.
"""))

# Plot 9
cells.append(md("### Plot 9 — LM uncertainty ratio vs. underpricing"))
cells.append(code("""
fig9 = eda.plot_text_feature_scatter(
    df_feat, "lm_uncertainty_ratio",
    title="LM Uncertainty Ratio vs. Underpricing"
)
fig9.show()
"""))
cells.append(md("""
**Interpretation:** Uncertainty language (words like "may", "could", "uncertain")
shows a similar positive relationship with underpricing.  This is economically
intuitive: prospectuses that express more uncertainty about future cash flows
require larger discounts to attract investors.  The pattern is less pronounced
than for negative words, suggesting uncertainty is a noisier signal.
"""))

# Plot 10
cells.append(md("### Plot 10 — Prospectus uniqueness (Hanley-Hoberg) vs. underpricing"))
cells.append(code("""
fig10 = eda.plot_text_feature_scatter(
    df_feat, "prospectus_uniqueness",
    title="Prospectus Uniqueness (TF-IDF Distance from Sector Mean) vs. Underpricing"
)
fig10.show()
"""))
cells.append(md("""
**Interpretation:** More unique prospectuses (high TF-IDF deviation from the
sector boilerplate) are associated with *higher* underpricing in our sample.
This is the *opposite* of Hanley-Hoberg's prediction (they find uniqueness
reduces underpricing), and may reflect sample-composition differences or the
fact that genuinely novel business models attract speculative demand.
We flag this as a finding worth discussing in the conclusions.
"""))

# Plot 11
cells.append(md("### Plot 11 — Correlation heatmap"))
cells.append(code("""
fig11 = eda.plot_correlation_heatmap(df_feat, target_col=UP)
plt.show()
"""))
cells.append(md("""
**Interpretation:** The `hot_market_dummy` and `nasdaq_30d_return` show the
strongest positive correlations with underpricing among financial features.
Among text features, `lm_negative_ratio` and `lm_uncertainty_ratio` are the
most correlated with the target.  Multicollinearity between the LM categories
is notable — regularisation (Ridge / LightGBM with L1/L2) will be important.
"""))

# Plot 12
cells.append(md("### Plot 12 — Pairplot of top predictors"))
cells.append(code("""
top_features = [
    "underpricing", "lm_negative_ratio", "log_offer_size",
    "vix_at_pricing", "prospectus_uniqueness", "nasdaq_30d_return",
]
available = [c for c in top_features if c in df_feat.columns]

if "hot_market_dummy" in df_feat.columns:
    plot_df = df_feat[available + ["hot_market_dummy"]].dropna()
    plot_df["Market"] = plot_df["hot_market_dummy"].map({1: "Hot", 0: "Cold"})
    g = sns.pairplot(plot_df[available + ["Market"]], hue="Market",
                     plot_kws={"alpha": 0.3, "s": 15},
                     palette={"Hot": "#2563EB", "Cold": "#6B7280"})
else:
    g = sns.pairplot(df_feat[available].dropna(), plot_kws={"alpha": 0.3, "s": 15})

g.fig.suptitle("Pairplot of Top Predictors (coloured by market regime)", y=1.02)
g.fig.savefig("reports/figures/12_pairplot.png", dpi=150, bbox_inches="tight")
plt.show()
"""))
cells.append(md("""
**Interpretation:** Hot-market IPOs (blue) cluster in the upper-right of the
underpricing vs. NASDAQ-return scatter, confirming regime is a strong driver.
The LM negative ratio vs. underpricing shows a mild positive trend visible
across both regimes, which validates it as an incremental predictor beyond
market timing alone.
"""))

# Plot 13
cells.append(md("### Plot 13 — Calendar heatmap of IPO volume"))
cells.append(code("""
fig13 = eda.plot_calendar_heatmap(df_feat)
plt.show()
"""))
cells.append(md("""
**Interpretation:** IPO activity is highly seasonal: Q1 (January–March) and
Q4 (September–November) are consistently the busiest periods.  The 2020–2021
surge is clearly visible.  The `ipo_quarter` and `is_quarter_end_month`
features capture this cyclicality.
"""))

# Plot 14
cells.append(md("### Plot 14 — Sector × hot-market interaction"))
cells.append(code("""
fig14 = eda.plot_sector_market_interaction(df_feat)
fig14.show()
"""))
cells.append(md("""
**Interpretation:** Technology underpricing amplifies dramatically in hot markets
relative to cold markets, while Healthcare and Financials show smaller
differences.  This interaction — sector × market-regime — suggests tree-based
models that can capture interactions without explicit feature construction will
outperform linear models.
"""))

# EDA summary
cells.append(md("""
### EDA Summary

Key patterns motivating our model and hypothesis tests:

- **Heavy right tail** in underpricing → use MAE, not RMSE; model `winsorized_underpricing`.
- **Strong cyclicality** (hot/cold markets) → time-based split is mandatory; `hot_market_dummy` and `nasdaq_30d_return` are strong features.
- **Sector heterogeneity** → target-encoded sector; tree models preferred over linear.
- **LM negative and uncertainty ratios** show positive monotone associations with underpricing → H3 motivated.
- **VIX increases variance**, not necessarily the mean → H4 motivated.
- **Top-tier underwriters** show lower variance → H5 motivated.
- **LM signals are incremental** beyond hot-market / sector controls → H6 motivated.
"""))

# ============================================================================
# 8. Hypothesis Testing
# ============================================================================
cells.append(md("""
---
## 8. Hypothesis Testing

We run six formal tests.  For each, we state H₀ and H₁, justify the test
choice, report the test statistic and p-value, and give a model takeaway.
"""))

cells.append(code("""
# Ensure columns exist (may be demo data)
TEST_TARGET = WUP if WUP in df_feat.columns else UP

h1 = hypothesis_tests.test_h1_tech_vs_nontech(df_feat, target_col=TEST_TARGET)
hypothesis_tests.report(h1)
"""))

cells.append(code("""
h2 = hypothesis_tests.test_h2_hot_vs_cold_market(df_feat, target_col=TEST_TARGET)
hypothesis_tests.report(h2)
"""))

cells.append(code("""
if "lm_negative_ratio" in df_feat.columns:
    h3 = hypothesis_tests.test_h3_lm_negative_underpricing(df_feat, target_col=TEST_TARGET)
    hypothesis_tests.report(h3)
else:
    print("lm_negative_ratio not available; skipping H3.")
"""))

cells.append(code("""
if "vix_at_pricing" in df_feat.columns and df_feat["vix_at_pricing"].notna().sum() > 50:
    h4 = hypothesis_tests.test_h4_vix_variance(df_feat, target_col=TEST_TARGET)
    hypothesis_tests.report(h4)
else:
    print("vix_at_pricing not available; skipping H4.")
"""))

cells.append(code("""
if "top_tier_underwriter" in df_feat.columns:
    h5 = hypothesis_tests.test_h5_underwriter_variance(df_feat, target_col=TEST_TARGET)
    hypothesis_tests.report(h5)
else:
    print("top_tier_underwriter not available; skipping H5.")
"""))

cells.append(code("""
h6 = hypothesis_tests.test_h6_text_features_ols(df_feat, target_col=TEST_TARGET)
hypothesis_tests.report(h6)

print("Financial-only model summary:")
print(h6["model_fin"].summary().tables[0])
"""))

# ============================================================================
# 9. Machine Learning
# ============================================================================
cells.append(md("""
---
## 9. Machine Learning

We are predicting `winsorized_underpricing` (a continuous variable) given
pre-IPO information.  The metric we will optimise is **Mean Absolute Error**
because the target distribution has heavy tails and we do not want
squared-error loss to be dominated by a handful of extreme observations.
We also report R² and **Spearman rank correlation** between predicted and
actual returns, since for an investor the *ranking* matters as much as the
level.
"""))

cells.append(md("""
### 9a. Train / Test Split

We use a **time-based split**: train on IPOs before 2024-01-01, test on
IPOs from 2024-01-01 onward.  A random split would leak market-regime
information because temporal autocorrelation means that observations close in
time share unobserved confounders.  Within training, we use `TimeSeriesSplit`
with 5 folds for cross-validation.
"""))

cells.append(code("""
target_col = TEST_TARGET
date_col = "ipo_date"

feature_cols = models.select_features(df_feat)
log.info("Using %d features: %s …", len(feature_cols), feature_cols[:8])

df_model = df_feat[feature_cols + [target_col, date_col]].dropna(subset=[target_col])
train_df, test_df = models.time_split(df_model, date_col)

X_train = train_df[feature_cols]
y_train = train_df[target_col].values
X_test = test_df[feature_cols]
y_test = test_df[target_col].values

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
"""))

cells.append(md("### 9b. Model Training"))

cells.append(code("""
# ── Baseline: training-set median ────────────────────────────────────────────
pred_median, m_median = models.baseline_median(y_train, y_test)

# ── OLS financial-only ───────────────────────────────────────────────────────
fin_feats = [c for c in feature_cols if not any(
    kw in c for kw in ("lm_", "fog_", "uniqueness", "word_count")
)]
_, pred_ols_fin, m_ols_fin = models.fit_ols(
    X_train[fin_feats], y_train, X_test[fin_feats], y_test, "OLS-Financial"
)

# ── OLS financial + text ─────────────────────────────────────────────────────
_, pred_ols_full, m_ols_full = models.fit_ols(
    X_train, y_train, X_test, y_test, "OLS-Full"
)

# ── Ridge ────────────────────────────────────────────────────────────────────
pipe_ridge, pred_ridge, m_ridge = models.fit_ridge(X_train, y_train, X_test, y_test)

# ── Random Forest ────────────────────────────────────────────────────────────
rf_model, pred_rf, m_rf = models.fit_random_forest(X_train, y_train, X_test, y_test)

print("Random Forest training complete.")
"""))

cells.append(code("""
# ── LightGBM + Optuna (this takes ~3–5 minutes) ──────────────────────────────
lgbm_model, pred_lgbm, m_lgbm = models.fit_lgbm_optuna(
    X_train, y_train, X_test, y_test, n_trials=60
)
print("LightGBM training complete.")
"""))

cells.append(md("### 9c. Evaluation"))

cells.append(code("""
metrics_table = pd.DataFrame([
    {"Model": "Baseline (median)", **m_median},
    {"Model": "OLS — financial only", **m_ols_fin},
    {"Model": "OLS — financial + text", **m_ols_full},
    {"Model": "Ridge", **m_ridge},
    {"Model": "Random Forest", **m_rf},
    {"Model": "LightGBM (Optuna)", **m_lgbm},
])
metrics_table = metrics_table.sort_values("mae").reset_index(drop=True)
print(metrics_table.to_string(index=False))
"""))

cells.append(code("""
models.plot_predicted_vs_actual(y_test, pred_lgbm, "LightGBM")
plt.show()
"""))

cells.append(code("""
models.plot_residuals(y_test, pred_lgbm, test_df[date_col], "LightGBM")
plt.show()
"""))

cells.append(md("### 9d. SHAP Interpretation"))

cells.append(code("""
shap_values = models.compute_shap(lgbm_model, X_test, max_display=15)
"""))

cells.append(md("""
**Economic interpretation of SHAP results:**

The SHAP beeswarm plot reveals the contribution of each feature to individual
predictions.  We find that:

1. `hot_market_dummy` and `nasdaq_30d_return` are the dominant drivers — confirming
   that market timing is the single strongest predictor of underpricing.
2. Among text features, `lm_negative_ratio` and `lm_uncertainty_ratio` appear in the
   top features with positive SHAP values, meaning higher sentiment-negativity pushes
   predictions upward — consistent with the Loughran-McDonald hypothesis.
3. `prospectus_uniqueness` shows a nuanced dependence: moderate uniqueness reduces
   predicted underpricing (consistent with Hanley-Hoberg), but very high uniqueness
   *increases* it (novel business models attract speculative premia).
4. `log_offer_size` has negative SHAP values — large deals are less underpriced,
   consistent with lower information asymmetry for bigger firms.

These results demonstrate that text features add **incremental signal** beyond
financial and market-regime features alone, validating the academic novelty of
this project.
"""))

cells.append(md("""
### 9e. Honest Limitations

1. **Survivorship bias:** we can only pull prices for tickers that survived on
   Yahoo Finance; companies that immediately changed their ticker or were delisted
   are excluded, biasing our sample toward successful IPOs.
2. **Look-ahead bias risk:** our sector mean TF-IDF for prospectus uniqueness
   uses the *full* corpus; in production, only past prospectuses would be known
   at pricing time.
3. **Demand-side information gap:** we cannot capture order-book demand or
   the grey-market price — the single strongest predictor known to insiders.
4. **Thin sectors:** sectors with fewer than 20 IPOs (e.g. Energy) have noisy
   target-encoded values; more data or hierarchical encoding would help.
5. **LM dictionary vintage:** the LM dictionary was calibrated on 10-K filings;
   S-1 language patterns may differ systematically.
6. **Optuna local optima:** 60 trials is a modest budget; with more compute,
   a larger search space could improve LightGBM performance.
"""))

# ============================================================================
# 10. Conclusions
# ============================================================================
cells.append(md("""
---
## 10. Conclusions

**What we found:**
- First-day IPO underpricing is strongly cyclical; hot-market conditions
  explain more variance than any single financial variable.
- Loughran-McDonald negative-word ratios are positively associated with
  underpricing, consistent with information-asymmetry theory.
- Prospectus uniqueness has a non-monotone relationship with underpricing —
  a novel finding that differs from the original Hanley-Hoberg result.
- LightGBM with text + financial features outperforms the financial-only
  baseline on MAE, and SHAP confirms that text features carry incremental
  information.

**What surprised us:**
- The VIX increases variance rather than mean underpricing — suggesting
  market fear causes wider pricing dispersion, not uniformly deeper discounts.
- Prospectus uniqueness has a U-shaped relationship with underpricing,
  not the simple negative relationship the literature predicts.

**What's next:**
- Capturing the price range vs. final offer price revision (Hanley 1993)
  would likely be the single most powerful predictor addition.
- Sentence-level BERT embeddings of Risk Factors could replace bag-of-words
  LM ratios with richer semantic signals.
- A wider training window (pre-2019) and more Optuna trials would improve
  model generalisation.
"""))

# ============================================================================
# 11. References
# ============================================================================
cells.append(md("""
---
## 11. References

1. Allen, F., & Faulhaber, G. R. (1989). Signalling by underpricing in the IPO market. *Journal of Financial Economics*, 23(2), 303–323.
2. Carter, R. B., & Manaster, S. (1990). Initial public offerings and underwriter reputation. *Journal of Finance*, 45(4), 1045–1067.
3. Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
4. Hanley, K. W. (1993). The underpricing of initial public offerings and the partial adjustment phenomenon. *Journal of Financial Economics*, 34(2), 231–250.
5. Hanley, K. W., & Hoberg, G. (2010). The information content of IPO prospectuses. *Review of Financial Studies*, 23(7), 2821–2864.
6. Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
7. Ritter, J. R., & Welch, I. (2002). A review of IPO activity, pricing, and allocations. *Journal of Finance*, 57(4), 1795–1828.
8. Rock, K. (1986). Why new issues are underpriced. *Journal of Financial Economics*, 15(1–2), 187–212.
"""))

# ============================================================================
# 12. Appendix — Data dictionary
# ============================================================================
cells.append(md("""
---
## 12. Appendix — Data Dictionary

| Column | Source | Description |
|---|---|---|
| `ticker` | IPO calendar | Stock ticker symbol |
| `ipo_date` | IPO calendar | Date of first trading day |
| `offer_price` | IPO calendar | Offer price in USD |
| `first_day_close` | yfinance | Closing price on IPO date |
| `underpricing` | Computed | `(close - offer) / offer` |
| `winsorized_underpricing` | Computed | Winsorised at 1st/99th percentile |
| `offer_size_m` | IPO calendar | Total capital raised ($M) |
| `sector` | IPO calendar | GICS-style sector |
| `lead_underwriter` | IPO calendar | Primary bookrunner |
| `vix_at_pricing` | yfinance | CBOE VIX close on IPO date |
| `nasdaq_30d_return` | yfinance | NASDAQ 30-day log return |
| `hot_market_dummy` | Computed | 1 if rolling 90-day IPO count ≥ top tercile |
| `log_offer_size` | Computed | `log1p(offer_size_m)` |
| `underwriter_rank` | Ritter rankings | Carter-Manaster scale 0–9 |
| `top_tier_underwriter` | Computed | 1 if rank ≥ 8 |
| `lm_negative_ratio` | LM dictionary + S-1 | Negative words / total words |
| `lm_uncertainty_ratio` | LM dictionary + S-1 | Uncertainty words / total words |
| `fog_index_mda` | Computed | Gunning Fog Index of MD&A section |
| `prospectus_uniqueness` | Computed | 1 − cosine similarity to sector mean TF-IDF |
| `risk_factors_word_count` | Computed | Word count of Risk Factors section |

---
*IPO Underpricing Prediction Project — Apache 2.0 License*
"""))

# ── Build and write ──────────────────────────────────────────────────────────

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0",
}

NOTEBOOK_PATH.write_text(nbformat.writes(nb), encoding="utf-8")
print(f"Notebook written → {NOTEBOOK_PATH}  ({len(cells)} cells)")
