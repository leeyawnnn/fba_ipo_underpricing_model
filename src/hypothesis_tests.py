"""
Hypothesis testing for IPO underpricing.

Implements all six hypothesis tests required by the rubric:

  H1 — Tech IPOs are more underpriced than non-tech (Mann-Whitney U + KS).
  H2 — Hot-market IPOs are more underpriced (bootstrap difference-of-medians).
  H3 — Higher LM negative ratio → higher underpricing (Spearman + Kruskal-Wallis).
  H4 — VIX affects the *variance* of underpricing (Levene's test).
  H5 — Top-tier underwriters reduce underpricing variance (Levene's test).
  H6 — Text features improve OLS fit beyond financial features (LR test + AIC/BIC).

Each function returns a structured results dict for easy display in the
notebook.  The ``report()`` helper prints a formatted summary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.utils import setup_logging

log = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def report(result: dict[str, Any]) -> None:
    """Print a formatted hypothesis test result to the log.

    Args:
        result: Dict returned by any of the test functions in this module.
    """
    print("=" * 60)
    print(f"  {result.get('hypothesis', '')}")
    print("=" * 60)
    print(f"  H0: {result.get('h0', 'N/A')}")
    print(f"  H1: {result.get('h1', 'N/A')}")
    print(f"  Test: {result.get('test', 'N/A')}")
    print(f"  Statistic: {result.get('statistic', 'N/A'):.4g}")
    print(f"  p-value: {result.get('p_value', 'N/A'):.4g}")
    if "effect_size" in result:
        print(f"  Effect size: {result['effect_size']:.4g}")
    if "ci_95" in result:
        lo, hi = result["ci_95"]
        print(f"  95% CI: [{lo:.4g}, {hi:.4g}]")
    print(f"  Decision: {'Reject H0' if result.get('reject_h0') else 'Fail to reject H0'}")
    print(f"  Interpretation: {result.get('interpretation', '')}")
    print(f"  ML takeaway: {result.get('ml_takeaway', '')}")
    print()


# ---------------------------------------------------------------------------
# H1: Tech vs. non-tech underpricing
# ---------------------------------------------------------------------------

def test_h1_tech_vs_nontech(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    sector_col: str = "sector",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Mann-Whitney U and KS test: tech IPOs more underpriced than non-tech?

    Args:
        df: IPO DataFrame.
        target_col: Target variable column.
        sector_col: Sector column.
        alpha: Significance level.

    Returns:
        Results dict.
    """
    data = df[[target_col, sector_col]].dropna()

    tech_mask = data[sector_col].str.lower().str.contains(
        r"tech|software|semiconductor|internet|information", na=False
    )
    tech = data.loc[tech_mask, target_col].values
    non_tech = data.loc[~tech_mask, target_col].values

    mw_stat, mw_p = stats.mannwhitneyu(tech, non_tech, alternative="greater")
    ks_stat, ks_p = stats.ks_2samp(tech, non_tech)

    # Rank-biserial correlation as effect size for Mann-Whitney
    n1, n2 = len(tech), len(non_tech)
    effect_size = 1 - (2 * mw_stat) / (n1 * n2)

    return {
        "hypothesis": "H1: Tech IPOs more underpriced than non-tech",
        "h0": "Tech and non-tech IPOs have equal median underpricing",
        "h1": "Tech IPOs have higher median underpricing (one-sided)",
        "test": "Mann-Whitney U (one-sided) + KS (two-sided)",
        "statistic": mw_stat,
        "p_value": mw_p,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "effect_size": effect_size,
        "n_tech": n1,
        "n_non_tech": n2,
        "tech_median": float(np.median(tech)),
        "non_tech_median": float(np.median(non_tech)),
        "reject_h0": mw_p < alpha,
        "interpretation": (
            f"Tech IPOs (n={n1}) have median underpricing "
            f"{np.median(tech):.1%} vs. {np.median(non_tech):.1%} for non-tech (n={n2}). "
            f"Mann-Whitney p={mw_p:.4f}; KS p={ks_p:.4f}."
        ),
        "ml_takeaway": (
            "Sector dummy (tech/non-tech) is likely a useful feature; "
            "target-encoded sector should carry this signal."
        ),
    }


# ---------------------------------------------------------------------------
# H2: Hot vs. cold market underpricing (bootstrap)
# ---------------------------------------------------------------------------

def test_h2_hot_vs_cold_market(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    hot_col: str = "hot_market_dummy",
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap difference-of-medians: hot markets more underpriced?

    Args:
        df: IPO DataFrame.
        target_col: Target variable.
        hot_col: Binary hot-market column.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level.
        rng_seed: Random seed for reproducibility.

    Returns:
        Results dict.
    """
    data = df[[target_col, hot_col]].dropna()
    hot = data.loc[data[hot_col] == 1, target_col].values
    cold = data.loc[data[hot_col] == 0, target_col].values

    observed_diff = np.median(hot) - np.median(cold)

    rng = np.random.default_rng(rng_seed)
    boot_diffs = []
    for _ in range(n_bootstrap):
        h_sample = rng.choice(hot, size=len(hot), replace=True)
        c_sample = rng.choice(cold, size=len(cold), replace=True)
        boot_diffs.append(np.median(h_sample) - np.median(c_sample))

    boot_diffs = np.array(boot_diffs)
    p_value = float(np.mean(boot_diffs <= 0))  # one-sided: hot > cold
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "hypothesis": "H2: Hot-market IPOs are more underpriced",
        "h0": "Median underpricing is equal in hot and cold markets",
        "h1": "Hot-market IPOs have higher median underpricing (one-sided)",
        "test": f"Bootstrap difference-of-medians (B={n_bootstrap:,})",
        "statistic": observed_diff,
        "p_value": p_value,
        "ci_95": (ci_lo, ci_hi),
        "n_hot": len(hot),
        "n_cold": len(cold),
        "hot_median": float(np.median(hot)),
        "cold_median": float(np.median(cold)),
        "reject_h0": p_value < alpha,
        "interpretation": (
            f"Observed median difference: {observed_diff:.1%}. "
            f"Bootstrap 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]. "
            f"p={p_value:.4f}."
        ),
        "ml_takeaway": (
            "The hot-market dummy and rolling IPO count are strong candidate "
            "features; market-regime lags should be included."
        ),
    }


# ---------------------------------------------------------------------------
# H3: LM negative ratio and underpricing (Spearman + Kruskal-Wallis)
# ---------------------------------------------------------------------------

def test_h3_lm_negative_underpricing(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    lm_col: str = "lm_negative_ratio",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Spearman rank correlation and Kruskal-Wallis test for H3.

    Args:
        df: IPO DataFrame.
        target_col: Target variable.
        lm_col: LM negative word ratio column.
        alpha: Significance level.

    Returns:
        Results dict.
    """
    data = df[[target_col, lm_col]].dropna()
    rho, spearman_p = stats.spearmanr(data[lm_col], data[target_col])

    # Tercile split and Kruskal-Wallis
    data["_tercile"] = pd.qcut(data[lm_col], q=3, labels=["Low", "Mid", "High"])
    groups = [grp[target_col].values for _, grp in data.groupby("_tercile", observed=True)]
    kw_stat, kw_p = stats.kruskal(*groups)

    return {
        "hypothesis": "H3: Higher LM negative ratio → higher underpricing",
        "h0": "No monotone association between LM negative ratio and underpricing",
        "h1": "Positive Spearman rank correlation (one-sided)",
        "test": "Spearman ρ (one-sided) + Kruskal-Wallis across terciles",
        "statistic": rho,
        "p_value": spearman_p / 2,   # one-sided
        "effect_size": rho,
        "kw_statistic": kw_stat,
        "kw_p_value": kw_p,
        "n": len(data),
        "reject_h0": (spearman_p / 2) < alpha,
        "interpretation": (
            f"Spearman ρ={rho:.3f} (p={spearman_p/2:.4f}, one-sided). "
            f"Kruskal-Wallis across terciles: H={kw_stat:.2f}, p={kw_p:.4f}."
        ),
        "ml_takeaway": (
            "LM negative ratio should appear as a feature; its monotone "
            "relationship supports tree-based models over linear ones."
        ),
    }


# ---------------------------------------------------------------------------
# H4: VIX and variance of underpricing (Levene)
# ---------------------------------------------------------------------------

def test_h4_vix_variance(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    vix_col: str = "vix_at_pricing",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Levene's test: does high VIX increase variance of underpricing?

    Args:
        df: IPO DataFrame.
        target_col: Target variable.
        vix_col: VIX column.
        alpha: Significance level.

    Returns:
        Results dict.
    """
    data = df[[target_col, vix_col]].dropna()
    data["_vix_tercile"] = pd.qcut(data[vix_col], q=3, labels=["Low", "Mid", "High"])

    groups = [grp[target_col].values for _, grp in data.groupby("_vix_tercile", observed=True)]
    lev_stat, lev_p = stats.levene(*groups, center="median")

    group_stds = {
        label: grp[target_col].std()
        for label, grp in data.groupby("_vix_tercile", observed=True)
    }

    return {
        "hypothesis": "H4: VIX affects the variance of underpricing",
        "h0": "Variance of underpricing is equal across VIX terciles",
        "h1": "Variance differs across VIX terciles",
        "test": "Levene's test (median-centered) across VIX terciles",
        "statistic": lev_stat,
        "p_value": lev_p,
        "group_stds": group_stds,
        "n": len(data),
        "reject_h0": lev_p < alpha,
        "interpretation": (
            f"Levene W={lev_stat:.2f}, p={lev_p:.4f}. "
            f"Std by VIX tercile: {', '.join(f'{k}={v:.2%}' for k, v in group_stds.items())}."
        ),
        "ml_takeaway": (
            "Heteroscedasticity with VIX suggests quantile regression or "
            "log-transformation of the target may improve model calibration."
        ),
    }


# ---------------------------------------------------------------------------
# H5: Underwriter tier and variance (Levene)
# ---------------------------------------------------------------------------

def test_h5_underwriter_variance(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    tier_col: str = "top_tier_underwriter",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Levene's test: do top-tier underwriters reduce underpricing variance?

    Args:
        df: IPO DataFrame.
        target_col: Target variable.
        tier_col: Binary top-tier underwriter column.
        alpha: Significance level.

    Returns:
        Results dict.
    """
    data = df[[target_col, tier_col]].dropna()
    top = data.loc[data[tier_col] == 1, target_col].values
    non_top = data.loc[data[tier_col] == 0, target_col].values

    lev_stat, lev_p = stats.levene(top, non_top, center="median")

    return {
        "hypothesis": "H5: Top-tier underwriters reduce underpricing variance",
        "h0": "Variance is equal for top-tier and non-top-tier underwriter groups",
        "h1": "Top-tier underwriters produce lower variance (certification hypothesis)",
        "test": "Levene's test (median-centered)",
        "statistic": lev_stat,
        "p_value": lev_p,
        "top_tier_std": float(np.std(top)),
        "non_top_tier_std": float(np.std(non_top)),
        "n_top": len(top),
        "n_non_top": len(non_top),
        "reject_h0": lev_p < alpha,
        "interpretation": (
            f"Top-tier σ={np.std(top):.2%}, non-top-tier σ={np.std(non_top):.2%}. "
            f"Levene W={lev_stat:.2f}, p={lev_p:.4f}. "
            "Consistent with Carter-Manaster (1990) certification hypothesis "
            "if top-tier variance is lower and H0 is rejected."
        ),
        "ml_takeaway": (
            "top_tier_underwriter dummy and underwriter_rank are likely "
            "informative features; their effect may be non-linear."
        ),
    }


# ---------------------------------------------------------------------------
# H6: Text features improve OLS fit (likelihood ratio test)
# ---------------------------------------------------------------------------

def test_h6_text_features_ols(
    df: pd.DataFrame,
    target_col: str = "winsorized_underpricing",
    financial_features: list[str] | None = None,
    text_features: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Likelihood ratio test comparing financial-only vs. financial+text OLS.

    Args:
        df: IPO DataFrame with feature columns.
        target_col: Target variable (winsorized underpricing recommended).
        financial_features: List of financial feature column names.  Defaults
            to a standard set if ``None``.
        text_features: List of text feature column names.  Defaults to
            LM ratios + readability + uniqueness if ``None``.
        alpha: Significance level.

    Returns:
        Results dict including adjusted R², AIC, BIC, and LR test outcome.
    """
    if financial_features is None:
        financial_features = [
            c for c in [
                "log_offer_size", "underwriter_rank", "top_tier_underwriter",
                "vix_at_pricing", "nasdaq_30d_return", "hot_market_dummy",
                "ipo_year", "ipo_quarter", "sector_encoded",
            ]
            if c in df.columns
        ]

    if text_features is None:
        text_features = [
            c for c in [
                "lm_negative_ratio", "lm_uncertainty_ratio",
                "lm_litigious_ratio", "lm_positive_ratio",
                "fog_index_mda", "prospectus_uniqueness",
                "risk_factors_word_count", "total_prospectus_word_count",
            ]
            if c in df.columns
        ]

    all_features = financial_features + text_features
    data = df[[target_col] + all_features].dropna()

    if len(data) < 50:
        log.warning("Only %d complete rows for H6; results may be unreliable.", len(data))

    y = data[target_col]

    # Restricted model (financial only)
    X_fin = sm.add_constant(data[financial_features].astype(float))
    model_fin = sm.OLS(y, X_fin).fit()

    # Unrestricted model (financial + text)
    X_full = sm.add_constant(data[all_features].astype(float))
    model_full = sm.OLS(y, X_full).fit()

    # Likelihood ratio test statistic
    lr_stat = 2 * (model_full.llf - model_fin.llf)
    df_diff = model_full.df_model - model_fin.df_model
    lr_p = float(stats.chi2.sf(lr_stat, df=df_diff))

    return {
        "hypothesis": "H6: Text features improve OLS explanatory power",
        "h0": "Adding text features does not improve model fit (LR test)",
        "h1": "Unrestricted model (financial + text) fits significantly better",
        "test": f"Likelihood ratio test (χ²({int(df_diff)}))",
        "statistic": lr_stat,
        "p_value": lr_p,
        "df_diff": int(df_diff),
        "fin_adj_r2": model_fin.rsquared_adj,
        "full_adj_r2": model_full.rsquared_adj,
        "fin_aic": model_fin.aic,
        "full_aic": model_full.aic,
        "fin_bic": model_fin.bic,
        "full_bic": model_full.bic,
        "n": len(data),
        "financial_features": financial_features,
        "text_features": text_features,
        "reject_h0": lr_p < alpha,
        "model_fin": model_fin,
        "model_full": model_full,
        "interpretation": (
            f"LR statistic={lr_stat:.2f} (χ²({int(df_diff)})), p={lr_p:.4f}. "
            f"Adj R²: financial-only={model_fin.rsquared_adj:.3f}, "
            f"with text={model_full.rsquared_adj:.3f}. "
            f"ΔAIC={model_fin.aic - model_full.aic:.1f}, "
            f"ΔBIC={model_fin.bic - model_full.bic:.1f}."
        ),
        "ml_takeaway": (
            "If H0 is rejected, text features carry incremental signal and "
            "should be included in the ML feature set alongside financial predictors."
        ),
    }
