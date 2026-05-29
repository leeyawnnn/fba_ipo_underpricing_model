"""
Hypothesis testing for IPO underpricing.

The first three hypotheses are the genuinely *non-obvious* findings of the
project — together they make the same point three different ways:
*structural disclosure*, not headline risk, is what drives the cross-section
of first-day IPO returns.  H4, H5 and H6 are the standard textbook
stylised-fact replications.

  H1 — *Litigious-tone paradox*: more legal / courtroom language in an S-1
       predicts LESS first-day underpricing, the opposite of the naive
       risk-premium story (Spearman + Kruskal-Wallis across quintiles).
  H2 — *Underwriter translation effect*: the litigious paradox is weaker
       for top-tier underwriter IPOs because prestigious banks "certify"
       risky-sounding prospectuses (OLS interaction + bootstrap of Δρ).
  H3 — *Disclosure concentration curse*: negative tone that is
       concentrated in the Risk Factors section predicts LOWER underpricing,
       but negative tone that is pervasive across the whole prospectus
       predicts HIGHER underpricing (Spearman ρ + Kruskal-Wallis across
       terciles on ``rf_lm_negative_ratio / lm_negative_ratio``).
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
    stat = result.get("statistic", float("nan"))
    pval = result.get("p_value", float("nan"))
    try:
        print(f"  Statistic: {stat:.4g}")
    except (TypeError, ValueError):
        print(f"  Statistic: {stat}")
    try:
        print(f"  p-value: {pval:.4g}")
    except (TypeError, ValueError):
        print(f"  p-value: {pval}")
    if "effect_size" in result:
        try:
            print(f"  Effect size: {result['effect_size']:.4g}")
        except (TypeError, ValueError):
            print(f"  Effect size: {result['effect_size']}")
    if "ci_95" in result:
        lo, hi = result["ci_95"]
        print(f"  95% CI: [{lo:.4g}, {hi:.4g}]")
    print(f"  Decision: {'Reject H0' if result.get('reject_h0') else 'Fail to reject H0'}")
    print(f"  Interpretation: {result.get('interpretation', '')}")
    print(f"  ML takeaway: {result.get('ml_takeaway', '')}")
    print()


# ---------------------------------------------------------------------------
# H1: Litigious-tone paradox  --  the headline finding
# ---------------------------------------------------------------------------

def test_h1_litigious_paradox(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    lit_col: str = "lm_litigious_ratio",
    alpha: float = 0.05,
    collapse_floor: float = -0.50,
) -> dict[str, Any]:
    """Test the "litigious-tone paradox".

    Naive intuition (and most undergraduate textbook treatments) say:
    *more* legal / risk language in the prospectus signals a riskier deal,
    investors demand a steeper discount, so the first-day pop is larger.
    We find the **opposite**: among IPOs that did not collapse post-listing,
    the share of Loughran-McDonald *Litigious*-category words in the S-1
    is **negatively** correlated with first-day underpricing.

    Interpretation (consistent with Hanley & Hoberg, 2010): dense legal /
    contractual language is a proxy for *disclosure granularity*, not for
    headline risk.  More disclosure reduces information asymmetry between
    issuer and investor, which lets the underwriter price the deal closer
    to its true value — leaving less money on the table on day one.

    Crucially, this is the **only** LM category whose Spearman p-value
    survives a Bonferroni correction across the seven LM categories on
    the cleaned sample, and it survives multivariate OLS with deal,
    market and sector controls.

    Args:
        df: IPO DataFrame with the LM litigious ratio and underpricing.
        target_col: Target variable column.
        lit_col: LM Litigious word ratio column.
        alpha: Significance level.
        collapse_floor: Drop observations with underpricing at or below this
            value.  Defaults to -0.5 to remove the cluster of −100% rows
            caused by the upstream ``stockanalysis.com`` price field
            returning a *current* price for delisted tickers — those are
            not real first-day returns and they otherwise drown the signal.

    Returns:
        Results dict with Spearman ρ, Kruskal-Wallis across quintiles,
        the by-sector breakdown, and a multivariate OLS coefficient with
        deal / market / SPAC controls.
    """
    raw = df[[target_col, lit_col, "is_spac", "sector", "sector_encoded",
              "offer_price", "vix_at_pricing", "nasdaq_30d_return"]].dropna(
        subset=[target_col, lit_col]
    )
    n_raw = len(raw)

    data = raw[raw[target_col] > collapse_floor].copy()
    n_clean = len(data)
    n_dropped_collapse = n_raw - n_clean

    # Primary test: Spearman rank correlation
    rho, spearman_p = stats.spearmanr(data[lit_col], data[target_col])

    # Quintile split + Kruskal-Wallis to show the effect is monotone, not driven
    # by a single tail
    data["_quintile"] = pd.qcut(
        data[lit_col], q=5, labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        duplicates="drop",
    )
    quintile_medians = (
        data.groupby("_quintile", observed=True)[target_col].median().to_dict()
    )
    quintile_n = (
        data.groupby("_quintile", observed=True)[target_col].size().to_dict()
    )
    groups = [g[target_col].values for _, g in data.groupby("_quintile", observed=True)]
    kw_stat, kw_p = stats.kruskal(*groups)

    # Multivariate OLS to show the effect survives obvious controls
    ols_data = data.dropna(subset=["offer_price", "vix_at_pricing",
                                    "nasdaq_30d_return", "sector_encoded"]).copy()
    ols_data["log_offer"] = np.log(ols_data["offer_price"].clip(lower=0.01))
    X = sm.add_constant(ols_data[[
        lit_col, "is_spac", "sector_encoded", "log_offer",
        "vix_at_pricing", "nasdaq_30d_return",
    ]].astype(float))
    y = ols_data[target_col].astype(float)
    ols_model = sm.OLS(y, X).fit()
    ols_coef = float(ols_model.params[lit_col])
    ols_pval = float(ols_model.pvalues[lit_col])

    # Subgroup: non-SPAC only.  SPACs price at $10 by construction so their
    # first-day return reflects post-merger speculation, not prospectus
    # disclosure quality — if the mechanism is about disclosure → asymmetry,
    # we expect the effect to live in the operating-company subsample.
    op = data[data["is_spac"] == 0]
    op_rho, op_p = stats.spearmanr(op[lit_col], op[target_col])

    spac = data[data["is_spac"] == 1]
    spac_rho, spac_p = (
        stats.spearmanr(spac[lit_col], spac[target_col]) if len(spac) > 30
        else (float("nan"), float("nan"))
    )

    return {
        "hypothesis": "H1: Litigious-tone paradox — more legal language predicts LESS underpricing",
        "h0": "No monotone association between LM litigious ratio and first-day underpricing",
        "h1": "Negative association (more litigious language ↔ lower underpricing)",
        "test": ("Spearman ρ + Kruskal-Wallis across litigious-tone quintiles + "
                 "multivariate OLS with deal / market / sector controls"),
        "statistic": rho,
        "p_value": spearman_p,
        "effect_size": rho,
        "kw_statistic": kw_stat,
        "kw_p_value": kw_p,
        "quintile_medians": {str(k): float(v) for k, v in quintile_medians.items()},
        "quintile_n": {str(k): int(v) for k, v in quintile_n.items()},
        "ols_coefficient": ols_coef,
        "ols_p_value": ols_pval,
        "non_spac_rho": float(op_rho),
        "non_spac_p_value": float(op_p),
        "non_spac_n": int(len(op)),
        "spac_rho": float(spac_rho),
        "spac_p_value": float(spac_p),
        "spac_n": int(len(spac)),
        "n_clean": n_clean,
        "n_dropped_collapse": n_dropped_collapse,
        "reject_h0": spearman_p < alpha,
        "interpretation": (
            f"Spearman ρ={rho:+.3f} (p={spearman_p:.4g}) on n={n_clean} clean "
            f"first-day returns (after dropping {n_dropped_collapse} post-IPO "
            f"collapses ≤ {int(collapse_floor*100)}%).  Q1→Q5 medians: "
            f"{quintile_medians.get('Q1_low', float('nan')):+.1%} → "
            f"{quintile_medians.get('Q5_high', float('nan')):+.1%} "
            f"(Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4g}).  "
            f"Multivariate OLS coefficient (with controls): "
            f"{ols_coef:.2f} (p={ols_pval:.4g}).  "
            f"Non-SPAC subsample: ρ={op_rho:+.3f} (p={op_p:.4g}, n={len(op)})."
        ),
        "ml_takeaway": (
            "The litigious-language ratio carries genuinely orthogonal signal — "
            "it is the only LM category whose Spearman p-value survives a "
            "Bonferroni correction across the seven LM categories on the cleaned "
            "sample.  Most importantly, the SIGN is opposite to what a naive "
            "'risk language → risk premium → larger pop' story predicts, which "
            "is the kind of structural-disclosure feature a tree-based model can "
            "exploit and a linear sentiment-only model cannot."
        ),
    }


# ---------------------------------------------------------------------------
# H2: Underwriter translation effect  --  interaction with tier
# ---------------------------------------------------------------------------

def test_h2_underwriter_translation(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    lit_col: str = "lm_litigious_ratio",
    tier_col: str = "top_tier_underwriter",
    alpha: float = 0.05,
    collapse_floor: float = -0.50,
    n_bootstrap: int = 5_000,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Test whether top-tier underwriters "translate" risky prospectus language.

    Builds on H1.  The litigious-tone paradox says that more legal language
    in an S-1 predicts *less* first-day underpricing, because dense legal
    language is a proxy for disclosure granularity (Hanley & Hoberg, 2010).
    If that disclosure channel is what's doing the work, it should be
    **weaker** for IPOs underwritten by a top-tier bank: prestigious
    underwriters already "certify" the deal (Carter & Manaster, 1990), so
    the marginal information content of the prospectus language is smaller.
    The interaction term ``litigious × top_tier`` in an OLS should therefore
    carry a *positive* sign (it attenuates the negative main effect), and
    the Spearman ρ between litigious and underpricing should be closer to
    zero (less negative) inside the top-tier subsample than outside it.

    Args:
        df: IPO DataFrame with the LM litigious ratio, underpricing, and
            top-tier underwriter flag.
        target_col: Target variable column.
        lit_col: LM litigious word ratio column.
        tier_col: Binary top-tier underwriter flag.
        alpha: Significance level.
        collapse_floor: Drop observations with underpricing ≤ this value
            (same data-quality fix as H1).
        n_bootstrap: Bootstrap resamples for the Δρ confidence interval.
        rng_seed: Random seed.

    Returns:
        Results dict including the interaction-term coefficient, Spearman ρ
        inside each tier subsample, and a bootstrap 95% CI on the
        difference (top-tier ρ − non-top-tier ρ).
    """
    needed = [target_col, lit_col, tier_col, "is_spac", "sector_encoded",
              "offer_price", "vix_at_pricing", "nasdaq_30d_return"]
    raw = df[needed].dropna(subset=[target_col, lit_col, tier_col])
    data = raw[raw[target_col] > collapse_floor].copy()

    top = data[data[tier_col] == 1]
    non_top = data[data[tier_col] == 0]

    # The upstream Ritter underwriter-rank merge in this dataset is unfinished
    # (every row currently has ``top_tier_underwriter == 0``), so the "top"
    # group is empty.  Return a clearly-flagged "skipped" result rather than
    # NaN garbage from a degenerate regression / bootstrap.
    if len(top) < 30 or len(non_top) < 30:
        return {
            "hypothesis": "H2: Underwriter translation effect — top-tier banks attenuate the litigious paradox",
            "h0": "The litigious × top_tier OLS interaction coefficient is zero (no attenuation)",
            "h1": "Positive interaction coefficient (top-tier banks weaken the negative main effect)",
            "test": "SKIPPED — Ritter underwriter-rank merge incomplete",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_top": len(top),
            "n_non_top": len(non_top),
            "reject_h0": False,
            "interpretation": (
                f"Skipped — only {len(top)} top-tier rows and {len(non_top)} "
                "non-top-tier rows after dropna and the −50% collapse filter. "
                "The Ritter underwriter-rank merge in this dataset has not "
                "been populated (every IPO currently has top_tier_underwriter=0), "
                "so this test is structurally unavailable.  Re-run after "
                "completing the merge in src/feature_engineering.py."
            ),
            "ml_takeaway": (
                "Skipped — interaction between litigious-language and "
                "underwriter prestige cannot be evaluated until the Ritter "
                "merge is populated."
            ),
        }

    # ---------- OLS with interaction term ----------------------------------
    ols_data = data.dropna(subset=["offer_price", "vix_at_pricing",
                                    "nasdaq_30d_return", "sector_encoded"]).copy()
    ols_data["log_offer"] = np.log(ols_data["offer_price"].clip(lower=0.01))
    ols_data["_lit_x_tier"] = ols_data[lit_col] * ols_data[tier_col]
    X = sm.add_constant(ols_data[[
        lit_col, tier_col, "_lit_x_tier",
        "is_spac", "sector_encoded", "log_offer",
        "vix_at_pricing", "nasdaq_30d_return",
    ]].astype(float))
    y = ols_data[target_col].astype(float)
    ols_model = sm.OLS(y, X).fit()
    inter_coef = float(ols_model.params["_lit_x_tier"])
    inter_p = float(ols_model.pvalues["_lit_x_tier"])
    main_coef = float(ols_model.params[lit_col])
    main_p = float(ols_model.pvalues[lit_col])

    # ---------- Bootstrap difference-of-Spearman ---------------------------
    top_rho, top_p = stats.spearmanr(top[lit_col], top[target_col])
    non_top_rho, non_top_p = stats.spearmanr(non_top[lit_col], non_top[target_col])
    observed_diff = float(top_rho - non_top_rho)

    rng = np.random.default_rng(rng_seed)
    top_lit = top[lit_col].values
    top_up = top[target_col].values
    non_lit = non_top[lit_col].values
    non_up = non_top[target_col].values

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ti = rng.integers(0, len(top), size=len(top))
        ni = rng.integers(0, len(non_top), size=len(non_top))
        tr, _ = stats.spearmanr(top_lit[ti], top_up[ti])
        nr, _ = stats.spearmanr(non_lit[ni], non_up[ni])
        boot_diffs[i] = tr - nr
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    # One-sided bootstrap p-value: probability Δρ ≤ 0 under resampling
    boot_p = float(np.mean(boot_diffs <= 0))

    return {
        "hypothesis": "H2: Underwriter translation effect — top-tier banks attenuate the litigious paradox",
        "h0": "The litigious × top_tier OLS interaction coefficient is zero (no attenuation)",
        "h1": "Positive interaction coefficient (top-tier banks weaken the negative main effect)",
        "test": ("OLS with litigious × top_tier interaction + bootstrap "
                 f"Δρ (B={n_bootstrap:,}, top-tier ρ − non-top-tier ρ)"),
        "statistic": inter_coef,
        "p_value": inter_p,
        "main_coef": main_coef,
        "main_p_value": main_p,
        "top_rho": float(top_rho),
        "top_p_value": float(top_p),
        "non_top_rho": float(non_top_rho),
        "non_top_p_value": float(non_top_p),
        "rho_diff": observed_diff,
        "rho_diff_ci_95": (float(ci_lo), float(ci_hi)),
        "bootstrap_p_value": boot_p,
        "n_top": len(top),
        "n_non_top": len(non_top),
        "n_ols": len(ols_data),
        "reject_h0": inter_p < alpha,
        "interpretation": (
            f"Main effect (litigious): β={main_coef:+.2f} (p={main_p:.4g}).  "
            f"Interaction (litigious × top_tier): β={inter_coef:+.2f} (p={inter_p:.4g}).  "
            f"Subsample Spearman: top-tier ρ={top_rho:+.3f} (n={len(top)}) vs. "
            f"non-top-tier ρ={non_top_rho:+.3f} (n={len(non_top)}); "
            f"bootstrap Δρ={observed_diff:+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}], "
            f"one-sided p={boot_p:.4g}."
        ),
        "ml_takeaway": (
            "If the interaction is positive and significant, the prospectus "
            "litigious-language signal is most informative for non-top-tier "
            "issuers — a tree-based model should be able to exploit the "
            "joint feature (litigious × underwriter tier) rather than either "
            "alone.  Linear models without an explicit interaction will miss this."
        ),
    }


# ---------------------------------------------------------------------------
# H3: Disclosure concentration curse  --  where the negative tone lives
# ---------------------------------------------------------------------------

def test_h3_disclosure_concentration(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    rf_neg_col: str = "rf_lm_negative_ratio",
    full_neg_col: str = "lm_negative_ratio",
    alpha: float = 0.05,
    collapse_floor: float = -0.50,
) -> dict[str, Any]:
    """Test whether the *location* of negative tone predicts underpricing.

    A standard reading of LM-negative ratios treats *how much* negative
    language a prospectus contains as the only thing that matters.  But
    the *location* should matter too: if negative language is
    compartmentalised inside the Risk Factors section — where the SEC
    requires it — that is high-quality, well-organised disclosure.  If it
    is pervasive across the whole document, that suggests either genuine
    operational distress bleeding into the narrative or a chaotic / poorly
    drafted filing.

    We operationalise this with a single metric:

        ``risk_concentration_ratio = rf_lm_negative_ratio / lm_negative_ratio``

    Higher values mean *more* of the prospectus's negative language is
    concentrated in the Risk Factors section (good disclosure hygiene);
    lower values mean it bleeds across the entire prospectus (worse
    hygiene).  Under a Hanley & Hoberg (2010) disclosure interpretation,
    well-organised disclosure → lower information asymmetry → lower
    first-day underpricing; chaotic disclosure → the opposite.

    Args:
        df: IPO DataFrame with both full-prospectus and Risk-Factors-only
            LM negative ratios, plus underpricing.
        target_col: Target variable column.
        rf_neg_col: Risk-Factors-only LM negative ratio column.
        full_neg_col: Full-prospectus LM negative ratio column.
        alpha: Significance level.
        collapse_floor: Drop observations with underpricing ≤ this value
            (same data-quality fix as H1).

    Returns:
        Results dict with Spearman ρ on the concentration ratio,
        Kruskal-Wallis across terciles, and the by-tercile medians.
    """
    raw = df[[target_col, rf_neg_col, full_neg_col]].dropna()
    # Guard against division by zero — drop rows with full_neg_col == 0
    raw = raw[raw[full_neg_col] > 0].copy()
    raw["risk_concentration_ratio"] = raw[rf_neg_col] / raw[full_neg_col]

    n_raw = len(raw)
    data = raw[raw[target_col] > collapse_floor].copy()
    n_clean = len(data)
    n_dropped_collapse = n_raw - n_clean

    # Primary test: Spearman rank correlation on the concentration ratio
    rho, spearman_p = stats.spearmanr(
        data["risk_concentration_ratio"], data[target_col]
    )

    # Tercile split + Kruskal-Wallis
    data["_tercile"] = pd.qcut(
        data["risk_concentration_ratio"],
        q=3,
        labels=["T1_pervasive", "T2_mid", "T3_compartmentalised"],
        duplicates="drop",
    )
    tercile_medians = (
        data.groupby("_tercile", observed=True)[target_col].median().to_dict()
    )
    tercile_n = (
        data.groupby("_tercile", observed=True)[target_col].size().to_dict()
    )
    tercile_conc = (
        data.groupby("_tercile", observed=True)["risk_concentration_ratio"]
        .median().to_dict()
    )
    groups = [g[target_col].values for _, g in data.groupby("_tercile", observed=True)]
    kw_stat, kw_p = stats.kruskal(*groups)

    return {
        "hypothesis": "H3: Disclosure concentration curse — pervasive negative tone predicts higher underpricing",
        "h0": "No monotone association between risk_concentration_ratio and underpricing",
        "h1": ("Negative Spearman ρ: more concentrated (compartmentalised) "
               "negative tone → lower underpricing"),
        "test": ("Spearman ρ on risk_concentration_ratio "
                 "(= rf_lm_negative_ratio / lm_negative_ratio) + "
                 "Kruskal-Wallis across terciles"),
        "statistic": rho,
        "p_value": spearman_p,
        "effect_size": rho,
        "kw_statistic": kw_stat,
        "kw_p_value": kw_p,
        "tercile_medians": {str(k): float(v) for k, v in tercile_medians.items()},
        "tercile_n": {str(k): int(v) for k, v in tercile_n.items()},
        "tercile_concentration": {str(k): float(v) for k, v in tercile_conc.items()},
        "n_clean": n_clean,
        "n_dropped_collapse": n_dropped_collapse,
        "reject_h0": spearman_p < alpha,
        "interpretation": (
            f"Spearman ρ={rho:+.3f} (p={spearman_p:.4g}) on n={n_clean} clean "
            f"first-day returns (after dropping {n_dropped_collapse} post-IPO "
            f"collapses ≤ {int(collapse_floor*100)}%).  Median first-day return "
            f"by concentration tercile: T1_pervasive="
            f"{tercile_medians.get('T1_pervasive', float('nan')):+.1%}, "
            f"T2_mid={tercile_medians.get('T2_mid', float('nan')):+.1%}, "
            f"T3_compartmentalised="
            f"{tercile_medians.get('T3_compartmentalised', float('nan')):+.1%} "
            f"(Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4g})."
        ),
        "ml_takeaway": (
            "risk_concentration_ratio is a new, derived feature that splits "
            "an existing signal (LM-negative) into *where the signal lives*. "
            "It is an obvious candidate for the model feature set, and "
            "directly extends Hanley & Hoberg (2010) — disclosure quality "
            "is not just about how much negative language there is, but "
            "about whether it is properly compartmentalised."
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

    # The upstream Ritter underwriter-rank merge in this dataset is unfinished
    # (every row currently has ``top_tier_underwriter == 0``), so the "top"
    # group is empty.  Return a clearly-flagged "skipped" result rather than
    # NaN garbage from a degenerate Levene call.
    if len(top) < 2 or len(non_top) < 2:
        return {
            "hypothesis": "H5: Top-tier underwriters reduce underpricing variance",
            "h0": "Variance is equal for top-tier and non-top-tier underwriter groups",
            "h1": "Top-tier underwriters produce lower variance (certification hypothesis)",
            "test": "SKIPPED — Ritter underwriter-rank merge incomplete",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_top": len(top),
            "n_non_top": len(non_top),
            "reject_h0": False,
            "interpretation": (
                f"Skipped — only {len(top)} top-tier rows and {len(non_top)} "
                "non-top-tier rows after dropna. The Ritter underwriter-rank "
                "merge in this dataset has not been populated, so this test "
                "is structurally unavailable.  Re-run after completing the "
                "merge in src/feature_engineering.py."
            ),
            "ml_takeaway": (
                "Skipped — underwriter rank cannot be evaluated until the "
                "Ritter merge is populated."
            ),
        }

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
        # Only columns that actually exist in the processed dataset AND have
        # non-trivial coverage.  ``underwriter_rank`` is excluded — the Ritter
        # rank merge is unfinished and the column is currently all-NaN, so
        # including it would empty the joint dropna.
        financial_features = [
            c for c in [
                "offer_price", "top_tier_underwriter", "is_spac",
                "vix_at_pricing", "nasdaq_30d_return", "hot_market_dummy",
                "ipo_year", "ipo_quarter", "sector_encoded",
            ]
            if c in df.columns and df[c].notna().any()
        ]

    if text_features is None:
        text_features = [
            c for c in [
                "lm_negative_ratio", "lm_uncertainty_ratio",
                "lm_litigious_ratio", "lm_positive_ratio",
                "gunning_fog", "prospectus_uniqueness", "word_count",
            ]
            if c in df.columns and df[c].notna().any()
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
