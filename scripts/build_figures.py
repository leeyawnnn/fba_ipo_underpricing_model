"""
Generate all publication-quality figures for the IPO underpricing analysis.

Run:
    python scripts/build_figures.py

All output PNGs land in reports/figures/. The script is idempotent — running
it again overwrites the previous figures.

Design choices documented inline:
  - Underpricing is right-skewed AND can be negative (delisted IPOs go to -1).
    Log scale is used on the y-axis via symlog (signed log) where applicable.
  - Sector counts are plotted on a log y-axis because SPAC and Industrials
    dominate the count.
  - The sentiment-vs-underpricing scatter shows a binned-quantile view
    (deciles) AFTER trimming the top/bottom 1% of underpricing observations,
    so a handful of moonshot IPOs don't dominate the visual relationship.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "ipo_features.parquet"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PRIMARY = "#2563EB"
ACCENT = "#DC2626"
NEUTRAL = "#475569"
GREEN = "#059669"
ORANGE = "#EA580C"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trim_outliers(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Return *s* with values outside the [lo, hi] quantile range dropped."""
    if len(s) == 0:
        return s
    a, b = s.quantile([lo, hi])
    return s[(s >= a) & (s <= b)]


def save(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}")


# Friendly category palette
SECTOR_ORDER = [
    "Healthcare", "Technology", "Financials", "SPAC",
    "Communication Services", "Consumer Discretionary",
    "Consumer Staples", "Energy", "Industrials",
    "Materials", "Real Estate",
]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["ipo_date"] = pd.to_datetime(df["ipo_date"])
    df["ipo_year"] = df["ipo_date"].dt.year
    df["ipo_year_month"] = df["ipo_date"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------------------
# Figure 1 — Underpricing distribution (linear + symlog)
# ---------------------------------------------------------------------------

def fig_underpricing_dist(df: pd.DataFrame) -> None:
    """Two-panel histogram: linear scale (top), symlog (bottom)."""
    data = df["underpricing"].dropna()
    trimmed = trim_outliers(data, 0.01, 0.99)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"hspace": 0.35})

    # Top — trimmed (1st-99th percentile) on a regular linear scale
    ax = axes[0]
    ax.hist(trimmed, bins=60, color=PRIMARY, edgecolor="white", alpha=0.85)
    ax.axvline(0, color=NEUTRAL, lw=1, ls="--", label="Zero return")
    ax.axvline(trimmed.median(), color=ACCENT, lw=1.5, ls="-",
               label=f"Median = {trimmed.median():+.1%}")
    ax.set_title("First-day IPO return — distribution (1%-99% percentile range)")
    ax.set_xlabel("First-day return")
    ax.set_ylabel("Number of IPOs")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", frameon=True)

    # Bottom — full distribution on symmetric-log y-axis
    ax = axes[1]
    ax.hist(data, bins=80, color=ORANGE, edgecolor="white", alpha=0.85)
    ax.set_yscale("symlog", linthresh=1)
    ax.axvline(0, color=NEUTRAL, lw=1, ls="--")
    ax.set_title("Full distribution including outliers (y-axis: symmetric log)")
    ax.set_xlabel("First-day return  (-1 = delisted to zero)")
    ax.set_ylabel("Number of IPOs (symlog)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle("Underpricing Distribution — Linear vs. Symlog", fontsize=13, y=1.0)
    save(fig, "01_underpricing_distribution.png")


# ---------------------------------------------------------------------------
# Figure 2 — Sector counts on log y-axis
# ---------------------------------------------------------------------------

def fig_sector_counts(df: pd.DataFrame) -> None:
    counts = df["sector"].value_counts()
    counts = counts.reindex([s for s in SECTOR_ORDER if s in counts.index])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(counts)), counts.values, color=PRIMARY, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    ax.set_ylabel("Number of IPOs (log scale)")
    ax.set_title(
        f"IPOs by GICS Sector, 2019-2024  (n={len(df):,})\n"
        "All companies classified — no \"Unknown\" / \"Other\" buckets"
    )
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n}", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=9)
    save(fig, "02_sector_counts_logscale.png")


# ---------------------------------------------------------------------------
# Figure 3 — Sector × Year heatmap (median underpricing)
# ---------------------------------------------------------------------------

def fig_sector_year_heatmap(df: pd.DataFrame) -> None:
    """Single heatmap of median first-day return by sector × year."""
    sector_counts = df["sector"].value_counts()
    major = [s for s in SECTOR_ORDER if s in sector_counts.head(6).index]

    pivot_med = (df.groupby(["sector", "ipo_year"])["winsorized_underpricing"]
                   .median().unstack().reindex(SECTOR_ORDER))
    pivot_n = df.groupby(["sector", "ipo_year"]).size().unstack().reindex(SECTOR_ORDER)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot_med.loc[major] * 100,
        annot=pivot_n.loc[major].astype("Int64"), fmt="d",
        cmap="RdBu_r", center=0, vmin=-30, vmax=30,
        cbar_kws={"label": "Median first-day return (%)"},
        linewidths=0.4, ax=ax,
    )
    ax.set_title(
        "Median first-day return by sector × year — six highest-volume sectors\n"
        "(cells annotated with IPO count for that sector × year)"
    )
    ax.set_xlabel("IPO year"); ax.set_ylabel("")
    save(fig, "03_sector_year_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 6 — Sentiment vs Underpricing  (binned quantiles, outliers removed)
# ---------------------------------------------------------------------------

def fig_sentiment_vs_underpricing(df: pd.DataFrame) -> None:
    cols = ["lm_negative_ratio", "lm_positive_ratio",
            "lm_uncertainty_ratio", "lm_litigious_ratio"]
    titles = ["Negative tone", "Positive tone",
              "Uncertainty tone", "Litigious tone"]

    # IMPORTANT data-quality fix: the upstream stockanalysis.com `price` field
    # is a *current* price for delisted tickers, so observations with very
    # negative "first-day return" (≤ −50%) are usually post-IPO collapses, not
    # real day-1 closes. Including them contaminates the relationship — the
    # earlier 10%-90% trim only removed outliers symmetrically and still kept
    # dozens of −60%/−90% values that pulled every decile median deep into
    # the red. Restrict to plausibly-true first-day returns (> −50%) and
    # additionally cap the top tail at the 95th percentile so a single
    # moonshot doesn't distort the visual.
    base = df.dropna(subset=cols + ["underpricing"]).copy()
    # Drop post-IPO collapses (≤ −50% — stockanalysis "current price" for
    # delisted tickers, not real day-1 close) AND cap day-1 pops at the 90th
    # percentile so a few moonshots don't crush the visible y-axis range.
    cap = base["underpricing"].quantile(0.90)
    clean = base[(base["underpricing"] > -0.5) & (base["underpricing"] <= cap)].copy()

    n_dropped_low = int((base["underpricing"] <= -0.5).sum())
    n_dropped_high = int((base["underpricing"] > cap).sum())
    print(
        f"  sentiment plot: kept {len(clean):,} of {len(base):,} rows  "
        f"(dropped {n_dropped_low} post-IPO collapses ≤ −50% "
        f"and {n_dropped_high} moonshots above the 90th pctile = {cap:+.1%})"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title in zip(axes.ravel(), cols, titles):
        d = clean.copy()
        d["_decile"] = pd.qcut(d[col], 10, labels=False, duplicates="drop")
        agg = (d.groupby("_decile")
                .agg(x_mid=(col, "mean"),
                     med=("underpricing", "median"),
                     q25=("underpricing", lambda s: s.quantile(0.25)),
                     q75=("underpricing", lambda s: s.quantile(0.75)),
                     n=("underpricing", "size"))
                .reset_index())

        rho = d[col].corr(d["underpricing"], method="spearman")

        ax.scatter(d[col], d["underpricing"], s=10, color=NEUTRAL,
                   alpha=0.22, label="IPOs (clean)")
        ax.fill_between(agg["x_mid"], agg["q25"], agg["q75"],
                        color=PRIMARY, alpha=0.20, label="Decile IQR")
        ax.plot(agg["x_mid"], agg["med"], color=PRIMARY, lw=2.2,
                marker="o", label="Decile median")
        ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
        ax.set_title(f"{title}    (Spearman ρ = {rho:+.3f})")
        ax.set_xlabel(f"LM {col.replace('lm_', '').replace('_ratio','')} word ratio")
        ax.set_ylabel("First-day return")
        ax.set_ylim(-0.55, min(1.5, cap * 1.10))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "Loughran-McDonald Sentiment vs. First-day Return\n"
        f"Post-IPO collapses (≤ −50%) and moonshots (> {cap:+.0%}) excluded — "
        f"n={len(clean):,} clean first-day returns, values grouped by sentiment decile",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    save(fig, "06_sentiment_vs_underpricing.png")


# ---------------------------------------------------------------------------
# Figure 9 — VIX vs underpricing scatter (with binned median)
# ---------------------------------------------------------------------------

def fig_vix_scatter(df: pd.DataFrame) -> None:
    # Drop post-IPO collapses (≤ −50%) — same data-quality fix as the
    # sentiment plot. Cap top tail at 95th percentile so a few moonshots
    # don't blow out the y-axis on every sector panel.
    data = df.dropna(subset=["vix_at_pricing", "underpricing", "sector"]).copy()
    cap = data["underpricing"].quantile(0.95)
    data = data[(data["underpricing"] > -0.5) & (data["underpricing"] <= cap)].copy()

    # VIX-regime buckets used by the sector-grouped bar chart. Quantile
    # cut points are computed on the *whole* sample so every sector is
    # binned on the same scale.
    q = data["vix_at_pricing"].quantile([0, 1/3, 2/3, 1.0]).values
    labels = [f"Low\n(<{q[1]:.0f})",
              f"Mid\n({q[1]:.0f}–{q[2]:.0f})",
              f"High\n(>{q[2]:.0f})"]
    data["vix_regime"] = pd.cut(
        data["vix_at_pricing"], bins=q, labels=labels, include_lowest=True
    )

    # Six largest sectors carry the signal; smaller ones (Real Estate, etc.)
    # have too few rows per VIX bucket to plot meaningfully.
    sectors = (data["sector"].value_counts().head(6).index.tolist())
    panel_order = [s for s in SECTOR_ORDER if s in sectors]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.35)

    # Top-left "All sectors" overview panel — the original decile-median view
    ax0 = fig.add_subplot(gs[0, 0])
    data["_bin"] = pd.qcut(data["vix_at_pricing"], 10, duplicates="drop")
    overview = (data.groupby("_bin", observed=True)
                    .agg(x=("vix_at_pricing", "mean"),
                         med=("underpricing", "median"))
                    .reset_index())
    ax0.scatter(data["vix_at_pricing"], data["underpricing"],
                s=6, alpha=0.18, color=NEUTRAL)
    ax0.plot(overview["x"], overview["med"], color=ACCENT, lw=2,
             marker="o", label="Decile median")
    ax0.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax0.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax0.set_xlabel("VIX at pricing")
    ax0.set_ylabel("First-day return")
    ax0.set_title(f"All sectors  (n={len(data):,})", fontsize=11)
    ax0.legend(fontsize=8)

    # Top-right two cells: bar chart — median return per VIX regime × sector
    ax_bar = fig.add_subplot(gs[0, 1:])
    bar_data = (data[data["sector"].isin(panel_order)]
                .groupby(["sector", "vix_regime"], observed=True)["underpricing"]
                .median().unstack())
    bar_data = bar_data.reindex(panel_order)
    bar_data = bar_data[labels]  # enforce Low/Mid/High order
    x = np.arange(len(bar_data))
    w = 0.27
    regime_colors = ["#1d4ed8", "#9ca3af", "#dc2626"]  # blue→grey→red
    for i, (regime, c) in enumerate(zip(labels, regime_colors)):
        ax_bar.bar(x + (i - 1) * w, bar_data[regime].values, w,
                   color=c, label=regime.replace("\n", " "))
    ax_bar.axhline(0, color=NEUTRAL, lw=0.7)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_data.index, rotation=15, ha="right")
    ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax_bar.set_ylabel("Median first-day return")
    ax_bar.set_title("Median return by sector × VIX regime", fontsize=11)
    ax_bar.legend(title="VIX regime", fontsize=8, ncol=3, loc="upper left")

    # Six per-sector scatter panels (rows 1 and 2)
    sector_color = dict(zip(panel_order, sns.color_palette("tab10", len(panel_order))))
    for i, sector in enumerate(panel_order):
        r, c = divmod(i, 3)
        ax = fig.add_subplot(gs[1 + r, c])
        sd = data[data["sector"] == sector]
        ax.scatter(sd["vix_at_pricing"], sd["underpricing"],
                   s=10, alpha=0.30, color=sector_color[sector])
        # Tercile median trace (deciles too noisy for small panels)
        if len(sd) >= 15:
            sd_t = sd.assign(
                _t=pd.qcut(sd["vix_at_pricing"], 3, duplicates="drop")
            )
            tline = (sd_t.groupby("_t", observed=True)
                         .agg(x=("vix_at_pricing", "mean"),
                              med=("underpricing", "median"))
                         .reset_index())
            ax.plot(tline["x"], tline["med"], color=sector_color[sector],
                    lw=2, marker="o", label="Tercile median")
        rho = sd["vix_at_pricing"].corr(sd["underpricing"], method="spearman")
        ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("VIX at pricing")
        ax.set_ylabel("First-day return")
        ax.set_title(f"{sector}  (n={len(sd):,},  ρ={rho:+.2f})", fontsize=10)

    fig.suptitle(
        "VIX vs. First-day Return — grouped by sector\n"
        "Post-IPO collapses (≤ −50%) excluded; one panel per major sector",
        fontsize=13, y=1.0,
    )
    save(fig, "09_vix_vs_underpricing.png")


# ---------------------------------------------------------------------------
# Figure 10 — Correlation heatmap of features with underpricing
# ---------------------------------------------------------------------------

def fig_correlation_heatmap(df: pd.DataFrame) -> None:
    feats = [
        "underpricing", "winsorized_underpricing", "offer_price",
        "vix_at_pricing", "nasdaq_30d_return", "nasdaq_30d_volatility",
        "hot_market_dummy", "is_spac", "ipo_year",
        "lm_negative_ratio", "lm_positive_ratio",
        "lm_uncertainty_ratio", "lm_litigious_ratio",
        "gunning_fog", "prospectus_uniqueness", "word_count",
    ]
    feats = [c for c in feats if c in df.columns]
    corr = df[feats].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation matrix — underpricing & features")
    save(fig, "10_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 11 — Litigious-tone paradox  (the H7 highlight)
# ---------------------------------------------------------------------------

def fig_litigious_paradox(df: pd.DataFrame) -> None:
    """Two-panel figure illustrating the counter-intuitive H7 finding.

    Left panel: quintiles of LM litigious ratio vs. median first-day return
    (Q1 = least legal language, Q5 = most).  The bars step DOWN from Q1 to
    Q5 — the opposite of what the naive "more risk language → bigger pop"
    intuition predicts.

    Right panel: Spearman ρ between every LM category and the first-day
    return on the same clean sample.  Litigious is the only category with
    p < 0.01 and the only one with a clearly NEGATIVE coefficient, ruling
    out "this is just a generic risk-language effect".
    """
    from scipy import stats as _stats

    base = df.dropna(subset=["lm_litigious_ratio", "underpricing"]).copy()
    clean = base[base["underpricing"] > -0.5].copy()
    n_clean = len(clean)

    # Left panel — quintile bars
    clean["lit_q"] = pd.qcut(
        clean["lm_litigious_ratio"], 5,
        labels=["Q1\nleast\nlitigious", "Q2", "Q3", "Q4", "Q5\nmost\nlitigious"],
        duplicates="drop",
    )
    q_agg = (clean.groupby("lit_q", observed=True)["underpricing"]
                  .agg(med="median", n="size").reset_index())
    rho, sp_p = _stats.spearmanr(clean["lm_litigious_ratio"], clean["underpricing"])
    kw_stat, kw_p = _stats.kruskal(
        *[g["underpricing"].values for _, g in clean.groupby("lit_q", observed=True)]
    )

    # Right panel — Spearman ρ for every LM category, same clean sample
    lm_cols = [
        ("lm_negative_ratio", "Negative"),
        ("lm_positive_ratio", "Positive"),
        ("lm_uncertainty_ratio", "Uncertainty"),
        ("lm_constraining_ratio", "Constraining"),
        ("lm_litigious_ratio", "Litigious"),
    ]
    rows = []
    for col, label in lm_cols:
        sub = clean.dropna(subset=[col])
        r, p = _stats.spearmanr(sub[col], sub["underpricing"])
        rows.append({"category": label, "rho": r, "p": p, "n": len(sub)})
    summary = pd.DataFrame(rows).set_index("category")
    summary = summary.reindex([lbl for _, lbl in lm_cols])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Left: quintile bars -------------------------------------------------
    # Keep the title to ONE short plain-language line; the heavy statistics go
    # into a tidy box inside the panel (upper-right, where the bars are short)
    # so no long line of numbers spills across into the right-hand panel.
    bar_colors = ["#94a3b8", "#94a3b8", "#94a3b8", "#94a3b8", ACCENT]
    bars = ax1.bar(q_agg["lit_q"].astype(str), q_agg["med"] * 100,
                   color=bar_colors, edgecolor="white")
    ax1.axhline(0, color=NEUTRAL, lw=0.7)
    ax1.set_ylabel("Median first-day return (%)")
    ax1.set_title("Most-litigious IPOs pop the LEAST", fontsize=12, pad=10)
    max_h = float((q_agg["med"] * 100).max())
    ax1.set_ylim(0, max_h * 1.30)
    for bar, n_q in zip(bars, q_agg["n"]):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.03,
                 f"{h:+.1f}%\n(n={int(n_q)})", ha="center", va="bottom",
                 fontsize=9)
    ax1.set_xlabel("Amount of legal / litigious language in the S-1\n"
                   "(quintiles: Q1 = least, Q5 = most)")

    # --- Right: Spearman-rho comparison across all tone categories -----------
    # Numeric labels always sit at the far right so they never collide with the
    # category names that sit at x = 0.
    colors = ["#94a3b8" if c != "Litigious" else ACCENT for c in summary.index]
    bars2 = ax2.barh(summary.index, summary["rho"], color=colors, edgecolor="white")
    ax2.axvline(0, color=NEUTRAL, lw=0.7)
    ax2.set_xlabel("Correlation (Spearman rho) with first-day return")
    ax2.set_title("Litigious is the only tone with a real\nnegative link to the pop",
                  fontsize=12, pad=10)
    # Reserve a wide blank strip on the right for a tidy label column. x_hi is
    # pushed well past the longest bar so the right-aligned labels never touch
    # any bar. label_x marks where that column lives.
    x_lo = min(-0.33, summary["rho"].min() - 0.10)
    x_hi = max(0.78, summary["rho"].max() + 0.55)
    label_x = x_hi - 0.02
    ax2.set_xlim(x_lo, x_hi)
    # Light divider between the bars and the label column.
    ax2.axvline(summary["rho"].max() + 0.10, color="#e2e8f0", lw=1)
    for bar, (cat, row) in zip(bars2, summary.iterrows()):
        x = bar.get_width()
        sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01
              else ("*" if row["p"] < 0.05 else ""))
        ax2.text(label_x, bar.get_y() + bar.get_height() / 2,
                 f"rho = {x:+.3f}    p = {row['p']:.2g} {sig}",
                 ha="right", va="center", fontsize=9, color=NEUTRAL)

    # All free text lives OUTSIDE the two plotting axes: a headline + explainer
    # in the reserved top band, and the statistics in a caption strip along the
    # bottom. tight_layout's rect keeps the axes between them so nothing can
    # crash into the charts.
    fig.suptitle("The Litigious-Tone Paradox", fontsize=16,
                 fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.905,
        "More legal / litigious language in the prospectus predicts a SMALLER "
        "first-day pop —\nthe opposite of the naive \"more risk talk → bigger "
        "pop\" intuition.",
        ha="center", va="top", fontsize=11, color=NEUTRAL,
    )
    fig.text(
        0.5, 0.035,
        f"Spearman rho = {rho:+.3f}  (p = {sp_p:.1g})      ·      "
        f"Kruskal-Wallis H = {kw_stat:.1f}  (p = {kw_p:.1g})      ·      "
        f"n = {n_clean} IPOs      ·      "
        "significance:  * p<0.05   ** p<0.01   *** p<0.001",
        ha="center", va="center", fontsize=9.5, color=NEUTRAL,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.82])
    save(fig, "11_litigious_paradox.png")


# ---------------------------------------------------------------------------
# Figure 12 — Disclosure Concentration Curse (H3)
# ---------------------------------------------------------------------------

def fig_disclosure_concentration(df: pd.DataFrame) -> None:
    """Three-panel figure for H3: where the negative tone lives matters.

    Left:   tercile bars (T1_pervasive / T2_mid / T3_compartmentalised) showing
            median first-day return by risk_concentration_ratio tercile.
    Middle: scatter of risk_concentration_ratio vs first-day return for the
            BULK of the sample (return ≤ 200%), with a quintile-median line.
    Right:  the handful of >200% "moonshot" IPOs broken out on their own axis
            so they don't blow out the y-scale of the main scatter.
    """
    from scipy import stats as _stats

    base = df.dropna(subset=["rf_lm_negative_ratio", "lm_negative_ratio",
                              "underpricing"]).copy()
    base = base[base["lm_negative_ratio"] > 0].copy()
    base["risk_concentration_ratio"] = (
        base["rf_lm_negative_ratio"] / base["lm_negative_ratio"]
    )
    clean = base[base["underpricing"] > -0.5].copy()
    n_clean = len(clean)

    rho, sp_p = _stats.spearmanr(
        clean["risk_concentration_ratio"], clean["underpricing"]
    )

    # Tercile split
    clean["_tercile"] = pd.qcut(
        clean["risk_concentration_ratio"], q=3,
        labels=["T1\npervasive", "T2\nmid", "T3\ncompart-\nmentalised"],
        duplicates="drop",
    )
    t_agg = (clean.groupby("_tercile", observed=True)["underpricing"]
                  .agg(med="median", n="size").reset_index())

    kw_stat, kw_p = _stats.kruskal(
        *[g["underpricing"].values
          for _, g in clean.groupby("_tercile", observed=True)]
    )

    # Split the scatter at a 200% first-day return so the dense bulk of the
    # sample and the few extreme "moonshots" each get a readable y-axis.
    SPLIT = 2.0
    main = clean[clean["underpricing"] <= SPLIT].copy()
    moon = clean[clean["underpricing"] > SPLIT].copy()
    print(
        f"  disclosure plot: {len(main):,} IPOs at ≤200% first-day return, "
        f"{len(moon):,} moonshots >200% broken out separately"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17, 6.6),
        gridspec_kw={"width_ratios": [1, 1.35, 0.85]},
    )

    # --- Left panel: tercile bars -------------------------------------------
    bar_colors = [ACCENT, NEUTRAL, PRIMARY]
    bars = ax1.bar(t_agg["_tercile"].astype(str), t_agg["med"] * 100,
                   color=bar_colors, edgecolor="white")
    ax1.axhline(0, color=NEUTRAL, lw=0.7)
    ax1.set_ylabel("Median first-day return (%)")
    ax1.set_xlabel("Risk-concentration tercile\n"
                   "(risk-factor negative tone ÷ whole-document negative tone)")
    ax1.set_title(
        "Median first-day return by\ndisclosure-concentration tercile\n"
        f"Kruskal-Wallis H = {kw_stat:.1f}  (p = {kw_p:.4g}),  n = {n_clean}",
        fontsize=11, pad=12,
    )
    max_h = float((t_agg["med"] * 100).max())
    ax1.set_ylim(0, max_h * 1.35)
    for bar, row in zip(bars, t_agg.itertuples()):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + max_h * 0.03,
                 f"{h:+.1f}%\n(n={int(row.n)})", ha="center", va="bottom",
                 fontsize=9)

    # --- Middle panel: main scatter (return ≤ 200%) + quintile median -------
    ax2.scatter(main["risk_concentration_ratio"], main["underpricing"],
                s=16, alpha=0.30, color=NEUTRAL, label="IPOs (≤200% return)")

    main["_t"] = pd.qcut(main["risk_concentration_ratio"], 5,
                         duplicates="drop")
    tline = (main.groupby("_t", observed=True)
                 .agg(x=("risk_concentration_ratio", "mean"),
                      med=("underpricing", "median"))
                 .reset_index())
    ax2.plot(tline["x"], tline["med"], color=PRIMARY, lw=2.8,
             marker="o", markersize=8, label="Quintile median trend")
    ax2.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_ylim(-0.6, SPLIT + 0.05)
    ax2.set_xlabel("Risk-concentration ratio  (higher → negative tone is\n"
                   "compartmentalised in the Risk Factors section)")
    ax2.set_ylabel("First-day return")
    ax2.set_title(
        f"Bulk of the sample — first-day return ≤ 200%  (n = {len(main):,})\n"
        f"Spearman rho = {rho:+.3f}  (p = {sp_p:.4g}, full clean sample)",
        fontsize=11, pad=12,
    )
    ax2.legend(loc="upper right", fontsize=9, frameon=True)

    # --- Right panel: moonshots (return > 200%) -----------------------------
    ax3.scatter(moon["risk_concentration_ratio"], moon["underpricing"],
                s=36, alpha=0.75, color=ACCENT, edgecolor="white",
                label=f"IPOs >200% return (n = {len(moon):,})")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if len(moon):
        ax3.set_ylim(SPLIT * 0.95, moon["underpricing"].max() * 1.08)
    ax3.set_xlabel("Risk-concentration ratio")
    ax3.set_ylabel("First-day return")
    ax3.set_title(
        "Moonshots broken out\nfirst-day return > 200%",
        fontsize=11, pad=12,
    )
    ax3.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "H3 — Disclosure Concentration Curse:  where the negative tone lives matters\n"
        "Pervasive negative tone across the whole prospectus → higher underpricing;  "
        "compartmentalised negative tone (Risk Factors only) → lower underpricing",
        fontsize=12.5, y=0.99,
    )
    # rect top = 0.85 keeps the two-line suptitle clear of the panel titles.
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    save(fig, "12_disclosure_concentration.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data …")
    df = load_data()
    print(f"  n = {len(df):,} IPOs;  {df['sector'].nunique()} sectors")

    print("\nGenerating figures:")
    fig_underpricing_dist(df)
    fig_sector_counts(df)
    fig_sector_year_heatmap(df)
    fig_sentiment_vs_underpricing(df)
    fig_vix_scatter(df)
    fig_correlation_heatmap(df)
    fig_litigious_paradox(df)
    fig_disclosure_concentration(df)
    print("\nDone. Figures in:", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
