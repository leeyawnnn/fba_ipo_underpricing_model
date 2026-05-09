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
    pivot_med = (df.groupby(["sector", "ipo_year"])["winsorized_underpricing"]
                   .median().unstack().reindex(SECTOR_ORDER))
    pivot_n = df.groupby(["sector", "ipo_year"]).size().unstack().reindex(SECTOR_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot_med * 100, annot=pivot_n.astype("Int64"), fmt="d",
        cmap="RdBu_r", center=0, vmin=-30, vmax=30,
        cbar_kws={"label": "Median first-day return (%)"},
        linewidths=0.4, ax=ax,
    )
    ax.set_title(
        "Median first-day return by sector × year   (cells annotated with IPO count)"
    )
    ax.set_xlabel("IPO year")
    ax.set_ylabel("")
    save(fig, "03_sector_year_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4 — Sector boxplot of underpricing (winsorized; symlog x)
# ---------------------------------------------------------------------------

def fig_sector_boxplot(df: pd.DataFrame) -> None:
    data = df[["sector", "winsorized_underpricing"]].dropna()
    order = (data.groupby("sector")["winsorized_underpricing"]
                 .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=data, y="sector", x="winsorized_underpricing",
        order=order, ax=ax, fliersize=2, color=PRIMARY,
    )
    ax.axvline(0, color=NEUTRAL, ls="--", lw=1)
    ax.set_title("Underpricing by Sector  (1%-99% winsorized; sorted by median)")
    ax.set_xlabel("First-day return (winsorized)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save(fig, "04_sector_boxplot.png")


# ---------------------------------------------------------------------------
# Figure 5 — Monthly volume + median return
# ---------------------------------------------------------------------------

def fig_monthly_volume(df: pd.DataFrame) -> None:
    monthly = (df.groupby("ipo_year_month")
                 .agg(med_ret=("underpricing", "median"),
                      n=("underpricing", "size"))
                 .reset_index())

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(monthly["ipo_year_month"], monthly["n"], width=22,
            color=NEUTRAL, alpha=0.45, label="IPO volume")
    ax1.set_ylabel("Number of IPOs (bar)")
    ax1.set_xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(monthly["ipo_year_month"], monthly["med_ret"],
             color=ACCENT, lw=2, label="Median first-day return")
    ax2.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax2.set_ylabel("Median return (line)", color=ACCENT)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.spines["top"].set_visible(False)

    ax1.set_title("Monthly IPO volume and median first-day return  (2019-2024)")
    fig.tight_layout()
    save(fig, "05_monthly_volume_returns.png")


# ---------------------------------------------------------------------------
# Figure 6 — Sentiment vs Underpricing  (binned quantiles, outliers removed)
# ---------------------------------------------------------------------------

def fig_sentiment_vs_underpricing(df: pd.DataFrame) -> None:
    cols = ["lm_negative_ratio", "lm_positive_ratio",
            "lm_uncertainty_ratio", "lm_litigious_ratio"]
    titles = ["Negative tone", "Positive tone",
              "Uncertainty tone", "Litigious tone"]

    # Aggressive outlier filter for clean visuals: drop top/bottom 10% of
    # underpricing AND clip remaining values to ±100% so a single moonshot
    # IPO doesn't dominate the panel. Decile statistics are computed on the
    # clipped data only — they remain robust because medians ignore tails.
    base = df.dropna(subset=cols + ["underpricing"]).copy()
    lo, hi = base["underpricing"].quantile([0.10, 0.90])
    clean = base[(base["underpricing"] >= lo) & (base["underpricing"] <= hi)].copy()

    print(f"  sentiment plot: kept {len(clean):,} of {len(base):,} rows after 10%-90% trim")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col, title in zip(axes.ravel(), cols, titles):
        # Decile bins of the sentiment ratio
        d = clean.copy()
        d["_decile"] = pd.qcut(d[col], 10, labels=False, duplicates="drop")
        agg = (d.groupby("_decile")
                .agg(x_mid=(col, "mean"),
                     med=("underpricing", "median"),
                     q25=("underpricing", lambda s: s.quantile(0.25)),
                     q75=("underpricing", lambda s: s.quantile(0.75)),
                     n=("underpricing", "size"))
                .reset_index())

        # Spearman rank correlation (robust to non-linearity)
        rho = d[col].corr(d["underpricing"], method="spearman")

        ax.scatter(d[col], d["underpricing"], s=10, color=NEUTRAL,
                   alpha=0.20, label="IPOs (trimmed)")
        ax.fill_between(agg["x_mid"], agg["q25"], agg["q75"],
                        color=PRIMARY, alpha=0.20, label="Decile IQR")
        ax.plot(agg["x_mid"], agg["med"], color=PRIMARY, lw=2.2,
                marker="o", label="Decile median")
        ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
        ax.set_title(f"{title}    (Spearman ρ = {rho:+.3f})")
        ax.set_xlabel(f"LM {col.replace('lm_', '').replace('_ratio','')} word ratio")
        ax.set_ylabel("First-day return")
        ax.set_ylim(-1.0, 1.5)  # keep visual focus on the median trend
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "Loughran-McDonald Sentiment vs. First-day Return\n"
        f"Outliers trimmed (10%-90% percentile, n={len(clean):,}) — values grouped by sentiment decile",
        fontsize=13, y=1.0,
    )
    fig.tight_layout()
    save(fig, "06_sentiment_vs_underpricing.png")


# ---------------------------------------------------------------------------
# Figure 7 — Calendar heatmap (year × month IPO volume)
# ---------------------------------------------------------------------------

def fig_calendar_heatmap(df: pd.DataFrame) -> None:
    pivot = (df.assign(month=df["ipo_date"].dt.month)
               .groupby(["ipo_year", "month"])
               .size().unstack(fill_value=0))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [months[c - 1] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.4, cbar_kws={"label": "IPOs"}, ax=ax)
    ax.set_title("IPO volume by year × month")
    ax.set_xlabel(""); ax.set_ylabel("Year")
    save(fig, "07_calendar_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 8 — Hot vs Cold market by sector
# ---------------------------------------------------------------------------

def fig_hot_vs_cold(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["winsorized_underpricing", "hot_market_dummy"]).copy()
    top = data["sector"].value_counts().head(8).index.tolist()
    data = data[data["sector"].isin(top)]
    summary = (data.groupby(["sector", "hot_market_dummy"])["winsorized_underpricing"]
                   .median().unstack())
    summary.columns = ["Cold", "Hot"]
    summary = summary.sort_values("Hot", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    w = 0.4
    ax.barh(x - w/2, summary["Cold"], w, color=NEUTRAL, label="Cold market")
    ax.barh(x + w/2, summary["Hot"], w, color=ACCENT, label="Hot market")
    ax.axvline(0, color=NEUTRAL, lw=0.7)
    ax.set_yticks(x); ax.set_yticklabels(summary.index)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Median winsorized first-day return")
    ax.set_title("Underpricing in hot vs. cold IPO markets, by sector")
    ax.legend(loc="lower right")
    save(fig, "08_hot_vs_cold_by_sector.png")


# ---------------------------------------------------------------------------
# Figure 9 — VIX vs underpricing scatter (with binned median)
# ---------------------------------------------------------------------------

def fig_vix_scatter(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["vix_at_pricing", "underpricing"]).copy()
    lo, hi = data["underpricing"].quantile([0.05, 0.95])
    data = data[(data["underpricing"] >= lo) & (data["underpricing"] <= hi)]

    data["bin"] = pd.qcut(data["vix_at_pricing"], 10, duplicates="drop")
    binned = data.groupby("bin", observed=True).agg(
        x=("vix_at_pricing", "mean"),
        med=("underpricing", "median"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data["vix_at_pricing"], data["underpricing"],
               s=10, alpha=0.25, color=NEUTRAL)
    ax.plot(binned["x"], binned["med"], color=ACCENT, lw=2, marker="o",
            label="Decile median")
    ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("VIX at pricing date")
    ax.set_ylabel("First-day return")
    ax.set_title("VIX vs. Underpricing  (5%-95% trimmed)")
    ax.legend()
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
# Figure 11 — Top 15 / Bottom 15 IPOs (with sector colours)
# ---------------------------------------------------------------------------

def fig_top_bottom_ipos(df: pd.DataFrame) -> None:
    top = df.nlargest(15, "underpricing")[["ticker", "company_name", "sector", "underpricing"]]
    bot = df.nsmallest(15, "underpricing")[["ticker", "company_name", "sector", "underpricing"]]

    sector_colors = dict(zip(SECTOR_ORDER, sns.color_palette("tab20", len(SECTOR_ORDER))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, data, title, asc in [
        (axes[0], top, "Top 15 first-day winners", False),
        (axes[1], bot, "Bottom 15 first-day losers", True),
    ]:
        data = data.sort_values("underpricing", ascending=asc)
        labels = [f"{t}: {c[:28]}" for t, c in zip(data["ticker"], data["company_name"])]
        colors = [sector_colors.get(s, NEUTRAL) for s in data["sector"]]
        ax.barh(labels, data["underpricing"], color=colors, edgecolor="white")
        ax.axvline(0, color=NEUTRAL, lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("First-day return")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        for y, v in enumerate(data["underpricing"]):
            ax.text(v, y, f" {v:+.0%}", va="center", fontsize=8,
                    ha="left" if v >= 0 else "right")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=sector_colors[s])
                      for s in SECTOR_ORDER if s in df["sector"].unique()]
    legend_labels = [s for s in SECTOR_ORDER if s in df["sector"].unique()]
    fig.legend(legend_handles, legend_labels,
               loc="lower center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Most extreme first-day moves  (2019-2024)", fontsize=13)
    fig.tight_layout()
    save(fig, "11_top_bottom_ipos.png")


# ---------------------------------------------------------------------------
# Figure 12 — Fog index vs underpricing
# ---------------------------------------------------------------------------

def fig_fog_vs_underpricing(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["gunning_fog", "underpricing"]).copy()
    lo, hi = data["underpricing"].quantile([0.05, 0.95])
    data = data[(data["underpricing"] >= lo) & (data["underpricing"] <= hi)]

    data["bin"] = pd.qcut(data["gunning_fog"], 10, duplicates="drop")
    agg = data.groupby("bin", observed=True).agg(
        x=("gunning_fog", "mean"),
        med=("underpricing", "median"),
        q25=("underpricing", lambda s: s.quantile(0.25)),
        q75=("underpricing", lambda s: s.quantile(0.75)),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data["gunning_fog"], data["underpricing"],
               s=10, alpha=0.22, color=NEUTRAL)
    ax.fill_between(agg["x"], agg["q25"], agg["q75"], color=GREEN, alpha=0.18, label="IQR")
    ax.plot(agg["x"], agg["med"], color=GREEN, lw=2, marker="o", label="Decile median")
    ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Gunning-Fog index of MD&A  (higher = harder to read)")
    ax.set_ylabel("First-day return")
    ax.set_title("Readability vs. Underpricing  (5%-95% trimmed)")
    ax.legend()
    save(fig, "12_fog_vs_underpricing.png")


# ---------------------------------------------------------------------------
# Figure 13 — LM dictionary: words per category
# ---------------------------------------------------------------------------

def fig_lm_dictionary_summary() -> None:
    lm_path = ROOT / "data" / "external" / "lm_dictionary.csv"
    lm = pd.read_csv(lm_path, low_memory=False)
    rows = [
        ("Negative", (lm["Negative"] != 0).sum()),
        ("Positive", (lm["Positive"] != 0).sum()),
        ("Uncertainty", (lm["Uncertainty"] != 0).sum()),
        ("Litigious", (lm["Litigious"] != 0).sum()),
        ("Constraining", (lm["Constraining"] != 0).sum()),
        ("Modal — Strong", (lm["Modal"] == 1).sum() if "Modal" in lm.columns else 0),
        ("Modal — Weak", (lm["Modal"] == 3).sum() if "Modal" in lm.columns else 0),
    ]
    cats, counts = zip(*rows)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(cats, counts, color=[ACCENT, GREEN, ORANGE, NEUTRAL,
                                       "#7c3aed", "#0891b2", "#ca8a04"])
    ax.set_yscale("log")
    ax.set_title(f"Loughran-McDonald Master Dictionary  (n={len(lm):,} total words, log y-axis)")
    ax.set_ylabel("Words in category (log)")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, n, f"{n}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    save(fig, "13_lm_dictionary_summary.png")


# ---------------------------------------------------------------------------
# Figure 14 — Sentiment by sector (boxplot)
# ---------------------------------------------------------------------------

def fig_sentiment_by_sector(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["lm_negative_ratio", "sector"])
    order = (data.groupby("sector")["lm_negative_ratio"]
                  .median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=data, x="lm_negative_ratio", y="sector",
                order=order, color=ACCENT, fliersize=2, ax=ax)
    ax.set_title("LM Negative-Sentiment Ratio by Sector  (full prospectus)")
    ax.set_xlabel("Fraction of words flagged Negative")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save(fig, "14_sentiment_by_sector.png")


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
    fig_sector_boxplot(df)
    fig_monthly_volume(df)
    fig_sentiment_vs_underpricing(df)
    fig_calendar_heatmap(df)
    fig_hot_vs_cold(df)
    fig_vix_scatter(df)
    fig_correlation_heatmap(df)
    fig_top_bottom_ipos(df)
    fig_fog_vs_underpricing(df)
    fig_lm_dictionary_summary()
    fig_sentiment_by_sector(df)
    print("\nDone. Figures in:", FIG_DIR.relative_to(ROOT))


if __name__ == "__main__":
    main()
