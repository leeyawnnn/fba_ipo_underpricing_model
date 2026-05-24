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
    """Sector × year view, split into THREE separate figures.

    The original was a single 11×6 heatmap crammed with both a median
    value and an IPO count per cell — too dense to read at a glance.
    Splitting:
      03a  — two heatmaps (median return, IPO count) side-by-side for the
             six high-volume sectors only
      03b  — small-multiples line chart, one panel per sector, showing the
             median first-day return trajectory year-over-year
      03c  — small-multiples bar chart, one panel per sector, showing IPO
             volume per year
    """
    df = df.copy()
    df["winsorized_underpricing"] = df["winsorized_underpricing"]
    sector_counts = df["sector"].value_counts()
    major = [s for s in SECTOR_ORDER if s in sector_counts.head(6).index]
    minor = [s for s in SECTOR_ORDER if s in sector_counts.index and s not in major]

    pivot_med = (df.groupby(["sector", "ipo_year"])["winsorized_underpricing"]
                   .median().unstack().reindex(SECTOR_ORDER))
    pivot_n = df.groupby(["sector", "ipo_year"]).size().unstack().reindex(SECTOR_ORDER)

    # ------------------------------------------------------------------
    # 03  — original single heatmap (kept for backwards compatibility,
    # but now restricted to the six high-volume sectors so cell colours
    # aren't dominated by sparse rows)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot_med.loc[major] * 100,
        annot=pivot_n.loc[major].astype("Int64"), fmt="d",
        cmap="RdBu_r", center=0, vmin=-30, vmax=30,
        cbar_kws={"label": "Median first-day return (%)"},
        linewidths=0.4, ax=ax,
    )
    ax.set_title(
        "Median first-day return by sector × year — high-volume sectors only\n"
        "(cells annotated with IPO count; smaller sectors split into 03b / 03c)"
    )
    ax.set_xlabel("IPO year"); ax.set_ylabel("")
    save(fig, "03_sector_year_heatmap.png")

    # ------------------------------------------------------------------
    # 03b — small-multiples: median first-day return per sector, by year
    # ------------------------------------------------------------------
    sectors_present = [s for s in SECTOR_ORDER if s in sector_counts.index]
    n = len(sectors_present)
    cols_grid = 4
    rows_grid = int(np.ceil(n / cols_grid))
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 3.0 * rows_grid),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    palette = sns.color_palette("tab10", n)
    for ax, sector, color in zip(axes, sectors_present, palette):
        s = pivot_med.loc[sector] * 100
        counts = pivot_n.loc[sector]
        ax.plot(s.index, s.values, color=color, lw=2, marker="o")
        ax.axhline(0, color=NEUTRAL, ls="--", lw=0.7)
        ax.set_title(f"{sector}  (n={int(counts.sum()):,})", fontsize=10)
        ax.set_ylim(-60, 50)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        # Annotate each marker with the per-year IPO count
        for x_yr, y_val, n_yr in zip(s.index, s.values, counts.values):
            if pd.notna(y_val) and pd.notna(n_yr):
                ax.annotate(f"n={int(n_yr)}", (x_yr, y_val),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7, color=NEUTRAL)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Median first-day return by sector × year — small-multiples view\n"
        "(one panel per sector; marker labels show IPO count that year)",
        fontsize=13, y=1.0,
    )
    fig.supxlabel("IPO year")
    fig.supylabel("Median first-day return")
    fig.tight_layout()
    save(fig, "03b_sector_year_returns_smallmultiples.png")

    # ------------------------------------------------------------------
    # 03c — small-multiples: IPO volume per sector, by year
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, 3.0 * rows_grid),
                             sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, sector, color in zip(axes, sectors_present, palette):
        counts = pivot_n.loc[sector].fillna(0).astype(int)
        bars = ax.bar(counts.index, counts.values, color=color, edgecolor="white")
        ax.set_title(f"{sector}  (total n={int(counts.sum()):,})", fontsize=10)
        for bar, n_yr in zip(bars, counts.values):
            if n_yr > 0:
                ax.annotate(f"{n_yr}", (bar.get_x() + bar.get_width() / 2, n_yr),
                            ha="center", va="bottom", fontsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "IPO volume by sector × year — small-multiples view",
        fontsize=13, y=1.0,
    )
    fig.supxlabel("IPO year")
    fig.supylabel("Number of IPOs")
    fig.tight_layout()
    save(fig, "03c_sector_year_counts_smallmultiples.png")


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
