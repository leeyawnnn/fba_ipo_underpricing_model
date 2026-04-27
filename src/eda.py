"""
Reusable plotting helpers for the EDA section.

All functions return a figure object (plotly or matplotlib) *and* save a PNG
to ``reports/figures/``.  Interactive charts use plotly; publication-quality
static charts use seaborn/matplotlib.

Convention: every function accepts ``save_path`` (defaults to
``reports/figures/<descriptive_name>.png``) so callers can override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from src.utils import setup_logging

log = setup_logging(__name__)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette
PALETTE_MAIN = "#2563EB"        # blue
PALETTE_ACCENT = "#DC2626"      # red
PALETTE_NEUTRAL = "#6B7280"     # grey
PLOTLY_TEMPLATE = "plotly_white"


def _save(fig, path: Path) -> None:
    """Save a figure (matplotlib or plotly) to *path* as PNG.

    Args:
        fig: A matplotlib Figure or plotly Figure.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fig, "write_image"):
        fig.write_image(str(path))
    else:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Figure saved → %s", path)


# ---------------------------------------------------------------------------
# 1. Distribution of first-day returns
# ---------------------------------------------------------------------------

def plot_return_distribution(
    df: pd.DataFrame,
    col: str = "underpricing",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Histogram + KDE of first-day IPO returns with median and zero lines.

    Args:
        df: IPO DataFrame.
        col: Column name for the return variable.
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / "01_return_distribution.png"
    data = df[col].dropna()
    median_val = data.median()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=80, name="Count",
        marker_color=PALETTE_MAIN, opacity=0.7,
        histnorm="probability density",
    ))

    # Approximate KDE using a smoothed histogram
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method="scott")
        x_range = np.linspace(data.quantile(0.005), data.quantile(0.995), 300)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde(x_range), mode="lines",
            line=dict(color=PALETTE_ACCENT, width=2.5), name="KDE",
        ))
    except ImportError:
        pass

    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE_NEUTRAL,
                  annotation_text="Zero return")
    fig.add_vline(x=median_val, line_dash="dot", line_color=PALETTE_ACCENT,
                  annotation_text=f"Median {median_val:.1%}")

    fig.update_layout(
        title="Distribution of First-Day IPO Returns",
        xaxis_title="Underpricing  (first-day return)",
        yaxis_title="Density",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
    )
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Monthly average underpricing time series
# ---------------------------------------------------------------------------

def plot_monthly_underpricing(
    df: pd.DataFrame,
    date_col: str = "ipo_date",
    col: str = "underpricing",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Monthly average underpricing + IPO count on secondary axis.

    Args:
        df: IPO DataFrame.
        date_col: Name of the IPO date column.
        col: Underpricing column.
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / "02_monthly_underpricing_ts.png"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["ym"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("ym").agg(
        avg_underpricing=(col, "median"),
        ipo_count=(col, "count"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=monthly["ym"], y=monthly["avg_underpricing"],
                   name="Median underpricing", line=dict(color=PALETTE_MAIN, width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=monthly["ym"], y=monthly["ipo_count"],
               name="# IPOs", marker_color=PALETTE_NEUTRAL, opacity=0.4),
        secondary_y=True,
    )
    fig.update_layout(
        title="Monthly Median Underpricing and IPO Volume",
        xaxis_title="Month", template=PLOTLY_TEMPLATE,
    )
    fig.update_yaxes(title_text="Median first-day return", secondary_y=False)
    fig.update_yaxes(title_text="Number of IPOs", secondary_y=True)
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Underpricing by sector — boxplot
# ---------------------------------------------------------------------------

def plot_sector_boxplot(
    df: pd.DataFrame,
    sector_col: str = "sector",
    col: str = "underpricing",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal boxplot of underpricing by sector, sorted by median.

    Args:
        df: IPO DataFrame.
        sector_col: Sector/industry column name.
        col: Underpricing column.
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "03_sector_boxplot.png"

    data = df[[sector_col, col]].dropna()
    order = data.groupby(sector_col)[col].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(6, len(order) * 0.45)))
    sns.boxplot(
        data=data, y=sector_col, x=col, order=order,
        palette="Blues_r", ax=ax, fliersize=2,
    )
    ax.axvline(0, color=PALETTE_NEUTRAL, linestyle="--", lw=1)
    ax.set_title("Underpricing by Sector (sorted by median)")
    ax.set_xlabel("First-day return")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Underpricing vs. offer size (log-log scatter)
# ---------------------------------------------------------------------------

def plot_offer_size_scatter(
    df: pd.DataFrame,
    col: str = "underpricing",
    size_col: str = "offer_size_m",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Scatter of underpricing vs. log offer size with LOESS smoother.

    Args:
        df: IPO DataFrame.
        col: Underpricing column.
        size_col: Offer size column (in $M).
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / "04_offer_size_scatter.png"

    data = df[[col, size_col]].dropna()
    data = data[data[size_col] > 0]

    fig = px.scatter(
        data, x=np.log10(data[size_col]), y=col,
        opacity=0.4, labels={"x": "log₁₀(Offer Size, $M)", "y": "First-day return"},
        title="Underpricing vs. Offer Size (log-log)",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[PALETTE_MAIN],
    )

    # Add binned medians as an overlay
    data["log_size_bin"] = pd.cut(np.log10(data[size_col]), bins=15)
    bin_medians = data.groupby("log_size_bin")[col].median().reset_index()
    bin_medians["x"] = bin_medians["log_size_bin"].apply(lambda iv: iv.mid)
    fig.add_trace(go.Scatter(
        x=bin_medians["x"], y=bin_medians[col], mode="lines+markers",
        name="Binned median", line=dict(color=PALETTE_ACCENT, width=2.5),
    ))
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Underpricing vs. VIX
# ---------------------------------------------------------------------------

def plot_vix_scatter(
    df: pd.DataFrame,
    col: str = "underpricing",
    vix_col: str = "vix_at_pricing",
    hot_col: str = "hot_market_dummy",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Scatter of underpricing vs. VIX, coloured by hot/cold market.

    Args:
        df: IPO DataFrame.
        col: Underpricing column.
        vix_col: VIX column.
        hot_col: Hot-market dummy column.
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / "05_vix_scatter.png"

    data = df[[col, vix_col, hot_col]].dropna()
    data["Market"] = data[hot_col].map({1: "Hot", 0: "Cold"})

    fig = px.scatter(
        data, x=vix_col, y=col, color="Market",
        opacity=0.45,
        color_discrete_map={"Hot": PALETTE_MAIN, "Cold": PALETTE_NEUTRAL},
        labels={vix_col: "VIX (at pricing date)", col: "First-day return"},
        title="Underpricing vs. VIX at Pricing",
        template=PLOTLY_TEMPLATE,
    )
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Underwriter tier — violin plot
# ---------------------------------------------------------------------------

def plot_underwriter_violin(
    df: pd.DataFrame,
    col: str = "underpricing",
    tier_col: str = "top_tier_underwriter",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Violin plot comparing underpricing for top-tier vs. non-top-tier banks.

    Args:
        df: IPO DataFrame.
        col: Underpricing column.
        tier_col: Binary top-tier underwriter column.
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "06_underwriter_violin.png"

    data = df[[col, tier_col]].dropna()
    data["Underwriter tier"] = data[tier_col].map({1.0: "Top-tier (rank ≥8)", 0.0: "Non-top-tier"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=data, x="Underwriter tier", y=col, palette=["#2563EB", "#6B7280"],
                   inner="box", ax=ax)
    ax.set_title("Underpricing by Underwriter Tier")
    ax.set_xlabel("")
    ax.set_ylabel("First-day return")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7 & 8. Text feature vs. underpricing — generic scatter with binned medians
# ---------------------------------------------------------------------------

def plot_text_feature_scatter(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "underpricing",
    n_bins: int = 20,
    title: str = "",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Scatter + binned-median overlay for any text feature vs. underpricing.

    Args:
        df: IPO DataFrame.
        feature_col: Feature column name.
        target_col: Target column name.
        n_bins: Number of quantile bins for the median overlay.
        title: Plot title.
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / f"scatter_{feature_col}.png"
    data = df[[feature_col, target_col]].dropna()

    fig = px.scatter(
        data, x=feature_col, y=target_col,
        opacity=0.3, color_discrete_sequence=[PALETTE_MAIN],
        labels={feature_col: feature_col.replace("_", " ").title(),
                target_col: "First-day return"},
        title=title or f"{feature_col.replace('_', ' ').title()} vs. Underpricing",
        template=PLOTLY_TEMPLATE,
    )

    data["_bin"] = pd.qcut(data[feature_col], q=n_bins, duplicates="drop")
    binned = data.groupby("_bin", observed=True)[target_col].median().reset_index()
    binned["x_mid"] = binned["_bin"].apply(lambda iv: iv.mid)

    fig.add_trace(go.Scatter(
        x=binned["x_mid"], y=binned[target_col], mode="lines+markers",
        name="Binned median",
        line=dict(color=PALETTE_ACCENT, width=2.5),
    ))
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 11. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str = "underpricing",
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Annotated correlation heatmap of numeric features vs. the target.

    Args:
        df: IPO DataFrame.
        target_col: Target column.
        top_n: Maximum number of features to include (by absolute correlation
            with the target).
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "11_correlation_heatmap.png"

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if target_col not in numeric.columns:
        log.warning("Target column %s not in numeric columns.", target_col)
        return plt.figure()

    corr_with_target = numeric.corr()[target_col].drop(target_col).abs().nlargest(top_n)
    top_cols = corr_with_target.index.tolist() + [target_col]
    corr_matrix = numeric[top_cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(top_cols) * 0.7), max(8, len(top_cols) * 0.65)))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Correlation Heatmap (top {top_n} features + target)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 13. Calendar heatmap (IPO volume year × month)
# ---------------------------------------------------------------------------

def plot_calendar_heatmap(
    df: pd.DataFrame,
    date_col: str = "ipo_date",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Calendar heatmap showing IPO volume by year and month.

    Args:
        df: IPO DataFrame.
        date_col: Date column name.
        save_path: Output PNG path.

    Returns:
        Matplotlib Figure.
    """
    save_path = save_path or FIGURES_DIR / "13_calendar_heatmap.png"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    pivot = (
        df.groupby([df[date_col].dt.year, df[date_col].dt.month])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: pivot.shape[1]]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.65)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title("IPO Volume by Year and Month")
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 14. Sector × hot-market interaction
# ---------------------------------------------------------------------------

def plot_sector_market_interaction(
    df: pd.DataFrame,
    col: str = "underpricing",
    sector_col: str = "sector",
    hot_col: str = "hot_market_dummy",
    top_n_sectors: int = 8,
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Grouped bar chart: sector median underpricing in hot vs. cold markets.

    Args:
        df: IPO DataFrame.
        col: Underpricing column.
        sector_col: Sector column.
        hot_col: Hot-market dummy column.
        top_n_sectors: Number of most-represented sectors to show.
        save_path: Output PNG path.

    Returns:
        Plotly Figure.
    """
    save_path = save_path or FIGURES_DIR / "14_sector_market_interaction.png"

    data = df[[col, sector_col, hot_col]].dropna()
    top_sectors = data[sector_col].value_counts().head(top_n_sectors).index.tolist()
    data = data[data[sector_col].isin(top_sectors)]

    summary = (
        data.groupby([sector_col, hot_col])[col]
        .median()
        .reset_index()
    )
    summary["Market"] = summary[hot_col].map({1: "Hot", 0: "Cold"})

    fig = px.bar(
        summary, x=sector_col, y=col, color="Market", barmode="group",
        color_discrete_map={"Hot": PALETTE_MAIN, "Cold": PALETTE_NEUTRAL},
        labels={col: "Median first-day return", sector_col: "Sector"},
        title="Sector Underpricing: Hot vs. Cold Markets",
        template=PLOTLY_TEMPLATE,
    )
    _save(fig, save_path)
    return fig
