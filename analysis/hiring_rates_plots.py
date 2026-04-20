"""Plotting helpers for hiring rate analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def _save_or_show(fig, output_path: Path, filename: str, show_plots: bool) -> None:
    """Save a figure and optionally display it."""
    fig.tight_layout()
    fig.savefig(output_path / filename)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _build_top_company_colors(count: int) -> list[str]:
    """Return a stable color palette for top company charts."""
    base_colors = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#BA7517"]
    if count <= len(base_colors):
        return base_colors[:count]

    cmap = mpl.colormaps["tab20"]
    extras = [mcolors.to_hex(cmap(i / count)) for i in range(count - len(base_colors))]
    return base_colors + extras


def plot_analysis_1_top_companies(
    top_companies: pd.DataFrame,
    top_n: int,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot top companies by number of postings."""
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    ax_bar.barh(
        top_companies["company_name"][::-1],
        top_companies["posting_count"][::-1],
        color="steelblue",
    )
    ax_bar.set_xlabel("Number of Job Postings")
    ax_bar.set_title(f"Top {top_n} Companies by Job Postings")

    for i, (val, _) in enumerate(
        zip(top_companies["posting_count"][::-1], top_companies["company_name"][::-1])
    ):
        ax_bar.text(val + 0.2, i, str(val), va="center")

    _save_or_show(fig_bar, output_path, f"{ax_bar.get_title()}.png", show_plots)


def plot_analysis_1_distribution(
    postings_per_company: pd.DataFrame,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot posting count distribution across companies."""
    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    ax_hist.hist(postings_per_company["posting_count"], bins=40, color="steelblue", edgecolor="white")
    ax_hist.set_xlabel("Number of Postings")
    ax_hist.set_ylabel("Number of Companies")
    ax_hist.set_title("Distribution of Job Postings across companies")
    ax_hist.axvline(postings_per_company["posting_count"].median(), color="red", linestyle="--", label="Median")
    ax_hist.axvline(postings_per_company["posting_count"].mean(), color="orange", linestyle="--", label="Mean")
    ax_hist.legend()

    _save_or_show(fig_hist, output_path, f"{ax_hist.get_title()}.png", show_plots)


def plot_analysis_2_hiring_rate_bar(
    hiring_rate: pd.DataFrame,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot hiring rate as average days between postings per company."""
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    ax_bar.barh(hiring_rate["company_name"][::-1], hiring_rate["avg_days_between_posts"][::-1], color="steelblue")
    ax_bar.set_xlabel("Avg Days Between Job Postings")
    ax_bar.set_title("Hiring Rate by Company\n(Lower = Posts More Frequently)")

    for i, val in enumerate(hiring_rate["avg_days_between_posts"][::-1]):
        ax_bar.text(val + 0.1, i, f"{val:.1f}d", va="center")

    _save_or_show(fig_bar, output_path, f"{ax_bar.get_title()[:22]}.png", show_plots)


def plot_analysis_2_hiring_rate_alt(
    hiring_rate_alt: pd.DataFrame,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot alternate hiring-rate chart with posting count encoded by color."""
    norm = mcolors.Normalize(vmin=hiring_rate_alt["posting_count"].min(), vmax=hiring_rate_alt["posting_count"].max())
    cmap = mpl.colormaps["OrRd"]
    colors = cmap(norm(hiring_rate_alt["posting_count"]))

    fig_alt, ax_alt = plt.subplots(figsize=(11, 8))
    ax_alt.barh(hiring_rate_alt["company_name"], hiring_rate_alt["avg_days_between_posts"], color=colors)

    for i, (days, count) in enumerate(zip(hiring_rate_alt["avg_days_between_posts"], hiring_rate_alt["posting_count"])):
        ax_alt.text(days + 0.1, i, f"{days:.1f}d", va="center", fontsize=9)
        ax_alt.text(-0.3, i, f"n={count}", va="center", ha="right", fontsize=8, color="gray")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_alt, pad=0.01)
    cbar.set_label("Number of Postings", fontsize=10)

    ax_alt.set_xlabel("Avg Days Between Job Postings")
    ax_alt.set_title(
        "Hiring Rate by Company Alt\n(Color = Volume of Postings, Lower Days = More Frequent)",
        pad=12,
    )
    ax_alt.spines[["top", "right"]].set_visible(False)
    ax_alt.set_xlim(left=-1.5)
    ax_alt.grid(axis="x", linestyle="--", alpha=0.3)

    _save_or_show(fig_alt, output_path, f"{ax_alt.get_title()[:26]}.png", show_plots)


def plot_analysis_3_stacked_area(
    timeline_wide: pd.DataFrame,
    top_n: int,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot stacked area chart for top company posting timelines."""
    colors = _build_top_company_colors(len(timeline_wide.columns))

    fig_stack, ax_stack = plt.subplots(figsize=(12, 6))
    ax_stack.stackplot(
        timeline_wide.index,
        [timeline_wide[col] for col in timeline_wide.columns],
        labels=timeline_wide.columns,
        colors=colors,
        alpha=0.85,
    )

    ax_stack.set_title(
        f"Job Postings Over Time - Top {top_n} Companies - Stacked Area Chart",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax_stack.set_xlabel("Week", fontsize=11)
    ax_stack.set_ylabel("Number of Postings", fontsize=11)

    ax_stack.tick_params(axis="x", rotation=45)
    ax_stack.spines[["top", "right"]].set_visible(False)
    ax_stack.grid(axis="y", linestyle="--", alpha=0.4)

    ax_stack.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_stack.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    ax_stack.legend(loc="upper left", frameon=False, fontsize=10)

    _save_or_show(fig_stack, output_path, f"{ax_stack.get_title()}.png", show_plots)


def plot_analysis_3_enhanced_line(
    timeline_wide: pd.DataFrame,
    top_n: int,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Plot enhanced multi-line chart with annotations for top companies."""
    colors = _build_top_company_colors(len(timeline_wide.columns))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    fig_enhanced, ax_enhanced = plt.subplots(figsize=(13, 6))
    fig_enhanced.patch.set_facecolor("#F9F9F9")
    ax_enhanced.set_facecolor("#F9F9F9")

    for idx, col in enumerate(timeline_wide.columns):
        color = colors[idx]
        marker = markers[idx % len(markers)]
        series = timeline_wide[col]

        ax_enhanced.fill_between(series.index, series.values, alpha=0.08, color=color)
        ax_enhanced.plot(
            series.index,
            series.values,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=col,
            zorder=3,
        )

        ax_enhanced.annotate(
            col,
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            va="center",
            fontweight="bold",
        )

    ax_enhanced.set_title(
        f"Enhanced Job Postings Over Time - Top {top_n} Companies",
        fontsize=14,
        fontweight="bold",
        pad=15,
        loc="left",
    )
    ax_enhanced.set_xlabel("Week", fontsize=11, labelpad=10)
    ax_enhanced.set_ylabel("Number of Postings", fontsize=11, labelpad=10)

    ax_enhanced.legend(
        loc="upper left",
        frameon=False,
        fontsize=9.5,
        labelcolor="linecolor",
    )

    ax_enhanced.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_enhanced.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_enhanced.tick_params(axis="x", rotation=45)

    ax_enhanced.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax_enhanced.spines[["top", "right"]].set_visible(False)
    ax_enhanced.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax_enhanced.tick_params(colors="#666666")

    ax_enhanced.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
    ax_enhanced.grid(axis="x", linestyle=":", alpha=0.2, color="#CCCCCC")

    fig_enhanced.subplots_adjust(right=0.78)

    _save_or_show(fig_enhanced, output_path, "Enhanced Job Postings Over Time.png", show_plots)