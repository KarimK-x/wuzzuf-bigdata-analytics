"""Reusable analysis functions extracted from HiringRatesAnalysis.ipynb.

This module currently includes:
- Analysis 1: Number of job postings per company.
- Analysis 2: Hiring rate (average days between postings).
- Analysis 3: Postings over time for top companies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .hiring_rates_plots import (
        plot_analysis_1_distribution,
        plot_analysis_1_top_companies,
        plot_analysis_2_hiring_rate_alt,
        plot_analysis_2_hiring_rate_bar,
        plot_analysis_3_enhanced_line,
        plot_analysis_3_stacked_area,
    )
except ImportError:
    from hiring_rates_plots import (
        plot_analysis_1_distribution,
        plot_analysis_1_top_companies,
        plot_analysis_2_hiring_rate_alt,
        plot_analysis_2_hiring_rate_bar,
        plot_analysis_3_enhanced_line,
        plot_analysis_3_stacked_area,
    )


def _project_root() -> Path:
    """Return the project root directory (parent of analysis folder)."""
    return Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    """Resolve paths robustly for script usage from different working directories."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    # Keep behavior for callers that intentionally pass cwd-relative paths.
    if candidate.exists():
        return candidate.resolve()

    # Fallback to project-root-relative paths, e.g. "cleaning/...".
    return (_project_root() / candidate).resolve()


def load_hiring_rates_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the grouped cleaned dataset used by the notebook."""
    resolved_csv_path = _resolve_path(csv_path)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_csv_path}")
    return pd.read_csv(resolved_csv_path)


def preprocess_hiring_rates_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook preprocessing steps required before running analyses."""
    result = df.copy()

    drop_candidates = ["Unnamed: 21", "city.1", "country.1"]
    existing_columns = [col for col in drop_candidates if col in result.columns]
    if existing_columns:
        result = result.drop(columns=existing_columns)

    result["posted_at"] = pd.to_datetime(result["posted_at"])
    return result


def run_analysis_1_number_of_job_postings_per_company(
    df: pd.DataFrame,
    output_dir: str | Path,
    min_postings: int = 4,
    top_n: int = 30,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index]:
    """Run Analysis 1 from the notebook and save the generated figures.

    Returns a dictionary of intermediate dataframes used by downstream analyses.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Companies eligible for hiring rate analysis.
    df_clean = df[df["company_name"] != "Unknown"].copy()
    posting_counts = df_clean["company_name"].value_counts()
    eligible_companies_index = posting_counts[posting_counts >= min_postings].index

    postings_per_company = df_clean["company_name"].value_counts().reset_index()
    postings_per_company.columns = ["company_name", "posting_count"]

    top_companies = postings_per_company.head(top_n)

    plot_analysis_1_top_companies(
        top_companies=top_companies,
        top_n=top_n,
        output_path=output_path,
        show_plots=show_plots,
    )
    plot_analysis_1_distribution(
        postings_per_company=postings_per_company,
        output_path=output_path,
        show_plots=show_plots,
    )

    return {
        "df_clean": df_clean,
        "posting_counts": posting_counts,
        "postings_per_company": postings_per_company,
        "eligible_companies_index": eligible_companies_index,
    }


def run_analysis_1_from_csv(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_postings: int = 4,
    top_n: int = 30,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index]:
    """Convenience wrapper to run Analysis 1 directly from the dataset CSV."""
    input_csv = input_csv or (_project_root() / "cleaning" / "clean_data_grouped_saeed.csv")
    output_dir = output_dir or (Path(__file__).resolve().parent / "output")

    df = load_hiring_rates_dataset(input_csv)
    df = preprocess_hiring_rates_dataset(df)
    return run_analysis_1_number_of_job_postings_per_company(
        df=df,
        output_dir=output_dir,
        min_postings=min_postings,
        top_n=top_n,
        show_plots=show_plots,
    )


def _avg_days_between_posts(series: pd.Series) -> float:
    """Compute average gap in days between consecutive posting dates."""
    gaps = series.diff().dt.days.dropna()
    return float(gaps.mean()) if not gaps.empty else float("nan")


def run_analysis_2_hiring_rate_avg_days_between_postings(
    df: pd.DataFrame,
    output_dir: str | Path,
    min_postings: int = 4,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index]:
    """Run Analysis 2 from the notebook and save the generated figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_clean = df[df["company_name"] != "Unknown"].copy()
    posting_counts = df_clean["company_name"].value_counts()
    eligible_companies_index = posting_counts[posting_counts >= min_postings].index

    df_eligible = df_clean[df_clean["company_name"].isin(eligible_companies_index)].copy()
    df_eligible = df_eligible.sort_values(["company_name", "posted_at"])

    hiring_rate = (
        df_eligible.groupby("company_name")["posted_at"]
        .apply(_avg_days_between_posts)
        .reset_index()
    )
    hiring_rate.columns = ["company_name", "avg_days_between_posts"]
    hiring_rate = hiring_rate.sort_values("avg_days_between_posts")

    plot_analysis_2_hiring_rate_bar(
        hiring_rate=hiring_rate,
        output_path=output_path,
        show_plots=show_plots,
    )

    # Chart 2: Alternate hiring-rate view with posting volume color encoding.
    hiring_rate = hiring_rate.copy()
    hiring_rate["posting_count"] = hiring_rate["company_name"].map(posting_counts)
    hiring_rate = hiring_rate.sort_values("avg_days_between_posts")

    hr = hiring_rate[::-1].reset_index(drop=True)

    plot_analysis_2_hiring_rate_alt(
        hiring_rate_alt=hr,
        output_path=output_path,
        show_plots=show_plots,
    )

    return {
        "df_clean": df_clean,
        "posting_counts": posting_counts,
        "eligible_companies_index": eligible_companies_index,
        "df_eligible": df_eligible,
        "hiring_rate": hiring_rate,
        "hiring_rate_alt": hr,
    }


def run_analysis_2_from_csv(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_postings: int = 4,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index]:
    """Convenience wrapper to run Analysis 2 directly from the dataset CSV."""
    input_csv = input_csv or (_project_root() / "cleaning" / "clean_data_grouped_saeed.csv")
    output_dir = output_dir or (Path(__file__).resolve().parent / "output")

    df = load_hiring_rates_dataset(input_csv)
    df = preprocess_hiring_rates_dataset(df)
    return run_analysis_2_hiring_rate_avg_days_between_postings(
        df=df,
        output_dir=output_dir,
        min_postings=min_postings,
        show_plots=show_plots,
    )


def run_analysis_3_postings_over_time_top_companies(
    df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 5,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index | list[str]]:
    """Run Analysis 3 from the notebook and save the generated figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_clean = df[df["company_name"] != "Unknown"].copy()
    postings_per_company = df_clean["company_name"].value_counts().reset_index()
    postings_per_company.columns = ["company_name", "posting_count"]

    top_companies = postings_per_company.head(top_n)["company_name"].tolist()

    df_top = df_clean[df_clean["company_name"].isin(top_companies)].copy()
    df_top["week"] = df_top["posted_at"].dt.to_period("W").dt.start_time

    # Shared timeline table for alternate charts.
    timeline_wide = (
        df_top.groupby(["week", "company_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_companies)
    )

    plot_analysis_3_stacked_area(
        timeline_wide=timeline_wide,
        top_n=top_n,
        output_path=output_path,
        show_plots=show_plots,
    )
    plot_analysis_3_enhanced_line(
        timeline_wide=timeline_wide,
        top_n=top_n,
        output_path=output_path,
        show_plots=show_plots,
    )

    return {
        "df_clean": df_clean,
        "postings_per_company": postings_per_company,
        "top_companies": top_companies,
        "df_top_companies": df_top,
        "timeline_wide": timeline_wide,
    }


def run_analysis_3_from_csv(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 5,
    show_plots: bool = False,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index | list[str]]:
    """Convenience wrapper to run Analysis 3 directly from the dataset CSV."""
    input_csv = input_csv or (_project_root() / "cleaning" / "clean_data_grouped_saeed.csv")
    output_dir = output_dir or (Path(__file__).resolve().parent / "output")

    df = load_hiring_rates_dataset(input_csv)
    df = preprocess_hiring_rates_dataset(df)
    return run_analysis_3_postings_over_time_top_companies(
        df=df,
        output_dir=output_dir,
        top_n=top_n,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    run_analysis_1_from_csv()
    run_analysis_2_from_csv()
    run_analysis_3_from_csv()