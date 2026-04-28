import streamlit as st
import pandas as pd
import json
import ast
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# ======================================================= Saeed =======================================================

@st.cache_data
def load_notebook_dataset(file_path="public/Saeed/grouped_output.csv"):
    df_g = pd.read_csv(file_path)
    def parse_exp(val):
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, str):
            try:
                res = ast.literal_eval(val)
                return res if isinstance(res, list) else []
            except (ValueError, SyntaxError):
                return []
        return val if isinstance(val, list) else []
        
    df_g['experience_ranges'] = df_g['experience_ranges'].apply(parse_exp)
    return df_g

def plot_global_experience_distribution(df_g, levels):
    fig, ax = plt.subplots(figsize=(10, 6))
    level_counts = df_g[levels].sum()
    
    sns.barplot(x=levels, y=level_counts[levels].values, palette='magma', ax=ax)
    ax.set_title('Total Representation by Experience Level (All Jobs)', fontsize=16, pad=15)
    ax.set_xlabel('Career Level', fontsize=12)
    ax.set_ylabel('Cumulative Count', fontsize=12)
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, xytext=(0, 5), 
                    textcoords='offset points')
                    
    fig.tight_layout()
    return fig

def plot_stacked_levels_all_jobs(df, levels):
    df_sorted = df.sort_values('job_count', ascending=True).set_index('job_title')[levels]
    
    fig_height = max(10, len(df_sorted) * 0.35)
    
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    df_sorted.plot(kind='barh', stacked=True, colormap='Set2', edgecolor='white', linewidth=0.5, ax=ax)
    
    ax.set_title('Experience Level Composition per Job (All Jobs)', fontsize=16, pad=15)
    ax.set_xlabel('Cumulative Count of Levels', fontsize=12)
    ax.set_ylabel('Job Title', fontsize=12)
    ax.legend(title='Career Level', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    fig.tight_layout()
    return fig

def plot_heatmap_all_jobs(df, levels):
    df_sorted = df.sort_values('job_count', ascending=False).set_index('job_title')[levels]
    
    fig_height = max(10, len(df_sorted) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    sns.heatmap(df_sorted, annot=True, fmt='g', cmap='Blues', linewidths=.5, cbar_kws={'label': 'Count Distribution'}, ax=ax)
    
    ax.set_title('Career Levels Heatmap (All Jobs)', fontsize=16, pad=15)
    ax.set_xlabel('Career Level', fontsize=12)
    ax.set_ylabel('Job Title', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    return fig

# ======================================================= Lotfy =======================================================
def get_job_counts(df):
    job = df["job_title_group"].value_counts().reset_index()
    job.columns = ["job_title_group", "count"]
    return job


def plot_job_counts(df):
    job = get_job_counts(df)

    job_sorted = job.sort_values("count", ascending=False)
    fig_height = max(10, len(job_sorted) * 0.35)

    fig, ax = plt.subplots(figsize=(14, fig_height))

    ax.bar(job_sorted["job_title_group"], job_sorted["count"])

    ax.set_title("Most Common Needed Jobs", fontsize=16, pad=15)
    ax.set_xlabel("Job Title", fontsize=12)
    ax.set_ylabel("Number of Jobs", fontsize=12)

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig

def get_top_job_per_city(df):
    job_per_city = df.groupby(["job_title", "city"]).size().reset_index()
    job_per_city.columns = ["job_title", "city", "count"]

    job_per_city = job_per_city.loc[
        job_per_city.groupby("city")["count"].idxmax()
    ]
    sorted_job_per_city = job_per_city.sort_values(by="count", ascending=False)

    return sorted_job_per_city


def plot_top_job_per_city(df, path=None):
    data = get_top_job_per_city(df)

    fig_height = max(8, len(data) * 0.4)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(data["city"], data["count"])

    for i in range(len(data)):
        ax.text(
            data["count"].iloc[i],
            i,
            f" {data['job_title'].iloc[i]}",
            va='center'
        )

    ax.set_title("Most Common Job per City", fontsize=16, pad=15)
    ax.set_xlabel("Number of Jobs", fontsize=12)
    ax.set_ylabel("City", fontsize=12)

    plt.tight_layout()

    return fig

def get_top_job_per_country(df):
    job_per_country = df.groupby(["job_title", "country"]).size().reset_index()
    job_per_country.columns = ["job_title", "country", "count"]

    job_per_country = job_per_country.loc[
        job_per_country.groupby("country")["count"].idxmax()
    ]

    sorted_job_per_country = job_per_country.sort_values(by="count", ascending=False)

    return sorted_job_per_country


def plot_top_job_per_country(df, path=None):
    data = get_top_job_per_country(df)

    fig_height = max(8, len(data) * 0.4)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(data["country"], data["count"])

    for i in range(len(data)):
        ax.text(
            data["count"].iloc[i],
            i,
            f" {data['job_title'].iloc[i]}",
            va='center'
        )

    ax.set_title("Most Common Job per Country", fontsize=16, pad=15)
    ax.set_xlabel("Number of Jobs", fontsize=12)
    ax.set_ylabel("Country", fontsize=12)

    plt.tight_layout()

    return fig

# ======================================================= Zeyad =======================================================
def get_part_time_jobs_by_country_and_work_setting(df):
    part_time_df = df[df["work_type"].str.contains("Part Time", case=False, na=False)]
    country_of_part_time_jobs = part_time_df[["country", "work_setting"]].value_counts().unstack(fill_value=0).sort_index()
    return country_of_part_time_jobs

def plot_part_time_jobs_by_country_and_work_setting(df):
    country_of_part_time_jobs = get_part_time_jobs_by_country_and_work_setting(df)

    fig_height = 6

    ax = country_of_part_time_jobs.plot(kind="bar", figsize=(14, fig_height))
    ax.set_xlabel("Country")
    ax.set_ylabel("Count")
    ax.set_title("Part-time jobs by Country and Work Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig = ax.get_figure()
    return fig

def get_part_time_jobs_by_city_and_work_setting(df):
    part_time_df = df[df["work_type"].str.contains("Part Time", case=False, na=False)]
    cities_of_part_time_jobs = part_time_df[["city", "work_setting"]].value_counts().unstack(fill_value=0).sort_index()
    return cities_of_part_time_jobs

def plot_part_time_jobs_by_city_and_work_setting(df):
    cities_of_part_time_jobs = get_part_time_jobs_by_city_and_work_setting(df)

    fig_height = 6

    ax = cities_of_part_time_jobs.plot(kind="bar", figsize=(14, fig_height))
    ax.set_xlabel("City")
    ax.set_ylabel("Count")
    ax.set_title("Part-time jobs by City and Work Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig = ax.get_figure()
    return fig

def get_education_level_by_job_title_group(df):
    jt_edu = pd.crosstab(df["job_title_group"], df["education_level"])
    jt_edu = jt_edu.drop(columns=["Not Specified"], errors="ignore")
    top_groups = df["job_title_group"].value_counts().index
    jt_edu_top = jt_edu.loc[top_groups]
    return jt_edu_top

def plot_education_level_by_job_title_group(df):
    jt_edu_top = get_education_level_by_job_title_group(df)

    ax = jt_edu_top.plot(kind="bar", stacked=True, figsize=(14, 6))
    ax.set_title("Education level needed to secure job (by Job Title Group)")
    ax.set_xlabel("Job Title Group")
    ax.set_ylabel("Number of jobs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig = ax.get_figure()
    return fig

# ======================================================= Mohahed =======================================================
def prepare_treemap_data(file_path, top_n=15):
    sheets = pd.read_excel(file_path, sheet_name=None)

    rows = []
    for job, df in sheets.items():
        for _, row in df.head(top_n).iterrows():
            rows.append({
                'Job': job,
                'Skill': row['Skill'],
                'Count': row['Count']
            })

    combined = pd.DataFrame(rows)
    return combined


def plot_treemap(file_path, plot_name="Treemap", save_path=None):
    data = prepare_treemap_data(file_path)

    fig = px.treemap(
        data,
        path=['Job', 'Skill'],
        values='Count',
        title=plot_name,
        color='Count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )

    return fig

def get_top_skills(file_path, top_n=10):
    df = pd.read_csv(file_path).head(top_n)
    return df


def plot_skills(file_path, plot_name="Top Skills", save_path=None):
    data = get_top_skills(file_path)

    fig_height = max(6, len(data) * 0.5)

    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(data['Skill'], data['Count'])

    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(plot_name, fontsize=16, pad=15)

    ax.invert_yaxis()

    plt.tight_layout()

    return fig

# ======================================================= Karim =======================================================


def preprocess_hiring_rates_dataset(df):
    result = df.copy()

    drop_candidates = ["Unnamed: 21", "city.1", "country.1"]
    existing_columns = [col for col in drop_candidates if col in result.columns]
    if existing_columns:
        result = result.drop(columns=existing_columns)

    result["posted_at"] = pd.to_datetime(result["posted_at"])
    return result


def prepare_company_postings(df):
    df = preprocess_hiring_rates_dataset(df)
    df_clean = df[df["company_name"] != "Unknown"].copy()
    postings_per_company = (
        df_clean["company_name"]
        .value_counts()
        .rename_axis("company_name")
        .reset_index(name="posting_count")
    )
    return df_clean, postings_per_company


def prepare_hiring_rate(df_clean):
    posting_counts = df_clean["company_name"].value_counts()
    eligible = posting_counts[posting_counts >= 4].index

    df_eligible = (
        df_clean[df_clean["company_name"].isin(eligible)]
        .copy()
        .sort_values(["company_name", "posted_at"])
    )

    hiring_rate = (
        df_eligible.groupby("company_name")["posted_at"]
        .apply(lambda dates: dates.diff().dt.days.dropna().mean())
        .reset_index(name="avg_days_between_posts")
        .sort_values("avg_days_between_posts")
    )
    return hiring_rate, posting_counts


def top_company_timeline(df_clean):
    """Return weekly posting timeline for top companies."""
    top_companies = df_clean["company_name"].value_counts().head(5).index.tolist()
    return (
        df_clean[df_clean["company_name"].isin(top_companies)]
        .assign(week=lambda d: d["posted_at"].dt.to_period("W").dt.start_time)
        .groupby(["week", "company_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_companies)
    )

def style_week_axis(ax):
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mpl.dates.WeekdayLocator(interval=1))
    ax.tick_params(axis="x", rotation=45)


def plot_num_of_job_postings(df):
    _, postings_per_company = prepare_company_postings(df)
    top30 = postings_per_company.head(30)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top30["company_name"][::-1], top30["posting_count"][::-1], color="steelblue")
    ax.set_xlabel("Number of Job Postings")
    ax.set_title("Top 30 Companies by Job Postings")

    for i, val in enumerate(top30["posting_count"][::-1]):
        ax.text(val + 0.2, i, str(val), va="center")

    fig.tight_layout()
    return fig


def plot_histogram_distribution_postings(df):
    _, postings_per_company = prepare_company_postings(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(postings_per_company["posting_count"], bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of Postings")
    ax.set_ylabel("Number of Companies")
    ax.set_title("Distribution of Job Postings across companies")
    ax.axvline(postings_per_company["posting_count"].median(), color="red", linestyle="--", label="Median")
    ax.axvline(postings_per_company["posting_count"].mean(), color="orange", linestyle="--", label="Mean")
    ax.legend()

    fig.tight_layout()
    return fig


def plot_hiring_interval(df):
    df_clean, _ = prepare_company_postings(df)
    hiring_rate, _ = prepare_hiring_rate(df_clean)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(hiring_rate["company_name"][::-1], hiring_rate["avg_days_between_posts"][::-1], color="steelblue")
    ax.set_xlabel("Avg Days Between Job Postings")
    ax.set_title("Hiring Rate by Company\n(Lower = Posts More Frequently)")

    for i, val in enumerate(hiring_rate["avg_days_between_posts"][::-1]):
        ax.text(val + 0.1, i, f"{val:.1f}d", va="center")

    fig.tight_layout()
    return fig


def plot_hiring_interval_colored(df):
    df_clean, _ = prepare_company_postings(df)
    hiring_rate, posting_counts = prepare_hiring_rate(df_clean)

    hr = (
        hiring_rate.assign(posting_count=hiring_rate["company_name"].map(posting_counts))
        .iloc[::-1]
        .reset_index(drop=True)
    )

    norm = mpl.colors.Normalize(vmin=hr["posting_count"].min(), vmax=hr["posting_count"].max())
    cmap = mpl.colormaps["OrRd"]
    colors = cmap(norm(hr["posting_count"]))

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(hr["company_name"], hr["avg_days_between_posts"], color=colors)

    for i, (days, count) in enumerate(zip(hr["avg_days_between_posts"], hr["posting_count"])):
        ax.text(days + 0.1, i, f"{days:.1f}d", va="center", fontsize=9)
        ax.text(-0.3, i, f"n={count}", va="center", ha="right", fontsize=8, color="gray")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Number of Postings", fontsize=10)

    ax.set_xlabel("Avg Days Between Job Postings")
    ax.set_title("Hiring Rate by Company Alt\n(Color = Volume of Postings, Lower Days = More Frequent)", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(left=-1.5)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_top5(df):
    df_clean, _ = prepare_company_postings(df)
    timeline = top_company_timeline(df_clean)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#BA7517"]

    ax.stackplot(
        timeline.index,
        [timeline[col] for col in timeline.columns],
        labels=timeline.columns,
        colors=colors[: len(timeline.columns)],
        alpha=0.85,
    )

    ax.set_title("Job Postings Over Time - Top 5 Companies - Stacked Area Chart", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Number of Postings", fontsize=11)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    style_week_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    return fig


def plot_top5_enhanced(df):
    df_clean, _ = prepare_company_postings(df)
    timeline = top_company_timeline(df_clean)

    colors = ["#534AB7", "#1D9E75", "#D85A30", "#D4537E", "#BA7517"]
    markers = ["o", "s", "^", "D", "P"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    for col, color, marker in zip(timeline.columns, colors, markers):
        series = timeline[col]
        ax.fill_between(series.index, series.values, alpha=0.08, color=color)
        ax.plot(
            series.index,
            series.values,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            label=col,
            zorder=3,
        )
        ax.annotate(
            col,
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            va="center",
            fontweight="bold"
        )

    ax.set_title("Enhanced Job Postings Over Time - Top 5 Companies", fontsize=14, fontweight="bold", pad=15, loc="left")
    ax.set_xlabel("Week", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Postings", fontsize=11, labelpad=10)

    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor="linecolor")
    style_week_axis(ax)
    plt.xticks(ha="right")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")
    ax.tick_params(colors="#666666")

    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CCCCCC")
    ax.grid(axis="x", linestyle=":", alpha=0.2, color="#CCCCCC")

    # Expand right margin so company name annotations don't get clipped
    plt.subplots_adjust(right=0.78)

    fig.tight_layout()
    return fig