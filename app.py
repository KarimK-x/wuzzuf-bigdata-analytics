import streamlit as st
import pandas as pd
import json
import ast
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plot as pt

# Configuration
st.set_page_config(page_title="Jobs Browser", layout="wide", page_icon="💼")
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.max_open_warning': 0})

def safe_parse_skills(val):
    if pd.isna(val):
        return []
    try:
        if isinstance(val, str):
            if val.startswith('['):
                return ast.literal_eval(val)
        return []
    except:
        return []

@st.cache_data
def load_data():
    df = pd.read_csv('public/clean_data_grouped.csv')
    df['job_title'] = df['job_title'].fillna('Untitled Job')
    df['company_name'] = df['company_name'].fillna('Not Specified')
    df['location'] = df['location'].fillna('Not Specified')
    df['job_title_group'] = df['job_title_group'].fillna('Uncategorized')
    df['parsed_skills'] = df['skills_and_tools'].apply(safe_parse_skills)
    if 'career_level' in df.columns:
        df['parsed_levels'] = df['career_level'].apply(safe_parse_skills)
    else:
        df['parsed_levels'] = [[] for _ in range(len(df))]
    df['searchText'] = (df['job_title'] + ' ' + df['company_name'] + ' ' + df['parsed_skills'].astype(str) + ' ' + df['job_title_group']).str.lower()
    return df

@st.cache_data
def get_stats(df):
    total_jobs = len(df)
    
    cat_counts = df['job_title_group'].value_counts().head(30).to_dict()
    
    all_skills = [skill for skills in df['parsed_skills'] for skill in skills]
    skills_series = pd.Series(all_skills)
    top_skills = skills_series.value_counts().head(15).to_dict()
    
    loc_counts = df['location'].value_counts().head(10).to_dict()
    
    return {
        'totalJobs': total_jobs,
        'topGroups': list(cat_counts.items()),
        'topSkills': list(top_skills.items()),
        'topLocations': list(loc_counts.items())
    }


def main():
    with st.spinner("Loading data..."):
        df = load_data()
        
    stats = get_stats(df)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Jobs Browser", "Analytics Dashboard"])
    
    if page == "Home":
        st.title("Platform Overview")
        st.markdown("Discover the value in our dataset")
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs Available", f"{stats['totalJobs']:,}")
            
        with col2:
            st.subheader("Top Job Groups")
            for cat, count in stats['topGroups'][:3]:
                st.write(f"**{cat}**: {count}")
                
        with col3:
            st.subheader("Frequent Skills")
            skills = [s for s, c in stats['topSkills'][:6]]
            st.write(", ".join(skills))
            
        with col4:
            st.subheader("Top Locations")
            for loc, count in stats['topLocations'][:3]:
                display_loc = loc.split(',')[0]
                st.write(f"**{display_loc}**: {count}")

    elif page == "Jobs Browser":
        st.title("Jobs Browser")
        st.markdown(f"Explore {len(df):,} available positions")
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("Search (jobs, companies, skills...)", "")
        with col2:
            groups = ["All"] + sorted(df['job_title_group'].unique().tolist())
            selected_grp = st.selectbox("Job Group Filter", groups)
            
        filtered_df = df.copy()
        if selected_grp != "All":
            filtered_df = filtered_df[filtered_df['job_title_group'] == selected_grp]
        
        if search_term:
            filtered_df = filtered_df[filtered_df['searchText'].str.contains(search_term.lower(), na=False)]
            
        st.write(f"Showing **{len(filtered_df)}** jobs")
        
        if len(filtered_df) == 0:
            st.info("No jobs found with the current filters.")
        else:
            grouped = filtered_df.groupby('job_title_group')
            for group_name, group_df in sorted(grouped, key=lambda x: len(x[1]), reverse=True):
                with st.expander(f"**{group_name}** - {len(group_df)} jobs"):
                    for _, row in group_df.iterrows():
                        st.markdown(f"🔹 **{row['job_title']}** at **{row['company_name']}**")
                        st.caption(f" {row['location']}")
                        skills = ", ".join(row['parsed_skills'][:6])
                        if skills:
                            st.caption(f"Skills: {skills}")
                        st.divider()
                    
    elif page == "Analytics Dashboard":
        st.title("Analytics Dashboard")
        st.markdown("Deeper insights into the job market data")
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Skills Distribution")
            top_skills_df = pd.DataFrame(stats['topSkills'], columns=['Skill', 'Count']).head(10)
            fig_skills = px.pie(top_skills_df, values='Count', names='Skill', hole=0.4)
            st.plotly_chart(fig_skills, use_container_width=True)
            
        with col2:
            st.subheader("Top Job Groups")
            top_grp_df = pd.DataFrame(stats['topGroups'], columns=['Job Group', 'Count'])
            fig_cat = px.bar(top_grp_df, x='Job Group', y='Count', 
                            color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig_cat, use_container_width=True)
            
        st.subheader("Total Representation by Experience Level (All Jobs)")
        levels = [
            'Fresh Graduate', 'Junior', 'Mid-Level', 'Senior', 
            'Lead / Principal', 'Executive / Expert'
        ]
        df_grouped = pt.load_notebook_dataset()
        fig_exp = pt.plot_global_experience_distribution(df_grouped, levels)
        st.pyplot(fig_exp)

        st.subheader("Experience Level Composition per Job (All Jobs)")
        fig_stacked = pt.plot_stacked_levels_all_jobs(df_grouped, levels)
        st.pyplot(fig_stacked)
        
        st.subheader("Heatmap of Career Levels (All Jobs)")
        fig_heatmap = pt.plot_heatmap_all_jobs(df_grouped, levels)
        st.pyplot(fig_heatmap)

        st.subheader("Most Common Needed Jobs")
        fig_job_counts = pt.plot_job_counts(df)
        st.pyplot(fig_job_counts)

        st.subheader("Most Common Job per City")
        fig_job_per_city = pt.plot_top_job_per_city(df)
        st.pyplot(fig_job_per_city)
        
        st.subheader("Most Common Job per country")
        fig_job_per_country = pt.plot_top_job_per_country(df)
        st.pyplot(fig_job_per_country)

        st.subheader("Part time jobs by country and work setting")
        fig_part_time_jobs_by_country_and_work_setting = pt.plot_part_time_jobs_by_country_and_work_setting(df)
        st.pyplot(fig_part_time_jobs_by_country_and_work_setting)

        st.subheader("Part time jobs by city and work setting")
        fig_part_time_jobs_by_city_and_work_setting = pt.plot_part_time_jobs_by_city_and_work_setting(df)
        st.pyplot(fig_part_time_jobs_by_city_and_work_setting)

        st.subheader("Education level by job title")
        fig_education_level_by_job_title = pt.plot_education_level_by_job_title_group(df)
        st.pyplot(fig_education_level_by_job_title)
        
        st.subheader("Top skills")
        fig_top_skills = pt.plot_skills("public/Mohamed/skill_counts.csv")
        st.pyplot(fig_top_skills)

        st.subheader("Top skills for manager")
        fig_top_skills = pt.plot_skills("public/Mohamed/skill_manager.csv")
        st.pyplot(fig_top_skills)

        st.subheader("Top skills for each job")
        fig_top_skills_for_each_job = pt.plot_treemap("public/Mohamed/job_skill_tables.xlsx")
        st.plotly_chart(fig_top_skills_for_each_job, use_container_width=True)

        st.subheader("Number of job postings")
        fig_num_of_job_postings = pt.plot_num_of_job_postings(df)
        st.pyplot(fig_num_of_job_postings)

        st.subheader("Distribution of job postings across companies")
        fig_distribution_postings = pt.plot_histogram_distribution_postings(df)
        st.pyplot(fig_distribution_postings)

        st.subheader("Hiring rate by company")
        fig_hiring_rate = pt.plot_hiring_interval(df)
        st.pyplot(fig_hiring_rate)

        st.subheader("Hiring rate by company alt")
        fig_hiring_rate_alt = pt.plot_hiring_interval_colored(df)
        st.pyplot(fig_hiring_rate_alt)

        st.subheader("Job postings over time - Top 5 Companies - Stacked Area Chart")
        fig_top5 = pt.plot_top5(df)
        st.pyplot(fig_top5)

        st.subheader("Enhanced Job Postings Over Time - Top 5 Companies")
        fig_top5_enhanced = pt.plot_top5_enhanced(df)
        st.pyplot(fig_top5_enhanced)

if __name__ == "__main__":
    main()
