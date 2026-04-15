import pandas as pd
import numpy as np
import json
import re

print("Loading dataset...")

with open(r"C:\Users\Radwa\Desktop\jobs_v2.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")


# Column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("&", "and", regex=False)
)


# Basic cleaning
exclude_cols = ["skills_and_tools", "experience_needed"]
text_cols = df.select_dtypes(include=["object", "str"]).columns
strip_cols = [c for c in text_cols if c not in exclude_cols]

for col in strip_cols:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace(["N/A", ""], np.nan)
    if col != "salary":
        df[col] = df[col].fillna("Unknown")


# Remove duplicates
df = df.drop_duplicates(subset="url").reset_index(drop=True)


# Fix Arabic locations
arabic_location_map = {
    "هليوبوليس, القاهرة, مصر": "Heliopolis, Cairo, Egypt",
    "القاهرة الجديدة, القاهرة, مصر": "New Cairo, Cairo, Egypt",
}
df["location"] = df["location"].replace(arabic_location_map)


# Company size
arabic_company_size = {
    "١١ - ٥٠ موظف": "11-50 employees",
    "٥٠١ - ١٠٠٠ موظف": "501-1000 employees",
}
df["company_size"] = df["company_size"].replace(arabic_company_size)

def parse_company_size(s):
    if pd.isna(s) or s == "Unknown":
        return np.nan, np.nan, np.nan

    s = str(s)

    more = re.search(r"more than\s+(\d+)", s, re.IGNORECASE)
    if more:
        v = int(more.group(1))
        return v, v, float(v)

    nums = re.findall(r"\d+", s)

    if len(nums) == 1:
        v = int(nums[0])
        return v, v, float(v)

    if len(nums) >= 2:
        lo, hi = int(nums[0]), int(nums[1])
        return lo, hi, (lo + hi) / 2

    return np.nan, np.nan, np.nan


df[["company_size_min", "company_size_max", "company_size_avg"]] = (
    df["company_size"].apply(parse_company_size).apply(pd.Series)
)

median_size = df["company_size_avg"].median()

df["company_size_avg"] = df["company_size_avg"].fillna(median_size)
df["company_size_min"] = df["company_size_min"].fillna(median_size)
df["company_size_max"] = df["company_size_max"].fillna(median_size)

df["company_size_avg"] = df["company_size_avg"].round().astype(int)
df["company_size_min"] = df["company_size_min"].round().astype(int)
df["company_size_max"] = df["company_size_max"].round().astype(int)


# Experience
def extract_experience(x):
    if isinstance(x, list) and len(x) >= 2:
        min_exp = x[0] if isinstance(x[0], (int, float)) else np.nan
        max_exp = x[1] if isinstance(x[1], (int, float)) else np.nan
        return min_exp, max_exp
    return np.nan, np.nan


df[["min_experience_years", "max_experience_years"]] = (
    df["experience_needed"].apply(extract_experience).apply(pd.Series)
)

df.drop(columns="experience_needed", inplace=True)

# Fill both min and max experience missing values
median_min_exp = df["min_experience_years"].median()
df["min_experience_years"] = df["min_experience_years"].fillna(median_min_exp)

median_max_exp = df["max_experience_years"].median()
df["max_experience_years"] = df["max_experience_years"].fillna(median_max_exp)


# Salary
def clean_salary(s):
    if pd.isna(s):
        return np.nan, np.nan, np.nan, np.nan

    s = re.sub(r",?\s*Bonus.*", "", str(s), flags=re.IGNORECASE)

    m = re.search(r"(\d[\d,]*)\s+to\s+(\d[\d,]*)\s+(\w+)\s+Per\s+(\w+)", s)
    if m:
        return (
            float(m.group(1).replace(",", "")),
            float(m.group(2).replace(",", "")),
            m.group(3).upper(),
            m.group(4).capitalize(),
        )

    return np.nan, np.nan, np.nan, np.nan


df[["salary_min", "salary_max", "salary_currency", "salary_period"]] = (
    df["salary"].apply(clean_salary).apply(pd.Series)
)

df.drop(columns="salary", inplace=True)

print(f"Salaries disclosed: {df['salary_min'].notna().sum()} / {len(df)}\n")


# Date
df["posted_at"] = pd.to_datetime(
    df["posted_at"],
    format="%m/%d/%Y %H:%M:%S",
    errors="coerce"
)


# Skills → list
df["skills_and_tools"] = df["skills_and_tools"].apply(
    lambda x: x if isinstance(x, list) else (
        [s.strip() for s in str(x).split(",") if s.strip()]
        if pd.notna(x) and x not in ["nan", ""]
        else []
    )
)


# Work type → list
df["work_type"] = df["work_type"].apply(
    lambda x: [s.strip() for s in str(x).split(",") if s.strip()]
    if pd.notna(x) and x not in ["Unknown", "nan"]
    else []
)

#  Translate Arabic values inside work_type list items
work_type_translations = {
    "دوام كامل": "Full Time",
    "دوام جزئي": "Part Time",
    "عمل من مقر الشركة": "On-Site",
    "عمل عن بعد": "Remote",
    "مستقل / مشروع": "Freelance / Project",
    "بناءً على الوردية": "Shift Based",
}

df["work_type"] = df["work_type"].apply(
    lambda lst: [work_type_translations.get(item, item) for item in lst]
)


# Categories
translations = {
    "دوام كامل": "Full Time",
    "عمل من مقر الشركة": "On-Site",
    "ذو خبرة (غير إداري)": "Experienced (Non-Manager)",
    "مستوى مبتدئ (مبتدئ / خريج جديد)": "Entry Level (Junior Level / Fresh Grad)",
    "غير محدد": "Not Specified",
    "درجة البكالوريوس": "Bachelor's Degree",
}

cat_cols = ["work_setting", "career_level", "education_level"]

for col in cat_cols:
    df[col] = df[col].replace(translations)
    df[col] = df[col].str.strip().str.title()
    df[col] = df[col].str.replace(
        r"'([A-Z])", lambda m: "'" + m.group(1).lower(), regex=True
    )

mode_setting = df[df["work_setting"] != "Unknown"]["work_setting"].mode()[0]
df["work_setting"] = df["work_setting"].replace("Unknown", mode_setting)


# Location split
def split_location(x):
    parts = [p.strip() for p in str(x).split(",")]
    country = parts[-1] if len(parts) >= 1 else np.nan
    city = parts[-2] if len(parts) >= 2 else np.nan
    return city, country


df[["city", "country"]] = df["location"].apply(split_location).apply(pd.Series)


# Final
print("Final shape:", df.shape)



# Save
df.to_csv("clean_data.csv", index=False, encoding="utf-8-sig")

df.to_json(
    "clean_data.json",
    orient="records",
    indent=4,
    force_ascii=False,
    date_format="iso"
)

print("\nDone")