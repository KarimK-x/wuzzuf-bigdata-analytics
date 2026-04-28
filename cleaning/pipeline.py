"""Data cleaning pipeline.

This pipeline adds consistent grouping columns for job titles so analytics can
aggregate similar titles (e.g., "Senior .Net Developer" and
"Mid-Level .Net Developer") under one role family.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SENIORITY_PATTERNS = [
    (re.compile(r"\b(intern|internship|trainee|fresh)\b", re.IGNORECASE), "Intern/Trainee"),
    (re.compile(r"\b(junior|jr)\b", re.IGNORECASE), "Junior"),
    (re.compile(r"\b(mid|middle|midlevel|mid-level)\b", re.IGNORECASE), "Mid"),
    (re.compile(r"\b(senior|sr)\b", re.IGNORECASE), "Senior"),
    (re.compile(r"\b(lead|principal|head)\b", re.IGNORECASE), "Lead"),
]


NORMALIZATION_RULES = [
    (re.compile(r"\(.*?\)"), " "),
    (re.compile(r"\bfront[\s\-]?end\b", re.IGNORECASE), "frontend"),
    (re.compile(r"\bback[\s\-]?end\b", re.IGNORECASE), "backend"),
    (re.compile(r"\bfull[\s\-]?stack\b", re.IGNORECASE), "fullstack"),
    (re.compile(r"\bdev[\s\-]?ops\b", re.IGNORECASE), "devops"),
    (re.compile(r"\bsite reliability engineer\b", re.IGNORECASE), "sre"),
    (re.compile(r"\.net", re.IGNORECASE), "dotnet"),
    (re.compile(r"\bc\+\+\b", re.IGNORECASE), "cpp"),
    (re.compile(r"\bteamleader\b", re.IGNORECASE), "team lead"),
    (re.compile(r"[/|,&]"), " "),
    (re.compile(r"[^a-zA-Z0-9\s]"), " "),
    (re.compile(r"\s+"), " "),
]


NOISE_TOKENS = {
    "remote",
    "remotely",
    "onsite",
    "on",
    "site",
    "hybrid",
    "part",
    "time",
    "contract",
    "based",
}


ROLE_PATTERNS = [
    (re.compile(r"\b(data scientist)\b", re.IGNORECASE), "Data Scientist"),
    (re.compile(r"\b(data analyst)\b", re.IGNORECASE), "Data Analyst"),
    (
        re.compile(
            r"\b(data engineer|etl|business intelligence|bi engineer|data visualization|data governance|data management architect|data analytics|analytics engineer)\b",
            re.IGNORECASE,
        ),
        "Data Engineer",
    ),
    (re.compile(r"\b(ai|machine learning|ml|llm|nlp|prompt engineer)\b", re.IGNORECASE), "AI/ML Engineer"),
    (re.compile(r"\b(devops|sre|platform engineer|aiops|private cloud|cloud operations)\b", re.IGNORECASE), "DevOps Engineer"),
    (re.compile(r"\b(frontend|ui ux|ui|ux)\b", re.IGNORECASE), "Frontend Developer"),
    (re.compile(r"\b(backend)\b", re.IGNORECASE), "Backend Developer"),
    (re.compile(r"\b(fullstack)\b", re.IGNORECASE), "Full Stack Developer"),
    (re.compile(r"\b(android|ios|flutter|react native|mobile app|mobile)\b", re.IGNORECASE), "Mobile Developer"),
    (
        re.compile(
            r"\b(qa|quality assurance|quality control|software quality control|software tester|test engineer|testing engineer|tester|test lead|qc engineer|automation tester|automation qa)\b",
            re.IGNORECASE,
        ),
        "QA Engineer",
    ),
    (
        re.compile(
            r"\b(help desk|helpdesk|helpdsek|service desk|technical support|it support|application support|support engineer|vendor support)\b",
            re.IGNORECASE,
        ),
        "Technical Support Engineer",
    ),
    (
        re.compile(
            r"\b(system administrator|system engineer|systems engineer|infrastructure|network engineer|network administrator|network admin|data center|noc|enterprise noc|database administrator|database administration|it associate|it professional service engineer|it system auditor|salesforce administrator|dynamics 365 administrator|splunk administrator|it specialist|it engineer)\b",
            re.IGNORECASE,
        ),
        "IT Infrastructure Engineer",
    ),
    (re.compile(r"\b(cyber|cybersecurity|security)\b", re.IGNORECASE), "Security Engineer"),
    (re.compile(r"\b(computer science computer engineering fresh graduates|fresh graduates?)\b", re.IGNORECASE), "Software Engineer"),
    (re.compile(r"\b(product owner|product manager|system owner|product system owner)\b", re.IGNORECASE), "Product Manager"),
    (re.compile(r"\b(project manager)\b", re.IGNORECASE), "Project Manager"),
    (re.compile(r"\b(scrum master)\b", re.IGNORECASE), "Scrum Master"),
    (re.compile(r"\b(business analyst|system analyst|process improvement|grc implementor)\b", re.IGNORECASE), "Business Analyst"),
    (re.compile(r"\b(odoo|sap|oracle|erp)\b", re.IGNORECASE), "ERP Specialist"),
    (
        re.compile(
            r"\b(solution architect|software architect|software technical lead|software engineering manager|software engineering lead|software developers|integration engineer|product engineer|software engineer|software developer|developer|programmer)\b",
            re.IGNORECASE,
        ),
        "Software Engineer",
    ),
]


BROAD_FALLBACK_PATTERNS = [
    (
        re.compile(
            r"\b(sales|telesales|account manager|business development|pre sales|presales|inside sales|outbound|lead generation|sales executive|sales specialist|sales development)\b",
            re.IGNORECASE,
        ),
        "Sales & Business Development",
    ),
    (re.compile(r"\b(marketing|media|content|ecommerce|e commerce|digital growth|seo)\b", re.IGNORECASE), "Marketing & Content"),
    (re.compile(r"\b(hr|human resources|recruitment|recruiter|talent)\b", re.IGNORECASE), "HR & Recruitment"),
    (
        re.compile(
            r"\b(data entry|document controller|coordinator|virtual assistant|admin|clerk|operations officer|office|wfm)\b",
            re.IGNORECASE,
        ),
        "Operations & Administration",
    ),
    (re.compile(r"\b(teacher|instructor|training)\b", re.IGNORECASE), "Education & Training"),
    (re.compile(r"\b(designer|design)\b", re.IGNORECASE), "Design & Creative"),
    (re.compile(r"\b(consultant|specialist)\b", re.IGNORECASE), "Consulting & Specialist"),
    (re.compile(r"\b(manager|director|chief|officer|head|team lead|lead)\b", re.IGNORECASE), "Management"),
    (re.compile(r"\b(engineer|architect|mechatronics|mechanical|hse|planning|low current)\b", re.IGNORECASE), "Other (Engineering)"),
]


def normalize_job_title(title: str) -> str:
    """Normalize a raw title into a comparable text form."""
    text = "" if pd.isna(title) else str(title)
    text = text.strip().lower()
    for pattern, replacement in NORMALIZATION_RULES:
        text = pattern.sub(replacement, text)
    tokens = [token for token in text.split() if token not in NOISE_TOKENS]
    return " ".join(tokens).strip()


def extract_seniority(title: str) -> str:
    """Extract a seniority label from a raw title."""
    text = "" if pd.isna(title) else str(title)
    for pattern, label in SENIORITY_PATTERNS:
        if pattern.search(text):
            return label
    return "Not Specified"


def canonicalize_title(clean_title: str) -> str:
    """Map normalized title text to a canonical role label."""
    if not clean_title:
        return "Unknown"
    for pattern, label in ROLE_PATTERNS:
        if pattern.search(clean_title):
            return label
    for pattern, label in BROAD_FALLBACK_PATTERNS:
        if pattern.search(clean_title):
            return label
    return "Other"


def add_job_title_groups(df: pd.DataFrame, title_column: str = "job_title") -> pd.DataFrame:
    """Add normalized and grouped title columns to a dataframe."""
    if title_column not in df.columns:
        raise KeyError(f"Column '{title_column}' does not exist in input dataframe")

    result = df.copy()
    result["job_title_clean"] = result[title_column].map(normalize_job_title)
    result["job_seniority"] = result[title_column].map(extract_seniority)
    result["job_title_group"] = result["job_title_clean"].map(canonicalize_title)
    return result


def run_pipeline(input_path: Path | None = None, output_path: Path | None = None) -> Path:
    """Run grouping step on a CSV and write an enriched CSV output."""
    base_dir = Path(__file__).resolve().parent
    resolved_input = input_path or (base_dir / "clean_data.csv")
    resolved_output = output_path or (base_dir / "clean_data_grouped.csv")

    df = pd.read_csv(resolved_input)
    grouped_df = add_job_title_groups(df, title_column="job_title")
    grouped_df.to_csv(resolved_output, index=False)
    return resolved_output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Group similar job titles in a CSV file.")
    parser.add_argument("--input", type=Path, default=None, help="Path to input CSV.")
    parser.add_argument("--output", type=Path, default=None, help="Path to output CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_file = run_pipeline(input_path=args.input, output_path=args.output)
    print(f"Saved grouped dataset to: {output_file}")
