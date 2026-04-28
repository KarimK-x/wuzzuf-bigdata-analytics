import scrapy
import json
import re
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class WuzzufSpider(CrawlSpider):
    """
    CrawlSpider to extract job data from Wuzzuf.
    Navigates through job listings and pagination to parse detailed job attributes
    from the embedded initialStoreState JSON.
    """
    name = "mycrawler"
    allowed_domains = ["wuzzuf.net"]
    
    custom_settings = {
        'ROBOTSTXT_OBEY': False,
        # Maintain a delay to respect server resources and prevent IP rate-limiting
        'DOWNLOAD_DELAY': 1, 
    }

    start_urls = [
        "https://wuzzuf.net/search/jobs?filters%5Broles%5D%5B0%5D=IT%2FSoftware%20Development"
    ]

    rules = (
        # Rule 1: Extract individual job detail pages and parse them via 'parse_job'
        Rule(LinkExtractor(allow=r'/jobs/p/'), callback='parse_job', follow=False),
        
        # Rule 2: Follow pagination links to ensure full coverage of the search results
        Rule(LinkExtractor(allow=r'start=\d+'), follow=True),
    )

    def parse_job(self, response):
        """
        Parses the job detail page by locating the Redux-style initial state 
        embedded within a script tag.
        """
        html = response.text
        # Locate the JSON object assigned to Wuzzuf.initialStoreState using regex
        match = re.search(r'Wuzzuf\.initialStoreState\s*=\s*(\{.*?\});\s*\n', html, re.DOTALL)
        
        if not match:
            self.logger.warning(f"Could not find JSON state in {response.url}")
            return
            
        try:
            store = json.loads(match.group(1))
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON in {response.url}")
            return

        # Access normalized entities from the store
        jobs = store.get("entities", {}).get("job", {}).get("collection", {})
        companies = store.get("entities", {}).get("company", {}).get("collection", {})
        
        if not jobs:
            return
            
        # Detail pages typically contain exactly one job entity in the collection
        job_id = list(jobs.keys())[0]
        job = jobs.get(job_id, {})
        attrs = job.get("attributes", {})

        # Extraction: Company Data 
        company_root = (job.get("relationships") or {}).get("company") or {}
        company_id   = (company_root.get("data") or {}).get("id")
        company_data = companies.get(str(company_id), {}) if company_id else {}
        company_attrs = company_data.get("attributes", {})
        company_name  = company_attrs.get("name", "N/A")

        # Company Size
        size_data = company_data.get("relationships", {}).get("size", {}).get("data") or {}
        size_id = size_data.get("id")
        size_lookup = store.get("lookups", {}).get("companySize", {}).get("collection", {})
        company_size = size_lookup.get(str(size_id), {}).get("attributes", {}).get("size", "N/A") if size_id else "N/A"

        # Location Data 
        loc = attrs.get("location") or {}
        city = (loc.get("city") or {}).get("name", "")
        area = (loc.get("area") or {}).get("name", "")
        country = (loc.get("country") or {}).get("name", "")
        location = ", ".join(filter(None, [area, city, country])) or "N/A"

        # Extraction: Job Characteristics 
        work_types = [wt.get("displayedName", "") for wt in (attrs.get("workTypes") or [])]
        work_setting = (attrs.get("workplaceArrangement") or {}).get("displayedName", "N/A")

        # Extraction: Experience Requirements 
        exp = attrs.get("workExperienceYears") or {}
        exp_min = exp.get("min")
        exp_max = exp.get("max")
        
        if exp_min is None and exp_max is None:
            experience = "N/A"
        else:
            experience = [
                exp_min if exp_min is not None else 0, 
                exp_max if exp_max is not None else "above_min"
            ]

        # Extraction: Career and Education 
        career_level = attrs.get("careerLevel") or {}
        career_level_str = career_level.get("name", "N/A")
        career_hint = career_level.get("hint", "")
        if career_hint:
            career_level_str = f"{career_level_str} ({career_hint})"

        candidate_prefs = attrs.get("candidatePreferences") or {}
        education_level = candidate_prefs.get("educationLevel") or {}
        education = education_level.get("name", "N/A")

        # Extraction: Compensation
        salary_data = attrs.get("salary", {})
        if salary_data.get("hideSalary") or not salary_data.get("isPaid"):
            salary = "N/A"
        elif salary_data.get("min") and salary_data.get("max"):
            currency = salary_data.get("currency", {}).get("code", "")
            period = salary_data.get("period", {}).get("name", "")
            salary = f"{salary_data['min']} to {salary_data['max']} {currency} {period}".strip()
            extra = salary_data.get("additionalDetails", "")
            if extra:
                salary += f", {extra}"
        else:
            salary = "N/A"

        # --- Extraction: Skills and Metadata ---
        skills = [kw.get("name", "") for kw in (attrs.get("keywords") or [])]

        # Helper function to strip HTML tags from description fields
        def _clean_html(html_str):
            if not html_str: return ""
            return re.sub(r'<[^>]+>', ' ', html_str).strip()

        yield {
            "Job Title": attrs.get("title", "N/A"),
            "Company Name": company_name,
            "Company Size": company_size,
            "Location": location,
            "Work Type": ", ".join(work_types) if work_types else "N/A",
            "Work Setting": work_setting,
            "Experience Needed": experience,
            "Career Level": career_level_str,
            "Education Level": education,
            "Salary": salary,
            "Skills & Tools": skills,
            "Job Description": _clean_html(attrs.get("description", "")),
            "Job Requirements": _clean_html(attrs.get("requirements", "")),
            "Posted At": attrs.get("postedAt", ""),
            "url": response.url,
        }