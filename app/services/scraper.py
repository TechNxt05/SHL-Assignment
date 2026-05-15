"""
SHL Catalog Scraper
Scrapes Individual Test Solutions from the SHL product catalog.
Stores normalized, deduplicated entries to data/processed/catalog.json.

IMPORTANT: Only "Individual Test Solutions" are scraped. Pre-packaged Job Solutions
are explicitly excluded as per assignment requirements.
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CATALOG_FILE = PROCESSED_DIR / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Map SHL test type labels to canonical single-letter codes
TEST_TYPE_MAP = {
    "ability": "A",
    "biodata": "B",
    "competencies": "C",
    "competency": "C",
    "knowledge": "K",
    "personality": "P",
    "simulation": "S",
    "skill": "K",
    "cognitive": "A",
    "behavioral": "B",
}


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_url(session: requests.Session, url: str) -> str:
    """Fetch a URL with retry logic. Returns raw HTML text."""
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def _cache_path(url: str) -> Path:
    """Generate a filesystem cache path for a URL."""
    safe_name = re.sub(r"[^\w\-_]", "_", url.replace(BASE_URL, ""))[:200]
    return RAW_DIR / f"{safe_name}.html"


def _fetch_with_cache(session: requests.Session, url: str) -> str:
    """Fetch URL using filesystem cache to avoid redundant requests."""
    cache_file = _cache_path(url)
    if cache_file.exists():
        logger.debug(f"Cache hit: {url}")
        return cache_file.read_text(encoding="utf-8")
    logger.debug(f"Fetching: {url}")
    html = _fetch_url(session, url)
    cache_file.write_text(html, encoding="utf-8")
    return html


def _clean_text(text: str) -> str:
    """Remove HTML artifacts, normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_duration(text: str) -> Optional[int]:
    """Extract numeric duration in minutes from text like '25 minutes' or '25-35 min'."""
    if not text:
        return None
    match = re.search(r"(\d+)(?:\s*[-–]\s*\d+)?\s*min", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_test_type_from_badges(soup: BeautifulSoup) -> str:
    """Extract test type from badge/icon elements on the detail page."""
    # Look for test type indicators
    for elem in soup.find_all(["span", "div", "p"], class_=re.compile(r"type|badge|tag|label|icon", re.I)):
        text = elem.get_text(strip=True).lower()
        for key, code in TEST_TYPE_MAP.items():
            if key in text:
                return code

    # Fallback: scan all text for type indicators
    page_text = soup.get_text(" ", strip=True).lower()
    for key, code in TEST_TYPE_MAP.items():
        if key in page_text:
            return code

    return "K"  # Default to knowledge test


def _extract_bool_feature(soup: BeautifulSoup, *keywords: str) -> bool:
    """Check if any of the keywords appear in the page in a positive context."""
    page_text = soup.get_text(" ", strip=True).lower()
    for kw in keywords:
        if kw.lower() in page_text:
            return True
    return False


def _extract_list_items(soup: BeautifulSoup, *label_patterns: str) -> List[str]:
    """Extract list items following a label element matching one of the patterns."""
    items = []
    for pattern in label_patterns:
        for header in soup.find_all(
            ["h3", "h4", "strong", "b", "dt", "th", "label"],
            string=re.compile(pattern, re.I),
        ):
            # Look in adjacent sibling or parent container
            container = header.find_next_sibling() or header.parent
            if container:
                for li in container.find_all(["li", "dd", "td", "span"]):
                    text = _clean_text(li.get_text())
                    if text and len(text) < 100:
                        items.append(text)
    return list(set(items))


def _build_keywords(assessment: dict) -> List[str]:
    """Build a flat keyword list from all searchable fields."""
    keywords = []
    keywords.append(assessment.get("name", "").lower())
    keywords.extend([s.lower() for s in assessment.get("skills_measured", [])])
    keywords.extend([c.lower() for c in assessment.get("competencies", [])])
    keywords.extend([j.lower() for j in assessment.get("job_levels", [])])
    # Extract individual words from description
    desc_words = re.findall(r"\b\w{4,}\b", assessment.get("description", "").lower())
    keywords.extend(desc_words[:50])
    return list(set(keywords))


def _scrape_detail_page(session: requests.Session, url: str, name: str) -> dict:
    """
    Scrape a single assessment detail page and extract metadata.
    """
    html = _fetch_with_cache(session, url)
    soup = BeautifulSoup(html, "lxml")

    # Initialize with defaults
    data = {
        "name": name,
        "url": url,
        "description": "",
        "test_type": "K",  # Default to Knowledge
        "duration_minutes": None,
        "remote_testing": True,
        "job_levels": [],
        "competencies": [],
        "skills_measured": [],
        "adaptive": False,
        "languages": ["English"],
    }

    # Helper to find text after any header label
    def get_para_after_header(label: str) -> str:
        header = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"], string=lambda x: x and label.lower() in x.lower())
        if header:
            # Try to find next paragraph or just the next sibling's text
            next_node = header.find_next(["p", "div", "span"])
            if next_node:
                return _clean_text(next_node.get_text())
        return ""

    # 1. Description
    data["description"] = get_para_after_header("Description")
    if not data["description"]:
        # Fallback: search all paragraphs for substantial text
        for p in soup.find_all("p"):
            text = _clean_text(p.get_text())
            if len(text) > 100:
                data["description"] = text
                break

    # 2. Job Levels
    levels_text = get_para_after_header("Job levels")
    if levels_text:
        data["job_levels"] = [l.strip() for l in levels_text.split(",") if l.strip()]

    # 3. Duration
    duration_text = get_para_after_header("Assessment length")
    if duration_text:
        # Format: "Approximate Completion Time in minutes = 36"
        match = re.search(r"minutes\s*=\s*(\d+)", duration_text, re.I)
        if match:
            data["duration_minutes"] = int(match.group(1))
        elif "untimed" in duration_text.lower():
            data["duration_minutes"] = 0
        else:
            # Fallback regex for just numbers
            num_match = re.search(r"(\d+)\s*min", duration_text, re.I)
            if num_match:
                data["duration_minutes"] = int(num_match.group(1))

    # 4. Test Type
    # Look for "Test Type:" label or header
    test_type_header = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"], string=lambda x: x and "test type" in x.lower())
    if test_type_header:
        type_text = _clean_text(test_type_header.find_next(string=True))
        if type_text and len(type_text) == 1 and type_text.upper() in "ABCKPS":
            data["test_type"] = type_text.upper()
    else:
        # Try finding string "Test Type:"
        label = soup.find(string=re.compile(r"Test Type:", re.I))
        if label:
            type_text = label.replace("Test Type:", "").strip()
            if type_text and len(type_text) == 1 and type_text.upper() in "ABCKPS":
                data["test_type"] = type_text.upper()

    # 5. Remote Testing
    remote_label = soup.find(string=re.compile(r"Remote Testing:", re.I))
    if remote_label:
        if "no" in remote_label.parent.get_text().lower():
            data["remote_testing"] = False

    # 6. Extract Skills/Competencies from Description
    if data["description"]:
        # Look for "measures..." or "assesses..."
        skills_match = re.search(r"(?:measures|assesses)\s+(.*?)(?:\.|$)", data["description"], re.I)
    # 7. Generate Enriched Search Text
    search_parts = [
        f"Assessment: {data['name']}",
        f"Description: {data['description']}",
        f"Measures: {', '.join(data['skills_measured'])}",
        f"Competencies: {', '.join(data['competencies'])}",
        f"Levels: {', '.join(data['job_levels'])}",
        f"Type: {data['test_type']}"
    ]
    data["search_text"] = " ".join(search_parts)

    return data


def _is_valid_assessment(data: dict) -> bool:
    """
    Validate that the scraped page is actually a specific assessment product.
    """
    # MUST have the specific view pattern in URL
    if "/view/" not in data["url"]:
        return False
        
    if not data["description"] or len(data["description"]) < 30:
        return False
        
    # Categories usually have vague descriptions
    vague_patterns = ["all our assessments", "explore our", "solutions for", "see more"]
    if any(p in data["description"].lower() for p in vague_patterns):
        return False
        
    # If it has job levels or duration, it's definitely a product
    if data["job_levels"] or data["duration_minutes"] is not None:
        return True
        
    # If it's linked from the catalog and has a substantial description, 
    # it's likely a skill/knowledge test even if the name is just ".NET"
    if len(data["description"]) > 100:
        return True
        
    # Fallback for short descriptions: check name
    keywords = ["test", "verify", "opq", "inventory", "assessment", "solution", "simulation", "short form"]
    if any(kw in data["name"].lower() for kw in keywords):
        return True
        
    return False


def _parse_catalog_listing(html: str) -> List[dict]:
    """
    Parse the catalog listing page using specific SHL catalog selectors.
    """
    soup = BeautifulSoup(html, "lxml")
    assessments = []
    seen_urls = set()

    # Priority 1: Specific row containers (browser inspection identified these)
    rows = soup.select(".product-catalog__row")
    
    links = []
    if rows:
        for row in rows:
            a_tag = row.select_one("a[href*='/products/product-catalog/view/']")
            if a_tag:
                links.append(a_tag)
    
    # Priority 2: Supplemental links in main content area
    # Looking for /view/ pattern in any link
    all_links = soup.find_all("a", href=True)
    for link in all_links:
        href = link.get("href", "")
        if "/products/product-catalog/view/" in href:
            links.append(link)

    for link in links:
        href = link.get("href", "")
        if not href:
            continue

        # Normalize URL
        if href.startswith("/"):
            href = urljoin(BASE_URL, href)
        
        # Strip trailing slash for consistency
        href = href.rstrip("/")
        
        # Skip category pages (usually don't have a specific product slug)
        if href.endswith("/view") or href.endswith("/view/"):
            continue

        if href in seen_urls:
            continue

        seen_urls.add(href)
        name = _clean_text(link.get_text())
        
        # If name is missing or generic (like 'View'), extract from slug
        if not name or len(name) < 3 or name.lower() in ["view", "learn more", "details"]:
            from urllib.parse import parse_qs
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            if "assessment" in params:
                name = params["assessment"][0].replace("-", " ").title()
            else:
                # Extract from path slug: .../view/SLUG
                parts = parsed.path.rstrip("/").split("/")
                if parts[-1] != "view":
                    name = parts[-1].replace("-", " ").title()
                else:
                    name = "Assessment"

        assessments.append({"name": name, "url": href})

    return assessments


def scrape_catalog(force_refresh: bool = False) -> List[dict]:
    """
    Main entry point: scrape the SHL catalog and return list of assessment dicts.
    Handles multi-page pagination.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if CATALOG_FILE.exists() and not force_refresh:
        logger.info(f"Loading cached catalog from {CATALOG_FILE}")
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    logger.info("Scraping SHL product catalog with pagination...")
    session = _make_session()
    
    all_stubs = []
    # SHL pagination uses ?start=X&type=1 where X is the offset (usually 0, 10, 20...)
    # Based on "32 pages", we might need to go up to ~310
    for offset in range(0, 350, 10):
        url = f"{CATALOG_URL}?start={offset}&type=1"
        try:
            html = _fetch_with_cache(session, url)
            stubs = _parse_catalog_listing(html)
            if not stubs:
                logger.info(f"No more assessments found at offset {offset}. Stopping.")
                break
            
            # Add only unique stubs
            new_count = 0
            for s in stubs:
                if s["url"] not in {existing["url"] for existing in all_stubs}:
                    all_stubs.append(s)
                    new_count += 1
            
            logger.info(f"Offset {offset}: found {len(stubs)} assessments ({new_count} new)")
            if new_count == 0: # Avoid infinite loops if page just returns same results
                break
                
            time.sleep(1) # Be nice
        except Exception as e:
            logger.error(f"Error fetching offset {offset}: {e}")
            break

    assessments = []
    for i, stub in enumerate(all_stubs):
        logger.info(f"[{i+1}/{len(all_stubs)}] Scraping details: {stub['name']}")
        try:
            assessment = _scrape_detail_page(session, stub["url"], stub["name"])
            if _is_valid_assessment(assessment):
                assessment["keywords"] = _build_keywords(assessment)
                assessments.append(assessment)
            else:
                logger.debug(f"Filtered out non-assessment page: {stub['name']}")
        except Exception as e:
            logger.error(f"Failed to scrape {stub['url']}: {e}")
        
        # Incremental save every 10 assessments
        if (i + 1) % 10 == 0:
            with open(CATALOG_FILE, "w", encoding="utf-8") as f:
                json.dump(assessments, f, indent=2, ensure_ascii=False)
            logger.info(f"Incremental save: {len(assessments)} assessments so far.")

        time.sleep(0.5)

    # Final save
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scraped {len(assessments)} total assessments")
    return assessments


def load_catalog_from_file(path: str = str(CATALOG_FILE)) -> List[dict]:
    """Load existing catalog JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    catalog = scrape_catalog(force_refresh=True)
    print(f"Total assessments scraped: {len(catalog)}")
    if catalog:
        print(f"Example: {catalog[0]['name']} — {catalog[0]['url']}")
