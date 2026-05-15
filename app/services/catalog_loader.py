"""
SHL Assessment Recommender - Catalog Loader
Loads and caches the catalog JSON for fast in-memory access.
Provides lookup by name, URL, and test_type.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CATALOG_CACHE: Optional[List[dict]] = None
_NAME_INDEX: Optional[Dict[str, dict]] = None
_URL_INDEX: Optional[Dict[str, dict]] = None


def load_catalog(path: Optional[str] = None) -> List[dict]:
    """
    Load catalog from disk (with in-memory cache).
    Falls back to environment variable CATALOG_PATH if path not provided.
    """
    global _CATALOG_CACHE, _NAME_INDEX, _URL_INDEX

    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    catalog_path = path or os.environ.get("CATALOG_PATH", "data/processed/catalog.json")

    if not Path(catalog_path).exists():
        logger.error(f"Catalog file not found: {catalog_path}")
        raise FileNotFoundError(
            f"Catalog not found at {catalog_path}. Run the scraper first: "
            "python -m app.services.scraper"
        )

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build indexes for fast lookup
    _NAME_INDEX = {}
    _URL_INDEX = {}
    for item in data:
        name_lower = item["name"].strip().lower()
        _NAME_INDEX[name_lower] = item
        _URL_INDEX[item["url"]] = item

    _CATALOG_CACHE = data
    logger.info(f"Loaded {len(data)} assessments from {catalog_path}")
    return data


def get_by_name(name: str) -> Optional[dict]:
    """Exact (case-insensitive) name lookup."""
    global _NAME_INDEX
    if _NAME_INDEX is None:
        load_catalog()
    return _NAME_INDEX.get(name.strip().lower())


def get_by_url(url: str) -> Optional[dict]:
    """URL-based lookup."""
    global _URL_INDEX
    if _URL_INDEX is None:
        load_catalog()
    return _URL_INDEX.get(url)


def fuzzy_name_lookup(name: str, threshold: float = 0.6) -> Optional[dict]:
    """
    Fuzzy name matching for comparison queries where user may use short names.
    Uses simple substring matching + token overlap scoring.
    """
    global _NAME_INDEX
    if _NAME_INDEX is None:
        load_catalog()

    name_lower = name.strip().lower()

    # Exact match first
    if name_lower in _NAME_INDEX:
        return _NAME_INDEX[name_lower]

    # Substring match
    for stored_name, item in _NAME_INDEX.items():
        if name_lower in stored_name or stored_name.startswith(name_lower):
            return item

    # Token overlap
    name_tokens = set(name_lower.split())
    best_score = 0.0
    best_item = None
    for stored_name, item in _NAME_INDEX.items():
        stored_tokens = set(stored_name.split())
        if not stored_tokens:
            continue
        overlap = len(name_tokens & stored_tokens) / max(len(name_tokens), len(stored_tokens))
        if overlap > best_score and overlap >= threshold:
            best_score = overlap
            best_item = item

    return best_item


def get_all_names() -> List[str]:
    """Return all assessment names for validation."""
    catalog = load_catalog()
    return [item["name"] for item in catalog]


def get_all_urls() -> List[str]:
    """Return all valid catalog URLs for hallucination checking."""
    catalog = load_catalog()
    return [item["url"] for item in catalog]


def reload_catalog() -> List[dict]:
    """Force reload from disk (clears cache)."""
    global _CATALOG_CACHE, _NAME_INDEX, _URL_INDEX
    _CATALOG_CACHE = None
    _NAME_INDEX = None
    _URL_INDEX = None
    return load_catalog()
