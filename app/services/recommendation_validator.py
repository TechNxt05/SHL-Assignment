import logging
from typing import List, Dict, Any
from app.services.catalog_loader import load_catalog

logger = logging.getLogger(__name__)

# Load catalog once for validation
_catalog = load_catalog()
_valid_names = {item["name"].lower(): item for item in _catalog}

def validate_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strictly validate that recommended assessments exist in the catalog.
    Prevents LLM hallucinations of names or URLs.
    """
    validated = []
    seen_names = set()
    
    for rec in recommendations:
        name = rec.get("name", "").strip()
        if not name:
            continue
            
        # 1. Exact or normalized name match
        catalog_item = _valid_names.get(name.lower())
        
        if catalog_item:
            # 2. Hard Lock: Force catalog URL and exact name
            clean_rec = catalog_item.copy()
            
            # Ensure URL is safe
            if not clean_rec.get("url", "").startswith("https://www.shl.com/"):
                logger.warning(f"Discarding recommendation with non-SHL URL: {clean_rec.get('url')}")
                continue
                
            # Deduplicate
            if clean_rec["name"] not in seen_names:
                validated.append(clean_rec)
                seen_names.add(clean_rec["name"])
        else:
            logger.warning(f"Hallucination detected and blocked: '{name}'")
            
    logger.info(f"Validated {len(validated)}/{len(recommendations)} recommendations")
    return validated
