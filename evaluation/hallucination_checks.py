import json
import logging
from typing import List, Dict, Any
from app.services.catalog_loader import load_catalog

logger = logging.getLogger(__name__)

def check_hallucinations(chat_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check for hallucinations in a list of chat responses.
    Verified against the official catalog.
    """
    catalog = load_catalog()
    catalog_names = {item["name"].lower() for item in catalog}
    catalog_urls = {item["url"].lower() for item in catalog}
    
    total_recs = 0
    hallucinated_names = 0
    hallucinated_urls = 0
    
    for resp in chat_responses:
        recs = resp.get("recommendations", [])
        total_recs += len(recs)
        
        for rec in recs:
            name = rec.get("name", "").lower()
            url = rec.get("url", "").lower()
            
            if name not in catalog_names:
                hallucinated_names += 1
                logger.warning(f"Hallucinated name: {name}")
                
            if url not in catalog_urls:
                hallucinated_urls += 1
                logger.warning(f"Hallucinated URL: {url}")
                
    return {
        "total_recommendations": total_recs,
        "hallucinated_names": hallucinated_names,
        "hallucinated_urls": hallucinated_urls,
        "hallucination_rate": (hallucinated_names + hallucinated_urls) / (2 * total_recs) if total_recs > 0 else 0.0
    }

if __name__ == "__main__":
    # Example usage with trace logs
    pass
