import logging
from typing import List, Dict, Any
from app.config import MAX_PER_CATEGORY

logger = logging.getLogger(__name__)

def balance_recommendations(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Implements a category-aware diversity filter.
    
    RATIONALE:
    A common failure mode in assessment RAG is 'knowledge-test saturation', where the top-k 
    semantic matches are all variants of the same technical test. This filter enforces 
    a Max-Per-Category constraint (MMR-inspired) to ensure a balanced candidate persona.
    """
    balanced = []
    category_counts = {}
    
    for rec in recommendations:
        # Use test_type as the primary diversity pivot
        category = rec.get("test_type", "K") 
        
        count = category_counts.get(category, 0)
        if count < MAX_PER_CATEGORY:
            balanced.append(rec)
            category_counts[category] = count + 1
        else:
            logger.info(f"Diversity: Suppressed {rec['name']} (Category {category} full)")
            
    # Audit log for observability
    logger.info(f"Diversity balancing complete. Final distribution: {category_counts}")
    return balanced
