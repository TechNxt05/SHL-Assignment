from enum import Enum
from typing import Dict, Any
from app.config import (
    BM25_WEIGHT_TECHNICAL, SEMANTIC_WEIGHT_TECHNICAL,
    BM25_WEIGHT_VAGUE, SEMANTIC_WEIGHT_VAGUE
)

class QueryType(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    VAGUE = "vague"

def classify_query(intent: Dict[str, Any]) -> QueryType:
    """
    Classify query type to determine retrieval weights.
    """
    if intent.get("technical_skills"):
        return QueryType.TECHNICAL
    
    if intent.get("soft_skills") and not intent.get("role"):
        return QueryType.BEHAVIORAL
        
    if not intent.get("role") and not intent.get("technical_skills"):
        return QueryType.VAGUE
        
    return QueryType.TECHNICAL # Default to balanced/technical

def get_retrieval_weights(query_type: QueryType) -> Dict[str, float]:
    """
    Return adaptive weights for BM25 and Semantic search.
    """
    if query_type == QueryType.TECHNICAL:
        return {
            "bm25": BM25_WEIGHT_TECHNICAL,
            "semantic": SEMANTIC_WEIGHT_TECHNICAL
        }
    else:
        return {
            "bm25": BM25_WEIGHT_VAGUE,
            "semantic": SEMANTIC_WEIGHT_VAGUE
        }
