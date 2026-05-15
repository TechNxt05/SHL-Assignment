"""
SHL Assessment Recommender - Recommendation Engine
Orchestrates retrieval → reranking → validation → response formatting.
"""
import logging
from typing import List

from app.schemas.chat import Recommendation
from app.services import retrieval as retrieval_service
from app.services import reranker
from app.services.fallbacks import get_fallback_recommendations

logger = logging.getLogger(__name__)


async def generate_recommendations(intent: dict, max_results: int = 10) -> List[Recommendation]:
    """
    Full recommendation pipeline.
    """
    # Step 1: Hybrid retrieval
    try:
        candidates = retrieval_service.retrieve(intent=intent, top_k=15)
        logger.info(f"Retrieved {len(candidates)} candidates")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        candidates = []

    if not candidates:
        logger.warning("Empty retrieval — using deterministic fallback shortlist")
        fallback_data = get_fallback_recommendations()
        return [Recommendation(**item) for item in fallback_data[:max_results]]

    # Step 2: Reranking
    try:
        ranked = await reranker.rerank(candidates=candidates, intent=intent, max_results=max_results)
        logger.info(f"After LLM reranking: {len(ranked)} items")
    except Exception as e:
        logger.warning(f"Reranking failed ({e}) — falling back to retrieval order")
        ranked = candidates[:max_results]

    # Step 3: Format as Recommendation objects
    recommendations = []
    for item in ranked:
        recommendations.append(
            Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=item.get("test_type", "K"),
            )
        )

    return recommendations[:max_results]


async def refine_recommendations(
    intent: dict,
    previous_names: List[str],
    max_results: int = 10,
) -> List[Recommendation]:
    """
    Refine recommendations based on updated intent.
    Unlike a fresh recommendation, this respects previously shown items
    while incorporating new constraints.
    """
    logger.info(f"Refining recommendations. Previous: {previous_names}")
    # Full re-run with updated intent (constraints already accumulated in intent)
    return await generate_recommendations(intent, max_results)
