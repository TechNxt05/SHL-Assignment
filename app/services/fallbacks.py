import logging
from typing import List, Dict, Any
from app.schemas.chat import ChatResponse

logger = logging.getLogger(__name__)

def safe_schema_response(
    reply: str = "I'm sorry, I'm having trouble processing that right now.",
    recommendations: List[Dict[str, Any]] = None,
    end_of_conversation: bool = False
) -> ChatResponse:
    """
    Ensure every response matches the ChatResponse schema, even under catastrophic failure.
    """
    return ChatResponse(
        reply=reply or "I encountered an issue. How can I help you?",
        recommendations=recommendations or [],
        end_of_conversation=end_of_conversation
    )

async def handle_llm_failure(error: Exception, stage: str) -> str:
    """
    Provide deterministic fallbacks when LLM fails (timeout/rate limit).
    """
    logger.warning(f"LLM failure in {stage}: {error}")
    
    fallbacks = {
        "clarify": "Could you tell me more about the role and seniority you're hiring for?",
        "recommend": "Based on our catalog, I've selected some relevant assessments for you.",
        "compare": "I'm unable to compare those specific assessments right now, but you can find detailed specs on the SHL website.",
        "refuse": "I can only assist with SHL assessment recommendations."
    }
    
    return fallbacks.get(stage, "How else can I help you find SHL assessments?")

def get_fallback_recommendations() -> List[Dict[str, Any]]:
    """
    Return a deterministic catalog-safe fallback shortlist.
    Used when retrieval fails or returns 0 results.
    """
    return [
        {
            "name": "OPQ32r (Occupational Personality Questionnaire)",
            "url": "https://www.shl.com/shl-test-catalog-details/occupational-personality-questionnaire-opq32r-v2/",
            "test_type": "P"
        },
        {
            "name": "GSA (General Ability Test)",
            "url": "https://www.shl.com/shl-test-catalog-details/verify-general-ability-test-gsa/",
            "test_type": "C"
        },
        {
            "name": "Verify Interactive - Inductive Reasoning",
            "url": "https://www.shl.com/shl-test-catalog-details/verify-interactive-inductive-reasoning/",
            "test_type": "C"
        }
    ]
