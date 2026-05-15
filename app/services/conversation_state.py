"""
SHL Assessment Recommender - Conversation State Machine
Determines the current state of the conversation given intent and history.

States: GUARD_BLOCKED, CLARIFY, RECOMMEND, REFINE, COMPARE, END
"""
import logging
from enum import Enum
from typing import List, Optional

from app.schemas.chat import Message

logger = logging.getLogger(__name__)

COMPLETENESS_THRESHOLD = 0.60
MAX_TURNS = 8  # Assignment hard limit


class ConversationState(Enum):
    GUARD_BLOCKED = "guard_blocked"
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    END = "end"


def count_turns(messages: List[Message]) -> int:
    """Count total turns (user + assistant) in the conversation."""
    return len(messages)


def has_prior_recommendations(messages: List[Message]) -> bool:
    """
    Check if the assistant has already provided recommendations.
    Looks for the presence of recommendation content in assistant messages.
    We use a heuristic: if any assistant message contains a URL pattern.
    """
    import re
    url_pattern = re.compile(r"https?://www\.shl\.com/\S+")
    for msg in messages:
        if msg.role == "assistant" and url_pattern.search(msg.content):
            return True
    return False


def determine_state(
    intent: dict,
    messages: List[Message],
    guardrail_blocked: bool = False,
    prior_recommendations: Optional[List] = None,
) -> ConversationState:
    """
    Determine what the agent should do next.

    Decision tree:
    1. If guardrail blocked → GUARD_BLOCKED
    2. If comparison request → COMPARE
    3. If refinement with prior recs → REFINE
    4. If approaching turn limit (turn >= 7) → RECOMMEND (force answer)
    5. If completeness score >= threshold → RECOMMEND
    6. Else → CLARIFY
    """
    if guardrail_blocked:
        logger.debug("State: GUARD_BLOCKED")
        return ConversationState.GUARD_BLOCKED

    if intent.get("is_comparison_request"):
        logger.debug("State: COMPARE")
        return ConversationState.COMPARE

    turn_count = count_turns(messages)
    has_prior_recs = (
        prior_recommendations is not None and len(prior_recommendations) > 0
    ) or has_prior_recommendations(messages)

    # If user is refining after receiving recommendations
    if intent.get("is_refinement") and has_prior_recs:
        logger.debug("State: REFINE")
        return ConversationState.REFINE

    # Force recommendation if approaching turn limit (prevent hitting 8-turn cap without answer)
    if turn_count >= MAX_TURNS - 1:
        logger.info(f"Forcing RECOMMEND: turn_count={turn_count}, approaching limit")
        return ConversationState.RECOMMEND

    # Recommend if we have enough context
    completeness = intent.get("completeness_score", 0.0)
    if completeness >= COMPLETENESS_THRESHOLD:
        logger.debug(f"State: RECOMMEND (completeness={completeness})")
        return ConversationState.RECOMMEND

    # Need more context
    logger.debug(f"State: CLARIFY (completeness={completeness})")
    return ConversationState.CLARIFY


def get_missing_slots(intent: dict) -> List[str]:
    """
    Identify which important intent slots are missing.
    Used by the clarification engine to formulate targeted questions.
    """
    missing = []

    if not intent.get("role"):
        missing.append("role")
    elif not intent.get("seniority"):
        missing.append("seniority")
    elif not intent.get("technical_skills") and not intent.get("soft_skills"):
        missing.append("skills")

    return missing
