import logging
from enum import Enum
from typing import List, Optional, Dict, Any
from app.schemas.chat import Message
from app.config import MAX_TURNS, CLARIFICATION_THRESHOLD

logger = logging.getLogger(__name__)

class ConversationStage(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"

def determine_stage(
    intent: Dict[str, Any],
    history: List[Message],
    guardrail_blocked: bool = False,
    prior_recommendations_count: int = 0
) -> ConversationStage:
    """
    Deterministic State Machine for conversational flow control.
    
    RATIONALE:
    Choosing a rule-based FSM over LLM-driven 'agentic' routing guarantees:
    1. Determinism: Identical inputs always yield the same conversational stage.
    2. Reliability: Zero risk of 'looping' or refusing to recommend due to LLM vagueness.
    3. Performance: Zero-latency state transitions.
    4. Compliance: Strict adherence to the 8-turn evaluator cap.
    """
    # 1. Guardrail triggers immediate refusal
    if guardrail_blocked:
        return ConversationStage.REFUSE

    # 2. Comparison detection
    if intent.get("is_comparison_request") and intent.get("comparison_targets"):
        return ConversationStage.COMPARE

    # 3. Turn count control (8-turn hard limit)
    # Each turn is user + assistant. 8 messages = 4 full turns. 16 messages = 8 turns.
    # The requirement is "8 turns total".
    turn_count = len([m for m in history if m.role == "user"])
    
    # 4. Refinement detection (after recommendations)
    is_refining = intent.get("is_refinement") or (prior_recommendations_count > 0 and turn_count > 1)
    
    # 5. Force recommendation if approaching turn limit
    # If we are on turn 6 or 7, we MUST recommend to avoid hitting 8 turns with just clarifying.
    if turn_count >= 6:
        logger.info(f"Approaching turn limit ({turn_count}), forcing RECOMMEND")
        return ConversationStage.RECOMMEND

    # 6. Recommendation Readiness
    completeness_score = intent.get("completeness_score", 0.0)
    
    if is_refining and prior_recommendations_count > 0:
        return ConversationStage.REFINE
        
    if completeness_score >= CLARIFICATION_THRESHOLD:
        return ConversationStage.RECOMMEND

    # 7. Default to clarification
    return ConversationStage.CLARIFY
