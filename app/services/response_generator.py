import logging
from typing import List
from app.schemas.chat import Message, Recommendation, ChatResponse
from app.services import recommendation_engine
from app.services.state_machine import ConversationStage
from app.services.completeness import compute_completeness, get_targeted_missing_desc
from app.services.recommendation_validator import validate_recommendations
from app.services.diversity import balance_recommendations
from app.services.fallbacks import safe_schema_response, handle_llm_failure
from app.services.llm import call_llm

logger = logging.getLogger(__name__)

# System Prompts (same as before)
CLARIFICATION_SYSTEM_PROMPT = """You are a concise, professional SHL assessment consultant.
Your ONLY task is to ask ONE targeted clarification question to better understand the hiring need.

RULES:
- Ask exactly ONE question. Do NOT ask multiple questions in one message.
- Be direct and professional.
- Do NOT recommend any assessments yet.
- Do NOT explain what SHL is.
- Base the question on what information is MISSING from the context provided.

MISSING INFORMATION: {missing_slots}
CURRENT CONTEXT: {context_summary}"""

RECOMMENDATION_SYSTEM_PROMPT = """You are an expert SHL assessment consultant.
Based on the hiring context and the selected assessments, write a brief, professional explanation
of why these assessments are recommended.

RULES:
- Reference ONLY the assessments listed below (do NOT mention others)
- Be concise: 2-3 sentences maximum for the overall explanation
- Do NOT make up capabilities or features not mentioned in the assessment data
- Do NOT recommend assessments not in the list
- End with a natural closing (e.g., "Would you like to refine these?")

HIRING CONTEXT: {context_summary}
SELECTED ASSESSMENTS:
{assessments_summary}"""

REFINE_SYSTEM_PROMPT = """You are an expert SHL assessment consultant.
The user has refined their requirements. Here is the updated shortlist.

RULES:
- Briefly acknowledge the update
- Present the refined shortlist naturally
- Be concise (1-2 sentences)

CONTEXT: {context_summary}
UPDATED ASSESSMENTS: {assessments_summary}"""


def _build_context_summary(intent: dict) -> str:
    """Build a one-line context summary from intent."""
    parts = []
    if intent.get("role"):
        parts.append(intent["role"])
    if intent.get("seniority"):
        parts.append(f"({intent['seniority']} level)")
    if intent.get("industry"):
        parts.append(f"in {intent['industry']}")
    if intent.get("technical_skills"):
        parts.append(f"— tech: {', '.join(intent['technical_skills'][:3])}")
    if intent.get("soft_skills"):
        parts.append(f"— soft: {', '.join(intent['soft_skills'][:3])}")
    return " ".join(parts) if parts else "general hiring"


def _format_assessments_for_prompt(recommendations: List[Recommendation]) -> str:
    """Format recommendation list for LLM prompts."""
    lines = []
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec.name} (Type: {rec.test_type}) — {rec.url}")
    return "\n".join(lines)


def safe_response(reply: str = "I encountered an issue. Please try again.") -> ChatResponse:
    """Legacy wrapper for backward compatibility."""
    return safe_schema_response(reply=reply)


async def generate_clarification_response(
    intent: dict,
    messages: List[Message],
) -> ChatResponse:
    """Generate a targeted clarification question."""
    comp = compute_completeness(intent)
    missing_desc = get_targeted_missing_desc(comp.missing_fields) or "additional hiring details"
    context_summary = _build_context_summary(intent)

    try:
        reply = await call_llm(
            system_prompt=CLARIFICATION_SYSTEM_PROMPT.format(
                missing_slots=missing_desc,
                context_summary=context_summary or "No context yet",
            ),
            user_message="Ask ONE clarification question now.",
            temperature=0.2,
            max_output_tokens=150,
        )
        reply = reply.strip()
    except Exception as e:
        reply = await handle_llm_failure(e, "clarify")

    return safe_schema_response(reply=reply)


async def generate_recommendation_response(
    intent: dict,
    stage: ConversationStage,
) -> ChatResponse:
    """Generate a recommendation response with 1-10 assessments."""
    try:
        recommendations = await recommendation_engine.generate_recommendations(intent)

        if not recommendations:
            return safe_schema_response(
                reply="I wasn't able to find matching assessments for your specific requirements. Could you provide more details?"
            )

        # 1. Hallucination Hard Lock: Validate against catalog
        rec_dicts = [r.model_dump() for r in recommendations]
        validated = validate_recommendations(rec_dicts)
        
        # 2. Diversity Balancing
        balanced = balance_recommendations(validated)
        
        recommendations = [Recommendation(**r) for r in balanced]

        if not recommendations:
            return safe_schema_response(reply="I encountered an issue retrieving assessments. Please try again.")

        # 3. Generate natural language explanation
        context_summary = _build_context_summary(intent)
        assessments_summary = _format_assessments_for_prompt(recommendations)

        try:
            prompt_template = REFINE_SYSTEM_PROMPT if stage == ConversationStage.REFINE else RECOMMENDATION_SYSTEM_PROMPT
            reply = await call_llm(
                system_prompt=prompt_template.format(
                    context_summary=context_summary,
                    assessments_summary=assessments_summary,
                ),
                user_message="Generate the recommendation response now.",
                temperature=0.2,
                max_output_tokens=300,
            )
            reply = reply.strip()
        except Exception as e:
            reply = await handle_llm_failure(e, "recommend")

        return safe_schema_response(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=True
        )

    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        return safe_schema_response()


async def generate_comparison_response(
    intent: dict,
    messages: List[Message],
) -> ChatResponse:
    """Generate a grounded comparison response."""
    targets = intent.get("comparison_targets", [])
    user_question = messages[-1].content if messages else ""

    try:
        comparison_text = await comparison_engine.generate_comparison(
            comparison_targets=targets,
            user_question=user_question,
        )
    except Exception as e:
        comparison_text = await handle_llm_failure(e, "compare")

    return safe_schema_response(reply=comparison_text)


async def generate_refusal_response(refusal_message: str) -> ChatResponse:
    """Generate a safe refusal response (guardrails)."""
    return safe_schema_response(reply=refusal_message)
