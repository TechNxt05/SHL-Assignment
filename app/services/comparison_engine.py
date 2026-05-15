"""
SHL Assessment Recommender - Comparison Engine
Produces grounded, catalog-only comparisons of SHL assessments.
No external knowledge is used — all claims come from catalog metadata.
"""
import logging
from typing import List, Optional

from app.services.catalog_loader import fuzzy_name_lookup
from app.services.llm import call_llm

logger = logging.getLogger(__name__)

COMPARISON_SYSTEM_PROMPT = """You are an SHL assessment product expert.
Your task is to compare SHL assessments based ONLY on the provided catalog data.

CRITICAL RULES:
- Use ONLY the data provided below. Do NOT use external knowledge.
- Do NOT invent capabilities, durations, or features not in the data.
- If a field is unknown/missing, say "not specified in catalog".
- Be factual, concise, and professional.
- Structure your response as: Overview → Key Differences → When to Use Each

CATALOG DATA FOR COMPARISON:
{catalog_data}

USER QUESTION: {user_question}

Provide a grounded comparison based strictly on the above data."""


def _format_assessment_for_comparison(item: dict) -> str:
    """Format a catalog entry as a readable comparison block."""
    lines = [
        f"Assessment: {item.get('name', 'Unknown')}",
        f"URL: {item.get('url', 'N/A')}",
        f"Type: {item.get('test_type', 'N/A')} ({_type_label(item.get('test_type', ''))})",
        f"Duration: {item.get('duration_minutes', 'not specified')} minutes",
        f"Remote Testing: {'Yes' if item.get('remote_testing') else 'No/Not specified'}",
        f"Adaptive: {'Yes' if item.get('adaptive') else 'No/Not specified'}",
        f"Description: {item.get('description', 'N/A')[:300]}",
        f"Skills Measured: {', '.join(item.get('skills_measured', [])) or 'Not specified'}",
        f"Competencies: {', '.join(item.get('competencies', [])) or 'Not specified'}",
        f"Job Levels: {', '.join(item.get('job_levels', [])) or 'Not specified'}",
    ]
    return "\n".join(lines)


def _type_label(code: str) -> str:
    """Expand test type code to full label."""
    labels = {
        "A": "Ability/Cognitive",
        "B": "Biodata/Behavioral",
        "C": "Competency",
        "K": "Knowledge/Skills",
        "P": "Personality",
        "S": "Simulation",
    }
    return labels.get(code.upper(), "Unknown")


async def generate_comparison(
    comparison_targets: List[str],
    user_question: str,
) -> str:
    """
    Generate a grounded comparison of the specified assessments.
    """
    if not comparison_targets:
        return "Please specify which assessments you'd like to compare."

    # Look up each target in catalog
    found = []
    for target in comparison_targets:
        item = fuzzy_name_lookup(target)
        if item:
            found.append(item)

    if len(found) < 1:
        return (
            f"I couldn't find the assessments you mentioned in the SHL catalog. "
            "Please ensure you're using exact names from the official Individual Test Solutions."
        )

    # Format catalog data as a structured block for the LLM
    catalog_data = []
    for item in found:
        catalog_data.append(_format_assessment_for_comparison(item))
    
    catalog_text = "\n\n---\n\n".join(catalog_data)

    try:
        reply = await call_llm(
            system_prompt=COMPARISON_SYSTEM_PROMPT.format(
                catalog_data=catalog_text,
                user_question=user_question,
            ),
            user_message="Compare these assessments now based strictly on the catalog data.",
            temperature=0.0,
        )
        return reply.strip()
    except Exception as e:
        logger.error(f"Comparison LLM failed: {e}")
        return "I'm sorry, I couldn't generate a comparison at this moment. Please check the individual assessment details on the SHL website."


def _fallback_comparison(found: List[dict], not_found: List[str]) -> str:
    """Structured fallback comparison without LLM."""
    lines = ["Here is a comparison based on catalog data:\n"]
    for item in found:
        lines.append(f"**{item['name']}**")
        lines.append(f"- Type: {item.get('test_type', 'N/A')} ({_type_label(item.get('test_type', ''))})")
        lines.append(f"- Duration: {item.get('duration_minutes', 'not specified')} min")
        lines.append(f"- Remote: {'Yes' if item.get('remote_testing') else 'Not specified'}")
        lines.append(f"- Measures: {', '.join(item.get('skills_measured', [])[:3]) or 'see catalog'}")
        lines.append("")

    if not_found:
        lines.append(f"Note: '{', '.join(not_found)}' not found in catalog.")
    return "\n".join(lines)
