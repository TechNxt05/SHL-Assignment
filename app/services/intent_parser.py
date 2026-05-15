"""
SHL Assessment Recommender - Intent Parser
Extracts structured hiring intent from conversation history.
Computes completeness score to determine if clarification is needed.
"""
import json
import logging
import re
from typing import List

from app.schemas.chat import Message
from app.services.llm import call_llm_json
from app.services.completeness import compute_completeness

logger = logging.getLogger(__name__)

# Intent completeness scoring is handled in completeness.py

INTENT_SYSTEM_PROMPT = """You are a precise intent extraction engine for an SHL assessment recommender system.

Your task: Analyze the conversation history and extract structured hiring intent as JSON.

OUTPUT FORMAT (strict JSON, no markdown, no commentary):
{
  "role": "exact job title or role name, empty string if unknown",
  "seniority": "one of: entry, junior, mid, senior, manager, director, executive, or empty string",
  "technical_skills": ["list of specific technical skills, tools, languages mentioned"],
  "soft_skills": ["list of soft/behavioral skills mentioned"],
  "assessment_preferences": ["list of assessment types preferred: cognitive, personality, knowledge, simulation, behavioral"],
  "constraints": ["list of hard constraints: e.g. remote testing required, specific languages, time limits"],
  "comparison_targets": ["list of assessment names if user is asking to compare specific assessments"],
  "industry": "industry sector if mentioned, empty string otherwise",
  "is_comparison_request": false,
  "is_refinement": false
}

RULES:
- Extract ONLY what is explicitly stated or strongly implied in the conversation
- Do NOT invent or assume information not present
- Accumulate context across ALL turns (user + assistant)
- If the user says "actually, add X" or "also include Y", set is_refinement to true
- If the user asks to compare specific assessments, set is_comparison_request to true and populate comparison_targets
- technical_skills: programming languages, frameworks, tools (Java, Python, SQL, etc.)
- soft_skills: communication, leadership, teamwork, problem-solving, etc.
- assessment_preferences: ONLY if user explicitly mentions a type preference

FEW-SHOT EXAMPLES:

Example 1:
User: "I'm hiring a Java developer"
Output: {"role": "Java Developer", "seniority": "", "technical_skills": ["Java"], "soft_skills": [], "assessment_preferences": [], "constraints": [], "comparison_targets": [], "industry": "", "is_comparison_request": false, "is_refinement": false}

Example 2 (multi-turn):
User: "Hiring a sales manager for our FMCG team"
Assistant: "What seniority level?"
User: "Senior, around 8 years experience. Also needs strong leadership skills"
Output: {"role": "Sales Manager", "seniority": "senior", "technical_skills": [], "soft_skills": ["leadership"], "assessment_preferences": [], "constraints": [], "comparison_targets": [], "industry": "FMCG", "is_comparison_request": false, "is_refinement": false}

Example 3 (comparison):
User: "What's the difference between OPQ and GSA?"
Output: {"role": "", "seniority": "", "technical_skills": [], "soft_skills": [], "assessment_preferences": [], "constraints": [], "comparison_targets": ["OPQ", "GSA"], "industry": "", "is_comparison_request": true, "is_refinement": false}

Example 4 (refinement):
[Previous recommendations given]
User: "Actually, can you also add personality tests to the mix?"
Output: {"role": "...", "seniority": "...", ..., "assessment_preferences": ["personality"], "is_refinement": true}"""


def _format_conversation(messages: List[Message]) -> str:
    """Format conversation history for the LLM prompt."""
    formatted = []
    for msg in messages:
        prefix = "User" if msg.role == "user" else "Assistant"
        formatted.append(f"{prefix}: {msg.content}")
    return "\n".join(formatted)


def _validate_and_clean_intent(raw: dict) -> dict:
    """Validate and sanitize the LLM's JSON output."""
    defaults = {
        "role": "",
        "seniority": "",
        "technical_skills": [],
        "soft_skills": [],
        "assessment_preferences": [],
        "constraints": [],
        "comparison_targets": [],
        "industry": "",
        "is_comparison_request": False,
        "is_refinement": False,
    }

    for key, default in defaults.items():
        if key not in raw:
            raw[key] = default
        elif isinstance(default, list) and not isinstance(raw[key], list):
            raw[key] = [raw[key]] if raw[key] else []
        elif isinstance(default, bool) and not isinstance(raw[key], bool):
            raw[key] = bool(raw[key])
        elif isinstance(default, str) and not isinstance(raw[key], str):
            raw[key] = str(raw[key]) if raw[key] else ""

    return raw


async def parse_intent(messages: List[Message]) -> dict:
    """
    Parse the full conversation history into structured intent.

    Returns:
        Intent dict with completeness_score field added.
    """
    conversation_text = _format_conversation(messages)

    # Quick heuristic checks before calling LLM
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_user_msg = msg.content.lower()
            break

    # Fast-path comparison detection
    comparison_keywords = ["compare", "difference between", "vs ", " versus ", "what is the diff"]
    is_likely_comparison = any(kw in last_user_msg for kw in comparison_keywords)

    try:
        raw_json = await call_llm_json(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_message=f"CONVERSATION:\n{conversation_text}\n\nExtract intent as JSON:",
            temperature=0.05,  # Very low for deterministic extraction
        )

        # Parse JSON
        intent = json.loads(raw_json)
        intent = _validate_and_clean_intent(intent)

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Intent parsing failed: {e}. Using heuristic fallback.")
        intent = _heuristic_intent_extraction(messages)

    # Add completeness score
    completeness = compute_completeness(intent)
    intent["completeness_score"] = completeness.score
    intent["ready_for_recommendation"] = completeness.ready_for_recommendation

    # Override comparison flag if heuristics detected it
    if is_likely_comparison and not intent.get("is_comparison_request"):
        intent["is_comparison_request"] = True

    logger.debug(f"Intent: {intent}")
    return intent


def _heuristic_intent_extraction(messages: List[Message]) -> dict:
    """
    Rule-based fallback intent extraction when LLM fails.
    Ensures the system never crashes.
    """
    all_text = " ".join(m.content for m in messages if m.role == "user").lower()

    # Role detection
    role_patterns = [
        r"hiring (?:a |an )?(.+?)(?:\s+who|\s+with|\s+for|$)",
        r"looking for (?:a |an )?(.+?)(?:\s+who|\s+with|\s+for|$)",
        r"need (?:a |an )?(.+?)(?:\s+who|\s+with|\s+for|$)",
        r"(?:a |an )?(.+?) (?:developer|engineer|manager|analyst|designer)",
    ]

    role = ""
    for pattern in role_patterns:
        match = re.search(pattern, all_text)
        if match:
            role = match.group(1).strip().title()
            break

    # Seniority detection
    seniority = ""
    seniority_map = {
        "junior": "junior", "entry": "entry", "mid": "mid", "senior": "senior",
        "manager": "manager", "director": "director", "lead": "senior",
        "experienced": "mid", "graduate": "entry",
    }
    for kw, level in seniority_map.items():
        if kw in all_text:
            seniority = level
            break

    # Technical skills
    tech_skills = re.findall(
        r"\b(java|python|sql|javascript|react|node|aws|c\+\+|golang|rust|"
        r"kubernetes|docker|machine learning|data science|excel|powerbi)\b",
        all_text, re.I
    )

    # Soft skills
    soft_skills = re.findall(
        r"\b(communication|leadership|teamwork|problem.solving|analytical|"
        r"creative|organizational|negotiation|presentation)\b",
        all_text, re.I
    )

    return {
        "role": role,
        "seniority": seniority,
        "technical_skills": list(set(tech_skills)),
        "soft_skills": list(set(soft_skills)),
        "assessment_preferences": [],
        "constraints": [],
        "comparison_targets": [],
        "industry": "",
        "is_comparison_request": False,
        "is_refinement": False,
    }
