from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class CompletenessResult:
    score: float
    ready_for_recommendation: bool
    missing_fields: List[str]

# Intent completeness weights (must sum to 1.0)
SLOT_WEIGHTS = {
    "role": 0.35,
    "seniority": 0.20,
    "technical_skills": 0.25,
    "soft_skills": 0.10,
    "assessment_preferences": 0.10,
}

def compute_completeness(intent: Dict[str, Any], threshold: float = 0.70) -> CompletenessResult:
    """
    Compute weighted intent completeness score.
    """
    score = 0.0
    missing = []

    # Role (35%)
    if intent.get("role"):
        score += SLOT_WEIGHTS["role"]
    else:
        missing.append("role")

    # Seniority (20%)
    if intent.get("seniority"):
        score += SLOT_WEIGHTS["seniority"]
    else:
        missing.append("seniority")

    # Technical Skills (25%)
    if intent.get("technical_skills"):
        score += SLOT_WEIGHTS["technical_skills"]
    elif not intent.get("soft_skills"):
        # If no technical skills, we really want at least some skills
        missing.append("skills")

    # Soft Skills (10%)
    if intent.get("soft_skills"):
        score += SLOT_WEIGHTS["soft_skills"]

    # Preferences (10%)
    if intent.get("assessment_preferences") or intent.get("constraints"):
        score += SLOT_WEIGHTS["assessment_preferences"]

    final_score = round(score, 2)
    ready = final_score >= threshold

    logger.debug(f"Completeness: score={final_score}, ready={ready}, missing={missing}")
    
    return CompletenessResult(
        score=final_score,
        ready_for_recommendation=ready,
        missing_fields=missing
    )

def get_targeted_missing_desc(missing_fields: List[str]) -> str:
    """
    Map internal field names to user-friendly descriptions for clarification.
    """
    mapping = {
        "role": "the specific job role or position",
        "seniority": "the seniority level (e.g., entry-level, mid, or senior)",
        "skills": "key technical or behavioral skills you want to assess",
    }
    return "; ".join(mapping.get(f, f) for f in missing_fields)
