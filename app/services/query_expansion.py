from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Domain mapping for skill expansion
SKILL_EXPANSION_MAP = {
    "java": ["backend", "spring", "oop", "jvm", "apis", "microservices"],
    "python": ["backend", "django", "flask", "fastapi", "data analysis", "scripting"],
    "sql": ["database", "postgresql", "mysql", "data querying", "rdbms"],
    "javascript": ["frontend", "react", "node.js", "typescript", "web development"],
    "leadership": ["management", "communication", "stakeholder management", "team leading"],
    "communication": ["interpersonal", "writing", "collaboration", "soft skills"],
    "sales": ["negotiation", "relationship management", "account management", "influencing"],
}

def expand_query(intent: Dict[str, Any]) -> str:
    """
    Expand intent into a rich search string for hybrid retrieval.
    """
    base_terms = []
    
    # Add role
    if intent.get("role"):
        base_terms.append(intent["role"])
        
    # Add technical skills + expansions
    tech_skills = intent.get("technical_skills", [])
    for skill in tech_skills:
        base_terms.append(skill)
        expansion = SKILL_EXPANSION_MAP.get(skill.lower(), [])
        base_terms.extend(expansion)
        
    # Add soft skills
    soft_skills = intent.get("soft_skills", [])
    for skill in soft_skills:
        base_terms.append(skill)
        expansion = SKILL_EXPANSION_MAP.get(skill.lower(), [])
        base_terms.extend(expansion)
        
    # Add seniority context
    if intent.get("seniority"):
        base_terms.append(intent["seniority"])
        
    # Deduplicate and join
    expanded_query = " ".join(list(set(base_terms)))
    logger.debug(f"Expanded Query: {expanded_query}")
    return expanded_query
