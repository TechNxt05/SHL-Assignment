"""
SHL Assessment Recommender - Guardrails Service
Two-layer safety enforcement:
  Layer 1: Fast pattern-based detection (pre-LLM)
  Layer 2: LLM-based classification for edge cases

Blocks: prompt injection, off-topic requests, competitor mentions,
        legal advice requests, and malicious instructions.
"""
import logging
import re
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    SAFE = "safe"
    PROMPT_INJECTION = "prompt_injection"
    OFF_TOPIC = "off_topic"
    COMPETITOR = "competitor"
    LEGAL = "legal"
    SYSTEM_PROBE = "system_probe"


# --- Pattern-based detection rules ---

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|all|your)\s+(instructions?|prompt|rules?|constraints?)",
    r"forget\s+(your\s+)?(system\s+)?(prompt|instructions?|role|context)",
    r"you\s+are\s+now\s+(a|an|DAN|GPT)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"\bDAN\b",
    r"jailbreak",
    r"bypass\s+(your\s+)?(restrictions?|rules?|filters?|safety)",
    r"act\s+as\s+(if\s+)?(you\s+(are|were|have\s+no))",
    r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompt)",
    r"new\s+instructions?:",
    r"system\s+prompt\s*(is|=|:)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"what\s+are\s+your\s+(hidden\s+)?(instructions?|rules?|prompts?)",
    r"override\s+(your\s+)?(safety|rules?|instructions?)",
]

OFF_TOPIC_PATTERNS = [
    r"employment\s+law",
    r"labor\s+law",
    r"discrimination\s+law",
    r"legal\s+advice",
    r"salary\s+(range|negotiation|benchmark)",
    r"visa\s+(application|process|sponsorship)",
    r"immigration",
    r"tax\s+(advice|filing|return)",
    r"how\s+to\s+fire",
    r"termination\s+(letter|process|policy)",
    r"stock\s+(option|market|trading)",
    r"investment\s+advice",
    r"medical\s+advice",
    r"political",
    r"weather",
    r"president",
    r"joke",
    r"capital\s+of",
    r"recipe",
    r"movie",
    r"song",
    r"game",
]

COMPETITOR_PATTERNS = [
    r"\bhackerrank\b",
    r"\bcodility\b",
    r"\btestgorilla\b",
    r"\bpymetrics\b",
    r"\btalentplus\b",
    r"\bkorn\s*ferry\b",
    r"\bpdri\b",
    r"\bhogan\b",
    r"\bpsi\s+(assessments?)?\b",
    r"\bpearson\s+(vue|assessments?)\b",
    r"\bmindtree\b",
    r"\bwondera\b",
    r"\bcriteria\s+(corp|assessments?)\b",
    r"\bevaltrak\b",
    r"\bimocha\b",
]

SYSTEM_PROBE_PATTERNS = [
    r"what\s+(model|llm|ai)\s+(are\s+you|do\s+you\s+use)",
    r"which\s+(model|llm)\s+(are|do)\s+you",
    r"(show|tell)\s+me\s+(your\s+)?(api\s+key|secret|token|credentials?)",
    r"what\s+is\s+your\s+(temperature|prompt|system\s+prompt|architecture)",
    r"how\s+(are\s+you\s+built|do\s+you\s+work\s+internally)",
    r"source\s+code",
]


def _check_patterns(text: str, patterns: list) -> bool:
    """Check if any pattern matches the input text (case-insensitive)."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def check_input(user_message: str) -> Tuple[bool, ThreatType, str]:
    """
    Check user input against guardrail rules.

    Returns:
        (is_blocked, threat_type, refusal_message)
        If is_blocked is False, proceed normally.
    """
    # Layer 1: Pattern matching (fast, no LLM)

    if _check_patterns(user_message, INJECTION_PATTERNS):
        logger.warning(f"Prompt injection detected: {user_message[:100]}")
        return (
            True,
            ThreatType.PROMPT_INJECTION,
            "I'm focused on helping you find the right SHL assessment. "
            "I can't respond to instructions that attempt to change my behavior. "
            "What role are you hiring for?",
        )

    if _check_patterns(user_message, LEGAL_PATTERNS):
        logger.info(f"Legal/off-topic query blocked: {user_message[:100]}")
        return (
            True,
            ThreatType.LEGAL,
            "I'm not able to provide legal advice. I specialize in recommending "
            "SHL psychometric assessments for hiring decisions. "
            "Would you like help finding the right assessment for your role?",
        )

    if _check_patterns(user_message, OFF_TOPIC_PATTERNS):
        logger.info(f"Off-topic query blocked: {user_message[:100]}")
        return (
            True,
            ThreatType.OFF_TOPIC,
            "That's outside my scope. I can only help with SHL assessment recommendations. "
            "What type of role are you assessing candidates for?",
        )

    if _check_patterns(user_message, COMPETITOR_PATTERNS):
        logger.info(f"Competitor mention blocked: {user_message[:100]}")
        return (
            True,
            ThreatType.COMPETITOR,
            "I only recommend SHL assessments from the official SHL catalog. "
            "I'm not able to recommend third-party assessment providers. "
            "Would you like to see SHL's offerings for your use case?",
        )

    if _check_patterns(user_message, SYSTEM_PROBE_PATTERNS):
        logger.info(f"System probe detected: {user_message[:100]}")
        return (
            True,
            ThreatType.SYSTEM_PROBE,
            "I'm an SHL assessment recommendation assistant. I can help you find "
            "the right assessment for your hiring needs. What role are you hiring for?",
        )

    return False, ThreatType.SAFE, ""


def validate_recommendations(recommendations: list, valid_urls: list) -> list:
    """
    Post-generation guardrail: ensure all recommended URLs are in the catalog.
    Filters out any hallucinated recommendations.
    """
    valid_url_set = set(valid_urls)
    filtered = []
    for rec in recommendations:
        if rec.get("url") in valid_url_set:
            filtered.append(rec)
        else:
            logger.warning(f"Hallucinated URL filtered out: {rec.get('url')}")
    return filtered


# Alias for clarity
LEGAL_PATTERNS = [
    r"employment\s+law",
    r"labor\s+law",
    r"discrimination\s+law",
    r"wrongful\s+termination",
    r"EEOC",
    r"ADA\s+compliance",
    r"GDPR\s+hiring",
    r"legal\s+advice",
    r"lawsuit",
    r"sue\s+(my\s+employer|them|the\s+company)",
]
