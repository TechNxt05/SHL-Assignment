"""
SHL Assessment Recommender - LLM Reranker
Re-ranks retrieval candidates using LLM-based semantic scoring.
Also applies diversity balancing to avoid homogeneous shortlists.
"""
import logging
from collections import defaultdict
from typing import List

from app.services.llm import call_llm

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are an SHL assessment selection expert. 
Your task is to rank the following candidate assessments for a hiring scenario.

HIRING CONTEXT:
{intent_summary}

CANDIDATE ASSESSMENTS (names only, in no particular order):
{candidates}

RANKING INSTRUCTIONS:
1. Rank assessments by relevance to the hiring context
2. Prioritize diversity: do not cluster similar test types together unless specifically requested  
3. Consider: role match, skill coverage, seniority fit, test type balance
4. Output ONLY a Python list of assessment names, most relevant first
5. Include ONLY names from the candidate list — do NOT add new names
6. Include between 1 and 10 names

OUTPUT FORMAT (Python list, one per line, no commentary):
["Assessment Name 1", "Assessment Name 2", ...]"""


def _build_intent_summary(intent: dict) -> str:
    """Build a human-readable summary of the hiring intent."""
    parts = []
    if intent.get("role"):
        parts.append(f"Role: {intent['role']}")
    if intent.get("seniority"):
        parts.append(f"Seniority: {intent['seniority']}")
    if intent.get("industry"):
        parts.append(f"Industry: {intent['industry']}")
    if intent.get("technical_skills"):
        parts.append(f"Technical skills needed: {', '.join(intent['technical_skills'])}")
    if intent.get("soft_skills"):
        parts.append(f"Soft skills needed: {', '.join(intent['soft_skills'])}")
    if intent.get("assessment_preferences"):
        parts.append(f"Assessment type preference: {', '.join(intent['assessment_preferences'])}")
    if intent.get("constraints"):
        parts.append(f"Constraints: {', '.join(intent['constraints'])}")
    return "\n".join(parts) if parts else "General hiring assessment needed"


def _diversity_balance(candidates: List[dict], max_per_type: int = 3) -> List[dict]:
    """
    Ensure the shortlist has diverse test types.
    No more than max_per_type assessments of the same test_type.
    """
    type_counts: dict = defaultdict(int)
    balanced = []

    for item in candidates:
        t = item.get("test_type", "K")
        if type_counts[t] < max_per_type:
            balanced.append(item)
            type_counts[t] += 1
        if len(balanced) == 10:
            break

    return balanced


def _parse_llm_ranking(llm_output: str, valid_names: set) -> List[str]:
    """
    Parse the LLM's ranking output (Python list format) safely.
    Falls back to original order if parsing fails.
    """
    import ast
    import re

    # Try to extract a list literal from the output
    list_match = re.search(r"\[.*?\]", llm_output, re.DOTALL)
    if list_match:
        try:
            parsed = ast.literal_eval(list_match.group())
            if isinstance(parsed, list):
                # Filter to only valid candidate names
                return [name for name in parsed if name in valid_names]
        except (ValueError, SyntaxError):
            pass

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]+)"', llm_output)
    if quoted:
        return [name for name in quoted if name in valid_names]

    return []


async def rerank(
    candidates: List[dict],
    intent: dict,
    max_results: int = 10,
) -> List[dict]:
    """
    LLM-based reranking of retrieval candidates.
    Falls back to retrieval order if LLM reranking fails.

    Args:
        candidates: List of assessment dicts from retrieval pipeline
        intent: Structured intent from intent parser
        max_results: Maximum number of results to return (1-10)

    Returns:
        Reranked and diversity-balanced list of assessment dicts
    """
    if not candidates:
        return []

    if len(candidates) <= 3:
        # No need to rerank small lists
        return _diversity_balance(candidates[:max_results])

    valid_names = {item["name"] for item in candidates}
    name_to_item = {item["name"]: item for item in candidates}

    intent_summary = _build_intent_summary(intent)
    candidates_str = "\n".join(f"- {item['name']}" for item in candidates)

    prompt = RERANK_PROMPT.format(
        intent_summary=intent_summary,
        candidates=candidates_str,
    )

    try:
        llm_output = await call_llm(
            system_prompt=(
                "You are a precise assessment selection expert. "
                "Output ONLY a Python list. No commentary."
            ),
            user_message=prompt,
            temperature=0.1,
        )

        ranked_names = _parse_llm_ranking(llm_output, valid_names)

        if ranked_names:
            # Reconstruct ordered list from names
            reranked = [name_to_item[name] for name in ranked_names if name in name_to_item]
            # Append any candidates not in the ranked output
            reranked_set = set(ranked_names)
            for item in candidates:
                if item["name"] not in reranked_set:
                    reranked.append(item)

            logger.debug(f"LLM reranked {len(reranked)} candidates")
            return _diversity_balance(reranked[:max_results])

    except Exception as e:
        logger.warning(f"LLM reranking failed, using retrieval order: {e}")

    # Fallback: retrieval order + diversity balancing
    return _diversity_balance(candidates[:max_results])
