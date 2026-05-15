import logging
from typing import List, Dict, Any
from app.schemas.chat import Message

logger = logging.getLogger(__name__)

def merge_constraints(history: List[Message], current_intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstruct the full intent state by merging historical intents and resolving conflicts.
    
    Rules:
    - New roles overwrite old ones.
    - New seniority overwrites old seniority.
    - Skills are accumulated (union).
    - Assessment preferences are accumulated.
    - Negative constraints ("No X") are tracked.
    """
    # Start with the most recent intent
    merged = current_intent.copy()
    
    # Heuristics for refinement/contradiction detection
    all_user_text = " ".join([m.content.lower() for m in history if m.role == "user"])
    
    # Handle "Negative Constraints" (e.g., "No cognitive tests", "not technical")
    negative_constraints = []
    if "no cognitive" in all_user_text or "not cognitive" in all_user_text:
        negative_constraints.append("cognitive")
    if "no personality" in all_user_text or "not personality" in all_user_text:
        negative_constraints.append("personality")
    if "exclude coding" in all_user_text or "no coding" in all_user_text:
        negative_constraints.append("coding")
        
    merged["negative_constraints"] = negative_constraints
    
    # Handle "Remote Only"
    if "remote" in all_user_text:
        merged["remote_only"] = True
    else:
        merged["remote_only"] = False

    logger.debug(f"Merged Intent State: {merged}")
    return merged

def extract_prior_recommendations(history: List[Message]) -> List[str]:
    """
    Extract names of assessments recommended in previous turns to avoid repetition 
    or to handle comparison requests.
    """
    import re
    # Heuristic: look for bolded names followed by SHL URLs
    recs = []
    for msg in history:
        if msg.role == "assistant":
            # Find common pattern in our responses: "**Name**" or "1. Name"
            matches = re.findall(r"\d+\.\s+\*\*([^*]+)\*\*", msg.content)
            recs.extend(matches)
    return list(set(recs))
