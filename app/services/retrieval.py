"""
SHL Assessment Recommender - Hybrid Retrieval Service
Implements BM25 + FAISS retrieval with Reciprocal Rank Fusion (RRF) merging
and metadata filtering.

Pipeline:
  Step 1: BM25 retrieval (lexical)
  Step 2: FAISS retrieval (semantic)
  Step 3: RRF merge
  Step 4: Metadata filtering (hard constraints)
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.services import embeddings as emb_service
from app.services.catalog_loader import load_catalog

logger = logging.getLogger(__name__)

# Module-level singletons, initialized at startup
_bm25: Optional[BM25Okapi] = None
_bm25_names: Optional[List[str]] = None
_faiss_index = None
_faiss_names: Optional[List[str]] = None
_catalog_lookup: Optional[Dict[str, dict]] = None

from app.services.query_expansion import expand_query
from app.services.query_classifier import classify_query, get_retrieval_weights
from app.config import RRF_K, RERANK_TOP_K

logger = logging.getLogger(__name__)

# Module-level singletons, initialized at startup
_bm25: Optional[BM25Okapi] = None
_bm25_names: Optional[List[str]] = None
_faiss_index = None
_faiss_names: Optional[List[str]] = None
_catalog_lookup: Optional[Dict[str, dict]] = None


def initialize_retrieval() -> None:
    """
    Initialize BM25 and FAISS indexes from the loaded catalog.
    Called at application startup.
    """
    global _bm25, _bm25_names, _faiss_index, _faiss_names, _catalog_lookup

    catalog = load_catalog()
    _catalog_lookup = {item["name"]: item for item in catalog}

    # Build BM25
    logger.info("Building BM25 index...")
    _bm25_names = [item["name"] for item in catalog]
    corpus = []
    for item in catalog:
        # Use enriched search_text
        doc = item.get("search_text", "")
        if not doc:
            doc = f"{item.get('name', '')} {item.get('description', '')}"
        tokens = doc.lower().split()
        corpus.append(tokens)

    _bm25 = BM25Okapi(corpus)
    logger.info(f"BM25 index built with {len(corpus)} documents")

    # Load or build FAISS
    try:
        _faiss_index, _faiss_names = emb_service.load_faiss_index()
    except FileNotFoundError:
        logger.info("FAISS index not found — building from catalog...")
        _faiss_index, _faiss_names = emb_service.build_faiss_index(catalog)


def _bm25_retrieve(query: str, top_k: int = 25) -> List[Tuple[str, float]]:
    """BM25 lexical retrieval. Returns (name, score) tuples."""
    if _bm25 is None:
        raise RuntimeError("Retrieval not initialized. Call initialize_retrieval() first.")

    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        if score > 0:
            results.append((_bm25_names[idx], float(score)))

    return results


def _faiss_retrieve(query: str, top_k: int = 25) -> List[Tuple[str, float]]:
    """FAISS semantic retrieval. Returns (name, score) tuples."""
    if _faiss_index is None:
        raise RuntimeError("Retrieval not initialized. Call initialize_retrieval() first.")
    return emb_service.faiss_search(query, _faiss_index, _faiss_names, top_k)


def _rrf_merge(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Weighted Reciprocal Rank Fusion across multiple ranked lists.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for i, ranked_list in enumerate(ranked_lists):
        weight = weights[i] if i < len(weights) else 1.0
        for rank, (name, _) in enumerate(ranked_list):
            rrf_scores[name] += weight * (1.0 / (k + rank + 1))

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


def _apply_metadata_filters(
    candidates: List[Tuple[str, float]],
    intent: dict,
) -> List[Tuple[str, float]]:
    """
    Apply hard metadata filters based on extracted intent.
    """
    if not _catalog_lookup:
        return candidates

    filtered = []
    for name, score in candidates:
        item = _catalog_lookup.get(name)
        if not item:
            continue

        # Hard filter: remote_testing if explicitly required
        if intent.get("remote_only"):
            if not item.get("remote_testing", True):
                continue

        # Negative constraints
        neg_cons = intent.get("negative_constraints", [])
        should_skip = False
        for con in neg_cons:
            if con.lower() in item.get("name", "").lower() or con.lower() in item.get("description", "").lower():
                should_skip = True
                break
        if should_skip:
            continue

        # Soft filter: test_type preference
        prefs = [p.lower() for p in intent.get("assessment_preferences", [])]
        if prefs:
            type_match = any(
                p in item.get("test_type", "").lower()
                for p in prefs
            )
            if type_match:
                score *= 1.3

        filtered.append((name, score))

    # Re-sort after potential boosts
    filtered.sort(key=lambda x: x[1], reverse=True)

    if not filtered:
        return candidates

    return filtered


def retrieve(
    intent: dict,
    top_k: int = RERANK_TOP_K,
    bm25_k: int = 30,
    faiss_k: int = 30,
) -> List[dict]:
    """
    Full adaptive hybrid retrieval pipeline.
    """
    # Step 1: Query Expansion
    expanded_query = expand_query(intent)

    # Step 2: Query Classification for weights
    q_type = classify_query(intent)
    weights_map = get_retrieval_weights(q_type)
    
    # Step 3: Branching Retrieval
    bm25_results = _bm25_retrieve(expanded_query, top_k=bm25_k)
    faiss_results = _faiss_retrieve(expanded_query, top_k=faiss_k)

    # Step 4: Weighted RRF merge
    weights = [weights_map["bm25"], weights_map["semantic"]]
    merged = _rrf_merge([bm25_results, faiss_results], weights=weights)

    # Step 5: Metadata filtering
    filtered = _apply_metadata_filters(merged, intent)

    # Return top_k as catalog dicts
    results = []
    for name, score in filtered[:top_k]:
        item = _catalog_lookup.get(name)
        if item:
            results.append(item)

    logger.debug(
        f"Retrieval ({q_type}): BM25={len(bm25_results)}, "
        f"FAISS={len(faiss_results)}, Final={len(results)}"
    )
    return results
