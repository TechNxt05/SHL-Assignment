"""
SHL Assessment Recommender - Embeddings Service
Singleton wrapper for sentence-transformers all-MiniLM-L6-v2.
Provides encode(), build_faiss_index(), and save/load utilities.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_DIR = Path(os.environ.get("FAISS_INDEX_PATH", "data/faiss/index.faiss")).parent
INDEX_PATH = FAISS_DIR / "index.faiss"
EMBEDDINGS_PATH = FAISS_DIR / "embeddings.npy"
NAMES_PATH = FAISS_DIR / "names.json"

_model = None  # Lazy singleton


def get_model():
    """Lazy-load the sentence transformer model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded")
    return _model


def encode(texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized float32 embeddings.
    Normalization enables cosine similarity via inner product in FAISS.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(catalog: List[dict]) -> tuple:
    """
    Build a FAISS IndexFlatIP (cosine similarity) from catalog entries.
    Documents are encoded from: name + description + skills + competencies.

    Returns:
        (index, names_list) where names_list[i] maps index position to assessment name
    """
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building FAISS index for {len(catalog)} assessments...")

    # Build document strings for embedding
    names = []
    documents = []
    for item in catalog:
        names.append(item["name"])
        doc = item.get("search_text", "")
        if not doc:
            # Fallback if search_text is missing
            doc = f"{item.get('name', '')} {item.get('description', '')}"
        documents.append(doc)

    embeddings = encode(documents)

    # Build FAISS index (inner product = cosine on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Persist
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(EMBEDDINGS_PATH), embeddings)

    import json
    with open(str(NAMES_PATH), "w") as f:
        json.dump(names, f)

    logger.info(f"FAISS index saved to {INDEX_PATH} ({index.ntotal} vectors)")
    return index, names


def load_faiss_index() -> tuple:
    """Load persisted FAISS index and names list."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Run build_faiss_index() first."
        )

    import json
    index = faiss.read_index(str(INDEX_PATH))
    with open(str(NAMES_PATH), "r") as f:
        names = json.load(f)

    logger.info(f"Loaded FAISS index: {index.ntotal} vectors")
    return index, names


def faiss_search(
    query: str,
    index: faiss.Index,
    names: List[str],
    top_k: int = 25,
) -> List[tuple]:
    """
    Search FAISS index for the top_k most similar documents.
    Returns list of (name, score) tuples.
    """
    query_vec = encode([query])
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(names):
            results.append((names[idx], float(score)))

    return results
