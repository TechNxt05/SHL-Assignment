import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CATALOG_PATH = PROCESSED_DATA_DIR / "catalog.json"
FAISS_DIR = DATA_DIR / "faiss"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
EMBEDDINGS_PATH = FAISS_DIR / "embeddings.npy"
NAMES_PATH = FAISS_DIR / "names.json"

# API Settings
PORT = int(os.getenv("PORT", 8000))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Conversation Constraints
MAX_TURNS = 8
CLARIFICATION_THRESHOLD = 0.70  # Higher for more robust data gathering
MAX_RECOMMENDATIONS = 10
MIN_RECOMMENDATIONS = 1

# Retrieval Settings
RRF_K = 60
BM25_WEIGHT_TECHNICAL = 0.7
SEMANTIC_WEIGHT_TECHNICAL = 0.3
BM25_WEIGHT_VAGUE = 0.3
SEMANTIC_WEIGHT_VAGUE = 0.7
RERANK_TOP_K = 15

# Diversity Settings
MAX_PER_CATEGORY = 3  # Max assessments of same Test Type (A, B, C, etc.)
MMR_LAMBDA = 0.5      # Diversity vs Relevance balance

# Timeout Settings
LLM_TIMEOUT = 15.0
TOTAL_REQUEST_TIMEOUT = 25.0

# LLM Models
PRIMARY_MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-1.5-flash"
