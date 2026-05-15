"""
Utility script to initialize FAISS and BM25 indexes after scraping.
"""
import os
import logging
from app.services.retrieval import initialize_retrieval
from app.services.catalog_loader import load_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing search indexes...")
    try:
        # Ensure catalog exists
        catalog = load_catalog()
        logger.info(f"Loaded {len(catalog)} assessments from catalog.")
        
        # Force build FAISS index
        from app.services import embeddings as emb_service
        emb_service.build_faiss_index(catalog)
        logger.info("Search indexes initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize indexes: {e}")

if __name__ == "__main__":
    main()
