"""
SHL Assessment Recommender - FastAPI Application
Startup, middleware, and app configuration.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure structured logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (load catalog, build indexes) and shutdown cleanup.
    """
    logger.info("Starting SHL Assessment Recommender...")

    try:
        # 1. Load catalog
        from app.services.catalog_loader import load_catalog
        catalog = load_catalog()
        logger.info(f"Catalog loaded: {len(catalog)} assessments")

        # 2. Initialize retrieval (BM25 + FAISS)
        from app.services.retrieval import initialize_retrieval
        from app.services.embeddings import get_model
        initialize_retrieval()
        get_model()  # WARM UP: Load model into RAM now
        logger.info("Retrieval indexes initialized and model warmed up")

        # 3. Verify Gemini API key is set
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — LLM calls will fail!")
        else:
            logger.info("Gemini API key configured")

        logger.info("SHL Recommender is ready")
        yield

    except FileNotFoundError as e:
        logger.error(f"Startup failed — catalog not found: {e}")
        logger.error("Run: python -m app.services.scraper")
        # Still yield so /health can respond (allows cold-start probes)
        yield
    except Exception as e:
        logger.exception(f"Startup error: {e}")
        yield

    finally:
        logger.info("Shutting down SHL Recommender")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SHL Assessment Recommender",
        description=(
            "Conversational agent for recommending SHL Individual Test Solutions. "
            "Stateless API: send full conversation history with each request."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow all origins for evaluation harness compatibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    from app.api.routes import router
    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
