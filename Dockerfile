# ============================================================
# Stage 1: Builder — install dependencies and pre-build indexes
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Set cache directories for the builder
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

# Install build tools for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code needed for index building
COPY app/ ./app/
COPY data/ ./data/
COPY utils/ ./utils/

# Pre-download the model AND pre-build the FAISS index
RUN python -m utils.setup_indexes

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Set cache directories for runtime
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source, PRE-BUILT DATA, and CACHED MODELS
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY utils/ ./utils/
COPY --from=builder /app/data/ ./data/
COPY --from=builder /app/.cache/ ./ .cache/

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p data/raw data/processed data/faiss .cache && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
