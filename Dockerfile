# ============================================================
# Stage 1: Builder — install dependencies and download model
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model to bake it into image
# This avoids cold-start download delays on deployment
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

# Copy application source
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY data/ ./data/

# Create data directories if they don't exist
RUN mkdir -p data/raw data/processed data/faiss

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
