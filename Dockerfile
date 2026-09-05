# Single image for both processes: the API and the Streamlit UI share the same
# dependencies, and the command is chosen at run time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the embedding model inside the image rather than the home directory.
    HF_HOME=/opt/models

WORKDIR /app

# Build tooling is needed for a few wheels; drop it from the final layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# Bake the embedding model into the image. Without this, the first request
# after every deploy pays a ~50MB download before it can answer.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY . .

# Run as a non-root user; the storage directory must be writable by it.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/storage/uploads /app/storage/faiss \
    && chown -R appuser:appuser /app /opt/models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Overridden by the UI service; see docker-compose.yml.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
