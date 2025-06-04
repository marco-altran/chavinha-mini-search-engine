FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and Poetry in one layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry

# Copy dependency files
COPY pyproject.toml ./

# Configure Poetry and install dependencies with optimizations
RUN poetry config virtualenvs.create false \
    && poetry config cache-dir /tmp/poetry-cache \
    && poetry install --only main --no-interaction --no-ansi --no-root \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers \
    && rm -rf /tmp/poetry-cache \
    && pip cache purge

# Copy application code
COPY api/ ./api/

# Copy Vespa certificates
COPY certs/ /app/certs/

# Create models directory
RUN mkdir -p api/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
