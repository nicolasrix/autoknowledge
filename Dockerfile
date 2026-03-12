# Stage 1: build deps in isolated layer for cache efficiency
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency manifest first (cache layer)
COPY pyproject.toml ./
COPY src/ ./src/

# Install production dependencies into a venv
RUN uv sync --no-dev

# Stage 2: minimal runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Container-friendly defaults (overridden by docker-compose env or user config)
ENV AUTOKNOWLEDGE_VAULT_PATH="/vault"
ENV AUTOKNOWLEDGE_DATA_DIR="/data"
ENV AUTOKNOWLEDGE_EMBEDDING_BASE_URL="http://embeddings:3000"

# vault = read-only bind mount from host; data = named volume for ChromaDB + SQLite
VOLUME ["/vault", "/data"]

ENTRYPOINT ["autoknowledge"]
CMD ["serve"]
