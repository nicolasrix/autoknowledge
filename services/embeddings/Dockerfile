# OpenLLM embedding service
# Serves an embedding model via an OpenAI-compatible /v1/embeddings API on port 3000.
FROM python:3.11-slim

RUN pip install --no-cache-dir "openllm[embeddings]"

EXPOSE 3000

# Model is configurable via the EMBEDDING_MODEL build arg or OPENLLM_MODEL env var.
# Default: BAAI/bge-small-en-v1.5 (384-dim, fast, high quality for its size)
ENV OPENLLM_MODEL="BAAI/bge-small-en-v1.5"

# Models are cached at this path — mount a volume here to persist across restarts.
ENV HF_HOME="/models"
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/readyz')"

CMD ["sh", "-c", "openllm serve ${OPENLLM_MODEL} --port 3000"]
