# Ollama embedding service with AMD ROCm GPU support.
# Uses the official ollama/ollama:rocm image and pulls the embedding model on first start.
#
# For NVIDIA GPUs, change the base image to: ollama/ollama:latest
# For CPU-only,                               ollama/ollama:latest  (no device mapping needed)
FROM ollama/ollama:rocm

# Install curl for the startup health poll and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434

# Model is configurable via EMBEDDING_MODEL env var.
# Default: nomic-embed-text (768-dim, high quality, ~274 MB)
ENV EMBEDDING_MODEL="nomic-embed-text"

# Model weights are stored here — mount a volume to persist across restarts.
ENV OLLAMA_MODELS="/models"

ENTRYPOINT ["/start.sh"]
