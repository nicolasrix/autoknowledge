#!/bin/bash
# Start Ollama, wait for it to be ready, pull the embedding model, then stay in foreground.

MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

# Start Ollama server in the background
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to start..."
until curl -sf http://localhost:11434/api/version > /dev/null 2>&1; do
  sleep 1
done
echo "Ollama started."

# Pull the model (no-op if already present in the volume)
echo "Ensuring model '${MODEL}' is available..."
ollama pull "${MODEL}"
echo "Model '${MODEL}' is ready."

# Re-foreground the server
wait $SERVER_PID
