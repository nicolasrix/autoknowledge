# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**autoknowledge** is a local-first RAG MCP server that indexes an Obsidian vault (Markdown files with YAML front-matter and `[[wikilinks]]`), creates embeddings via a local OpenLLM instance, stores them in ChromaDB + SQLite, and exposes hybrid semantic+BM25 search through three MCP tools to Claude Code.

## Common Commands

```bash
# Install / sync dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_config.py -v

# Run tests with coverage
uv run pytest --cov=src/autoknowledge --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Type-check
uv run mypy src/

# Index the vault (local, with config.toml)
uv run autoknowledge index

# Start MCP server (stdio, for Claude Code)
uv run autoknowledge serve

# Force full reindex
uv run autoknowledge index --full
```

## Docker

```bash
# Copy and edit environment config
cp .env.example .env        # set VAULT_PATH

# Start the full stack (embeddings + app)
docker compose up -d

# Run indexing as a one-shot job
docker compose run --rm autoknowledge index

# MCP server via stdio (for Claude Code integration)
docker compose run --rm -T autoknowledge serve
```

**Claude Code MCP config** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "autoknowledge": {
      "command": "docker",
      "args": ["compose", "-f", "/path/to/autoknowledge/docker-compose.yml",
               "run", "--rm", "-T", "autoknowledge", "serve"]
    }
  }
}
```

## Architecture

Four layers:

1. **Vault layer** (`src/autoknowledge/vault/`) — scans `.md` files, parses YAML front-matter and `[[wikilinks]]`, chunks by heading/paragraph with a 400–800 token cap and 100-token overlap.

2. **Storage layer** (`src/autoknowledge/index/`) — three components:
   - `HashTracker` (SQLite): tracks `path → SHA-256` to enable incremental indexing
   - `ChromaStore` (ChromaDB): stores chunk embeddings + metadata; validates embedding model/dimension on startup
   - `BM25Store` (rank_bm25): rebuilt in-memory from ChromaDB chunk texts at startup

3. **Retrieval layer** (`src/autoknowledge/retrieval/`) — hybrid search: over-fetches from both ChromaDB (semantic) and BM25, min-max normalises scores, returns `alpha * semantic + (1-alpha) * bm25` ranked results.

4. **MCP server** (`src/autoknowledge/server/`) — exposes three tools:
   - `search_knowledge(query, top_k?, filters?)` — hybrid ranked results
   - `get_document(path)` — full Markdown of a document (path-traversal protected)
   - `reindex(scope?)` — incremental or full reindex

## Key Conventions

- Config loaded from `config.toml` (auto-discovered) then overridden by env vars: `AUTOKNOWLEDGE_VAULT_PATH`, `AUTOKNOWLEDGE_DATA_DIR`, `AUTOKNOWLEDGE_EMBEDDING_BASE_URL`.
- All dataclasses are frozen (immutable). New objects are created, never mutated in-place.
- Embedding client points at OpenLLM's OpenAI-compatible `/v1/embeddings` endpoint.
- ChromaDB collection metadata stores `embedding_model` + `embedding_dimension`; if config changes, server refuses to start and tells the user to `reindex --full`.
- BM25 store uses a `threading.RLock` — safe for concurrent MCP tool calls.
- Deleted vault files are detected on every index run (tracked paths − current paths).
