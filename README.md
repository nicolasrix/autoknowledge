# autoknowledge

A local-first RAG (Retrieval-Augmented Generation) MCP server that indexes your Obsidian vault and gives Claude Code semantic search over your scientific knowledge base — entirely on your own machine, no cloud required.

## How it works

```
Obsidian vault (.md files)
        │
        ▼
  [Indexer pipeline]
  ├── Scan & parse YAML front-matter
  ├── Resolve [[wikilinks]]
  ├── Chunk by headings (400–800 tokens, 100-token overlap)
  ├── Embed via OpenLLM → ChromaDB (cosine similarity)
  └── Build BM25 keyword index (rank_bm25)
        │
        ▼
  [MCP Server]  ◄──── Claude Code
  ├── search_knowledge  (hybrid semantic + BM25)
  ├── get_document      (full file read)
  └── reindex           (incremental update)
```

Retrieval combines semantic similarity (ChromaDB) and keyword matching (BM25) using weighted score fusion. Only changed files are re-embedded on subsequent runs (SHA-256 hash tracking).

---

## Requirements

- Docker and Docker Compose (recommended), **or** Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- [OpenLLM](https://github.com/bentoml/OpenLLM) serving an embedding model (handled automatically by Docker Compose)
- `cifs-utils` on the Docker host **if your vault is on an SMB/NAS share** (see [SMB vault](#smb-vault-truenas--samba))

---

## Quickstart (Docker)

### 1. Clone and configure

```bash
git clone https://github.com/nicolasrix/autoknowledge
cd autoknowledge

cp .env.example .env
```

Edit `.env` for your vault location. Choose one:

**Local path:**
```env
VAULT_VOLUME=vault_local
VAULT_PATH=/home/yourname/obsidian-vault
```

**SMB share (TrueNAS, Samba, NAS):**
```env
SMB_HOST=truenas.local
SMB_USER=smbshare
SMB_PASSWORD=yourpassword
SMB_SHARE_PATH=Vault/Media/obsidian/truenas-obsidian-vault
```
The default `VAULT_VOLUME` is `vault_smb`, so no extra variable is needed for SMB.

### 2. Start the embedding service

```bash
docker compose up -d embeddings
```

The first start downloads the embedding model weights (~130 MB for `BAAI/bge-small-en-v1.5`). Watch readiness with:

```bash
docker compose logs -f embeddings
```

Wait until you see the service pass its health check (up to ~2 minutes on first run).

### 3. Index your vault

```bash
docker compose run --rm autoknowledge index
```

You will see progress printed to stderr:

```
Indexing 1/42: Attention Is All You Need.md
Indexing 2/42: BERT.md
...
Indexed:   42/42 files
Chunks:    1 847 total
```

### 4. Connect Claude Code

Add the MCP server to your Claude Code configuration (`~/.claude.json`):

```json
{
  "mcpServers": {
    "autoknowledge": {
      "command": "docker",
      "args": [
        "compose",
        "-f", "/absolute/path/to/autoknowledge/docker-compose.yml",
        "run", "--rm", "-T",
        "autoknowledge", "serve"
      ]
    }
  }
}
```

The `-T` flag disables pseudo-TTY so that stdio communication works correctly.

Restart Claude Code. You should see `autoknowledge` in the connected MCP servers list.

---

## Quickstart (local, no Docker)

### 1. Install dependencies

```bash
git clone https://github.com/nicolasrix/autoknowledge
cd autoknowledge
uv sync
```

### 2. Start OpenLLM

Install and run OpenLLM separately (outside this repo):

```bash
pip install openllm
openllm serve BAAI/bge-small-en-v1.5 --port 3000
```

### 3. Configure

```bash
cp config.toml.example config.toml
```

Edit `config.toml`:

```toml
[vault]
path = "~/obsidian-vault"   # your Obsidian vault

[embedding]
base_url = "http://localhost:3000"
model = "BAAI/bge-small-en-v1.5"
dimension = 384

[index]
data_dir = "~/.local/share/autoknowledge"
```

### 4. Index and serve

```bash
# Index the vault
uv run autoknowledge index

# Start MCP server (stdio)
uv run autoknowledge serve
```

### 5. Connect Claude Code

```json
{
  "mcpServers": {
    "autoknowledge": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/autoknowledge", "autoknowledge", "serve"]
    }
  }
}
```

---

## MCP Tools

Once connected, Claude Code can call three tools:

### `search_knowledge`

Hybrid semantic + keyword search over your vault.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language search query |
| `top_k` | int | 10 | Number of results to return |
| `filters` | object | null | Optional metadata filters (see below) |

**Filters:**

```json
{
  "tags": ["transformer", "attention"],   // must contain at least one tag
  "path_prefix": "papers/2024/",          // doc path must contain this prefix
  "title": "attention"                    // title must contain this string
}
```

**Example response:**

```markdown
## Result 1 (score: 0.912)
**Source**: `papers/attention-is-all-you-need.md`
**Section**: Architecture > Multi-Head Attention
**Tags**: transformer, attention, nlp

The multi-head attention mechanism allows the model to jointly attend to
information from different representation subspaces at different positions...

---
## Result 2 (score: 0.874)
...
```

### `get_document`

Returns the full raw Markdown of any document in your vault.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Path relative to vault root (e.g., `papers/bert.md`) |

Paths are validated to stay within the vault directory (no path traversal).

### `reindex`

Triggers an incremental reindex. Only changed files are re-embedded.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scope` | string | null | Optional subdirectory or file path relative to vault root |

Returns a summary:

```
Indexed:   3/42 files
Skipped:   39 unchanged
Deleted:   0 removed files
Chunks:    1 903 total
```

---

## Configuration reference

All settings live in `config.toml`. In Docker, `vault.path` and `index.data_dir` are set via environment variables and the file is optional.

```toml
[vault]
# Path to your Obsidian vault (local) or set AUTOKNOWLEDGE_VAULT_PATH (Docker)
path = "~/obsidian-vault"

[embedding]
# OpenLLM base URL — set AUTOKNOWLEDGE_EMBEDDING_BASE_URL in Docker
base_url = "http://localhost:3000"
# Must match the model loaded in OpenLLM and the stored index dimension.
# Changing this after indexing requires a full reindex (--full flag).
model = "BAAI/bge-small-en-v1.5"
dimension = 384
batch_size = 64        # chunks per embedding request

[index]
# Where ChromaDB and SQLite files are stored — set AUTOKNOWLEDGE_DATA_DIR in Docker
data_dir = "~/.local/share/autoknowledge"
chunk_min_tokens = 400
chunk_max_tokens = 800
chunk_overlap_tokens = 100   # context carried into the next chunk

[retrieval]
alpha = 0.7            # semantic weight (0 = pure BM25, 1 = pure semantic)
default_top_k = 10
```

### Environment variable overrides

These override the corresponding config.toml values and are used by Docker:

| Variable | Overrides |
|----------|-----------|
| `AUTOKNOWLEDGE_VAULT_PATH` | `vault.path` |
| `AUTOKNOWLEDGE_DATA_DIR` | `index.data_dir` |
| `AUTOKNOWLEDGE_EMBEDDING_BASE_URL` | `embedding.base_url` |
| `AUTOKNOWLEDGE_EMBEDDING_MODEL` | `embedding.model` |

---

## CLI reference

```
autoknowledge [--config PATH] [--verbose] <command>

Commands:
  index   Index or reindex the vault
  serve   Start the MCP server
```

### `autoknowledge index`

```
autoknowledge index [--full] [--path VAULT_PATH]

  --full          Drop and rebuild the entire index (use after changing
                  the embedding model or dimension)
  --path PATH     Override the vault path from config
```

### `autoknowledge serve`

```
autoknowledge serve [--transport {stdio,sse}] [--host HOST] [--port PORT]

  --transport     stdio (default, for Claude Code) or sse (for HTTP clients)
  --host          SSE bind address (default: 0.0.0.0)
  --port          SSE port (default: 8080)
```

---

## Vault format

autoknowledge works with standard Obsidian vaults. Front-matter is optional but recommended for filtering:

```markdown
---
title: "Attention Is All You Need"
authors: [Vaswani, Shazeer, Parmar]
year: 2017
tags: [transformer, attention, nlp]
---

# Introduction

The dominant sequence transduction models are based on complex recurrent
or convolutional neural networks...

## Background

See [[Bahdanau Attention]] for the original attention mechanism.
```

**Supported front-matter fields used for filtering:**

| Field | Type | Used in |
|-------|------|---------|
| `title` | string | `title` filter, result display |
| `tags` | list or string | `tags` filter, result display |

All other front-matter fields are stored as metadata and visible in search results.

**`[[wikilinks]]`** are extracted and stored as metadata. Aliases (`[[Target|Display]]`) are resolved to the target name.

Files in `.obsidian/` and `.trash/` directories are automatically excluded.

---

## Changing the embedding model

If you want to use a different model (e.g., `BAAI/bge-large-en-v1.5` for higher accuracy):

1. Update `config.toml` (or the `OPENLLM_MODEL` env var in `.env`):
   ```toml
   [embedding]
   model = "BAAI/bge-large-en-v1.5"
   dimension = 1024
   ```
2. Restart the embeddings container so OpenLLM loads the new model:
   ```bash
   docker compose restart embeddings
   ```
3. Force a full reindex to rebuild the vector store with the new dimensions:
   ```bash
   docker compose run --rm autoknowledge index --full
   ```

> The server validates the stored model name and dimension on startup and refuses to start if they don't match the config, with a clear error message pointing to `--full`.

---

## Keeping the index up to date

Re-run the indexer whenever you add or modify notes. Only changed files are re-embedded (incremental by default):

```bash
# Docker
docker compose run --rm autoknowledge index

# Local
uv run autoknowledge index
```

Or, while the MCP server is running, ask Claude Code to call `reindex` directly.

---

## Docker reference

```bash
# Start the embedding service in the background
docker compose up -d embeddings

# Index the vault (one-shot job)
docker compose run --rm autoknowledge index

# Force full reindex
docker compose run --rm autoknowledge index --full

# Run MCP server in SSE mode (long-running service)
docker compose up autoknowledge

# Run MCP server via stdio (used by Claude Code)
docker compose run --rm -T autoknowledge serve

# View embedding service logs
docker compose logs -f embeddings

# Stop everything
docker compose down
```

Named volumes:

| Volume | Contents |
|--------|----------|
| `autoknowledge_embeddings_models` | Downloaded model weights (persisted across restarts) |
| `autoknowledge_autoknowledge_data` | ChromaDB vectors + SQLite hash tracker |

---

## Development

```bash
uv sync --dev

# Run all tests
uv run pytest

# Single test file
uv run pytest tests/vault/test_chunker.py -v

# With coverage
uv run pytest --cov=src/autoknowledge --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Type-check
uv run mypy src/
```

### Project structure

```
src/autoknowledge/
  cli.py              CLI (index / serve subcommands)
  config.py           Config model (Pydantic + TOML + env vars)
  types.py            Shared dataclasses
  vault/              Scanning, parsing, wikilinks, chunking
  embedding/          OpenLLM HTTP client
  index/              ChromaDB store, BM25 store, SHA-256 hash tracker
  retrieval/          Hybrid score fusion, filter builder
  indexer/            Full + incremental indexing pipeline
  server/             FastMCP server, tool implementations
```

### Test layout

Tests mirror the source tree. Integration tests live in `tests/integration/`. All external HTTP calls are mocked with `respx`; ChromaDB tests use a real PersistentClient against a temp directory.

---

## SMB vault (TrueNAS / Samba)

autoknowledge uses a Docker-managed CIFS volume to mount the SMB share directly inside the container. No host-level `/etc/fstab` entry or manual `mount` command is needed.

### Prerequisites

`cifs-utils` must be installed on the Docker host so the kernel CIFS driver is available:

```bash
# Arch / CachyOS / Manjaro
sudo pacman -S cifs-utils

# Debian / Ubuntu
sudo apt install cifs-utils

# Fedora / RHEL
sudo dnf install cifs-utils
```

### `.env` configuration

```env
SMB_HOST=truenas.local
SMB_USER=smbshare
SMB_PASSWORD=yourpassword
# Share name followed by the path within the share — no leading slashes
SMB_SHARE_PATH=Vault/Media/obsidian/truenas-obsidian-vault
```

The `vault_smb` volume in `docker-compose.yml` translates these into a CIFS mount equivalent to:

```bash
mount -t cifs //truenas.local/Vault/Media/obsidian/truenas-obsidian-vault /vault \
  -o vers=3.0,username=smbshare,password=yourpassword,ro,iocharset=utf8,noperm
```

### Verify the mount

Before indexing, confirm Docker can mount the volume:

```bash
docker compose run --rm autoknowledge ls /vault
```

You should see your Obsidian vault files listed. If you see an error, check:
- The TrueNAS hostname resolves: `ping truenas.local`
- The share path is correct (share name first, subdirectory after, no `//` prefix)
- Credentials are valid: try mounting manually with `mount -t cifs` first
- SMB protocol version: if the server requires SMB 2.1 or 1.0, change `vers=3.0` in `docker-compose.yml` → `vault_smb` → `driver_opts.o`

### Switching between SMB and local vault

The active vault volume is controlled by `VAULT_VOLUME` in `.env`:

| Vault location | `VAULT_VOLUME` | Extra variables needed |
|---------------|----------------|------------------------|
| SMB / NAS share | `vault_smb` (default) | `SMB_HOST`, `SMB_USER`, `SMB_PASSWORD`, `SMB_SHARE_PATH` |
| Local host path | `vault_local` | `VAULT_PATH` |

---

## Troubleshooting

**Embedding service not reachable**

```
RuntimeError: Embedding service not reachable at http://localhost:3000
```

Start OpenLLM (`openllm serve ...`) or `docker compose up -d embeddings` and wait for the health check to pass.

**Embedding dimension mismatch**

```
RuntimeError: Embedding model mismatch: collection was built with 'BAAI/bge-small-en-v1.5'
but config says 'BAAI/bge-large-en-v1.5'. Run 'autoknowledge index --full' to rebuild.
```

You changed the model. Run `autoknowledge index --full` to drop and rebuild the index.

**Index is empty / no search results**

Run the indexer first: `autoknowledge index` (or `docker compose run --rm autoknowledge index`).

**Claude Code does not show autoknowledge in MCP servers**

- Verify the path in `~/.claude.json` is absolute and correct.
- Check that Docker is running: `docker info`.
- Test the server manually: `docker compose run --rm -T autoknowledge serve` — it should start without errors and wait on stdin.

**SMB vault: `ls /vault` shows empty or permission denied**

```
docker compose run --rm autoknowledge ls /vault
# shows nothing, or: ls: cannot access '/vault': Permission denied
```

1. Verify the share path — the `SMB_SHARE_PATH` must start with the TrueNAS **share name** (e.g., `Vault`), not the full filesystem path:
   ```env
   # Correct
   SMB_SHARE_PATH=Vault/Media/obsidian/truenas-obsidian-vault
   # Wrong
   SMB_SHARE_PATH=/mnt/pool/Media/obsidian/truenas-obsidian-vault
   ```
2. Test credentials with a manual mount:
   ```bash
   sudo mount -t cifs //truenas.local/Vault/Media/obsidian/truenas-obsidian-vault /mnt/test \
     -o vers=3.0,username=smbshare,password=yourpassword,ro
   ```
3. If the server requires a different SMB version, edit the `o:` line in `docker-compose.yml` under `vault_smb`:
   ```yaml
   o: "vers=2.1,username=..."   # or vers=1.0 for older servers
   ```
4. TrueNAS: ensure the SMB share has **Guest access disabled** and the user `smbshare` has read permission on the share in TrueNAS → Sharing → SMB → Edit → Access.

**SMB vault: `docker compose up` fails with "can't mount"**

Docker creates the CIFS volume on first use. If creation fails, Docker will report it at `compose up` or `compose run` time (not at `compose up -d embeddings`). Check `docker volume inspect autoknowledge_vault_smb` for error details and verify `cifs-utils` is installed: `pacman -Q cifs-utils`.
