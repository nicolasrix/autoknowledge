"""MCP server: startup, tool registration, and transport selection."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP

from autoknowledge.config import Config
from autoknowledge.embedding.client import EmbeddingClient
from autoknowledge.index.bm25_store import BM25Store
from autoknowledge.index.chroma_store import ChromaStore
from autoknowledge.retrieval.hybrid import HybridRetriever
from autoknowledge.server.tools import get_document, ingest_pdfs, reindex, search_knowledge

logger = logging.getLogger(__name__)


async def run_server(
    config: Config,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Initialise all services and start the MCP server."""
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Initialise storage
    chroma = ChromaStore(config.data_dir, config.embedding)
    bm25 = BM25Store()

    # Warm up BM25 from existing ChromaDB data
    chunk_count = chroma.count()
    if chunk_count > 0:
        logger.info("Building BM25 index from %d existing chunks …", chunk_count)
        bm25.build(chroma.get_all_chunks_text())
    else:
        logger.warning(
            "Index is empty. Run 'autoknowledge index' to index your vault first."
        )

    async with EmbeddingClient(config.embedding) as embedder:
        # Pre-flight check
        if not await embedder.healthcheck():
            logger.warning(
                "Embedding service not reachable at %s. "
                "search_knowledge will fail until the service is available.",
                config.embedding.base_url,
            )

        retriever = HybridRetriever(chroma, bm25, embedder, config.retrieval)

        # Build FastMCP app
        mcp = FastMCP(
            name="autoknowledge",
            instructions=(
                "Local RAG server over an Obsidian vault. "
                "Use search_knowledge to find relevant passages, "
                "get_document to read a full file, "
                "ingest_pdfs to convert PDF files to Markdown, "
                "and reindex to refresh the index."
            ),
            host=host,
            port=port,
        )

        @mcp.tool()
        async def search_knowledge_tool(
            query: Annotated[str, "Natural language search query"],
            top_k: Annotated[int, "Number of results to return (1–100)"] = config.retrieval.default_top_k,
            filters: Annotated[
                dict[str, Any] | None,
                "Optional metadata filters: tags (list), path_prefix (str), title (str)",
            ] = None,
        ) -> str:
            """Search the knowledge vault using hybrid semantic + keyword search."""
            return await search_knowledge(query, max(1, min(top_k, 100)), filters, retriever)

        @mcp.tool()
        def get_document_tool(
            path: Annotated[str, "Path to document, relative to vault root or absolute"],
        ) -> str:
            """Return the full Markdown content of a document from the vault."""
            return get_document(path, config.vault_path)

        @mcp.tool()
        async def ingest_pdfs_tool(
            input_path: Annotated[str, "Path to a PDF file or directory of PDFs to convert"],
            output_dir: Annotated[
                str | None,
                "Output directory for .md files. Defaults to same directory as each PDF.",
            ] = None,
            describe_images: Annotated[
                bool,
                "Describe images via Claude vision API (requires ANTHROPIC_API_KEY env var)",
            ] = False,
        ) -> str:
            """Convert PDF files to Markdown and save them for indexing.

            After conversion, call reindex to make the new notes searchable.
            """
            return await ingest_pdfs(input_path, output_dir, describe_images, config)

        @mcp.tool()
        async def reindex_tool(
            scope: Annotated[
                str | None,
                "Optional path (file or folder) relative to vault root to limit reindexing scope",
            ] = None,
        ) -> str:
            """Trigger incremental reindexing of the vault (or a scoped subset)."""
            return await reindex(scope, config, chroma, bm25)

        logger.info("Starting MCP server (transport=%s)", transport)

        if transport == "stdio":
            await mcp.run_stdio_async()
        elif transport == "sse":
            await mcp.run_sse_async()
        else:
            raise ValueError(f"Unknown transport: {transport!r}")
