"""CLI entry point for autoknowledge."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoknowledge",
        description="Local-first RAG MCP server for Obsidian vaults",
    )
    parser.add_argument(
        "--config", type=Path, default=None, metavar="PATH",
        help="Path to config.toml (default: auto-discover)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_p = subparsers.add_parser("index", help="Index the vault")
    index_p.add_argument(
        "--full", action="store_true",
        help="Force full reindex, ignoring cached hashes",
    )
    index_p.add_argument(
        "--path", type=Path, default=None, metavar="VAULT_PATH",
        help="Override vault path from config",
    )

    serve_p = subparsers.add_parser("serve", help="Start the MCP server")
    serve_p.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="MCP transport type (default: stdio)",
    )
    serve_p.add_argument(
        "--host", default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    serve_p.add_argument(
        "--port", type=int, default=8080,
        help="Port for SSE transport (default: 8080)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from autoknowledge.config import load_config
    config = load_config(args.config)

    if args.command == "index":
        import asyncio
        from autoknowledge.indexer.pipeline import run_index
        stats = asyncio.run(run_index(config, full=args.full, vault_override=args.path))
        print(stats.summary())

    elif args.command == "serve":
        import asyncio
        from autoknowledge.server.mcp_server import run_server
        asyncio.run(run_server(config, transport=args.transport, host=args.host, port=args.port))
