"""Configuration loading and validation."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class VaultConfig(BaseModel, frozen=True):
    path: Path = Path("~/obsidian-vault")


class EmbeddingConfig(BaseModel, frozen=True):
    base_url: str = "http://localhost:3000"
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = Field(default=384, gt=0)
    batch_size: int = Field(default=64, gt=0)


class IndexConfig(BaseModel, frozen=True):
    data_dir: Path = Path("~/.local/share/autoknowledge")
    chunk_min_tokens: int = Field(default=400, gt=0)
    chunk_max_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)

    @field_validator("chunk_max_tokens")
    @classmethod
    def max_greater_than_min(cls, v: int, info: Any) -> int:
        if "chunk_min_tokens" in (info.data or {}) and v <= info.data["chunk_min_tokens"]:
            raise ValueError("chunk_max_tokens must be greater than chunk_min_tokens")
        return v


class RetrievalConfig(BaseModel, frozen=True):
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    default_top_k: int = Field(default=10, ge=1, le=100)


class Config(BaseModel, frozen=True):
    vault: VaultConfig = VaultConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    index: IndexConfig = IndexConfig()
    retrieval: RetrievalConfig = RetrievalConfig()

    @property
    def vault_path(self) -> Path:
        return self.vault.path.expanduser().resolve()

    @property
    def data_dir(self) -> Path:
        return self.index.data_dir.expanduser().resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides for containerized deployments."""
    overrides: dict[str, Any] = {}

    if vault_path := os.environ.get("AUTOKNOWLEDGE_VAULT_PATH"):
        overrides["vault"] = {"path": vault_path}
    if data_dir := os.environ.get("AUTOKNOWLEDGE_DATA_DIR"):
        overrides.setdefault("index", {})["data_dir"] = data_dir
    if base_url := os.environ.get("AUTOKNOWLEDGE_EMBEDDING_BASE_URL"):
        overrides.setdefault("embedding", {})["base_url"] = base_url
    if model := os.environ.get("AUTOKNOWLEDGE_EMBEDDING_MODEL"):
        overrides.setdefault("embedding", {})["model"] = model
    if dimension := os.environ.get("AUTOKNOWLEDGE_EMBEDDING_DIMENSION"):
        overrides.setdefault("embedding", {})["dimension"] = int(dimension)

    return _deep_merge(raw, overrides) if overrides else raw


def load_config(path: Path | None = None) -> Config:
    """Load config from file (auto-discovered) and apply env var overrides.

    Search order:
    1. Explicit path argument
    2. ./config.toml
    3. ~/.config/autoknowledge/config.toml
    4. Defaults (no file required)
    """
    search_paths: list[Path | None] = [
        path,
        Path("./config.toml"),
        Path("~/.config/autoknowledge/config.toml").expanduser(),
    ]

    raw: dict[str, Any] = {}
    for p in search_paths:
        if p is not None and p.exists():
            with open(p, "rb") as f:
                raw = tomllib.load(f)
            break

    raw = _apply_env_overrides(raw)
    return Config.model_validate(raw)
