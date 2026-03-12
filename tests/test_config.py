"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from autoknowledge.config import Config, load_config


def test_defaults_without_file() -> None:
    config = load_config()
    assert config.vault.path == Path("~/obsidian-vault")
    assert config.embedding.base_url == "http://localhost:3000"
    assert config.embedding.dimension == 384
    assert config.retrieval.alpha == 0.7
    assert config.retrieval.default_top_k == 10


def test_load_from_explicit_file(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[vault]\npath = "/my/vault"\n'
        '[embedding]\nbase_url = "http://myhost:4000"\ndimension = 768\n'
    )
    config = load_config(cfg_file)
    assert config.vault.path == Path("/my/vault")
    assert config.embedding.base_url == "http://myhost:4000"
    assert config.embedding.dimension == 768
    # Unspecified values fall back to defaults
    assert config.retrieval.alpha == 0.7


def test_load_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text('[vault]\npath = "/cwd/vault"\n')
    config = load_config()
    assert config.vault.path == Path("/cwd/vault")


def test_env_overrides_vault_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOKNOWLEDGE_VAULT_PATH", "/env/vault")
    config = load_config()
    assert config.vault.path == Path("/env/vault")


def test_env_overrides_embedding_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOKNOWLEDGE_EMBEDDING_BASE_URL", "http://embeddings:3000")
    config = load_config()
    assert config.embedding.base_url == "http://embeddings:3000"


def test_env_overrides_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOKNOWLEDGE_DATA_DIR", "/data")
    config = load_config()
    assert config.index.data_dir == Path("/data")


def test_env_overrides_take_precedence_over_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('[vault]\npath = "/file/vault"\n')
    monkeypatch.setenv("AUTOKNOWLEDGE_VAULT_PATH", "/env/vault")
    config = load_config(cfg_file)
    assert config.vault.path == Path("/env/vault")


def test_vault_path_property_expands_home() -> None:
    config = load_config()
    assert not str(config.vault_path).startswith("~")
    assert config.vault_path.is_absolute()


def test_data_dir_property_expands_home() -> None:
    config = load_config()
    assert not str(config.data_dir).startswith("~")
    assert config.data_dir.is_absolute()


def test_invalid_alpha_raises() -> None:
    with pytest.raises(Exception):
        Config.model_validate({"retrieval": {"alpha": 1.5}})


def test_invalid_top_k_raises() -> None:
    with pytest.raises(Exception):
        Config.model_validate({"retrieval": {"default_top_k": 0}})
