"""Fetcher không được nhận nhầm repo cha khi clone source bị dở."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_official_repos", ROOT / "scripts" / "fetch_official_repos.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkout_identity_rejects_parent_repo(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()
    target = tmp_path / "external" / "goodpoints"
    target.mkdir(parents=True)

    def fake_git(_target, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("remote", "get-url", "origin"):
            return "https://github.com/microsoft/goodpoints.git"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_git", fake_git)
    with pytest.raises(RuntimeError, match="không phải Git top-level"):
        module._validate_checkout_identity(
            "goodpoints", target, "https://github.com/microsoft/goodpoints.git",
        )


def test_remote_normalization_keeps_exact_repository_identity() -> None:
    module = _load_script()
    assert module._normalized_remote("https://github.com/X/Y.git/") == (
        "https://github.com/x/y"
    )
