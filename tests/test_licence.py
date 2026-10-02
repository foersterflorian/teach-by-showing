"""Licence: MIT (DECISIONS.md §1)."""

from __future__ import annotations

import tomllib

from tests.support import ROOT


def test_pyproject_and_licence_file_agree_on_mit() -> None:
    meta: dict[str, object] = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert meta["license"] == "MIT"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_readme_excludes_the_logos_from_the_licence() -> None:
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "not covered by the MIT" in readme
