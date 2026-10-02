"""Funding notice (DECISIONS.md §2)."""

from __future__ import annotations

import re

import pytest

from teach_by_showing import funding
from tests.support import ROOT

# Leading bytes of the two formats the funder ships.
MAGIC: dict[str, bytes] = {".jpg": b"\xff\xd8\xff", ".png": b"\x89PNG\r\n\x1a\n"}


def test_notice_in_code_matches_readme_verbatim() -> None:
    """One wording everywhere. The README links the project name, so the
    Markdown link is stripped before comparing."""
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    section: str = readme.split("## Funding", 1)[1].split("\n\n", 2)[1]
    plain: str = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", section).strip()
    assert plain == funding.FUNDING_TEXT


def test_readme_links_the_funding_url() -> None:
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"]({funding.FUNDING_URL})" in readme


def test_readme_shows_the_packaged_logos() -> None:
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    for name, _ in funding.FUNDING_LOGOS:
        assert f"src/teach_by_showing/funding/{name}" in readme


@pytest.mark.parametrize("name", [n for n, _ in funding.FUNDING_LOGOS])
def test_logos_load_from_the_package(name: str) -> None:
    data: bytes = funding.funding_asset(name)
    assert data.startswith(MAGIC[name[name.rfind(".") :]])


@pytest.mark.parametrize("name", ["../funding.py", "__init__.py", "missing.png"])
def test_only_listed_logos_are_served(name: str) -> None:
    with pytest.raises(KeyError):
        funding.funding_asset(name)
