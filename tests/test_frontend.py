"""Frontend as served (DECISIONS.md §9, §2). The JavaScript itself is not
executed here; what is tested is what Python controls: the page, its files,
the funding notice, and that the scripts hold no crane logic."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from teach_by_showing import funding
from teach_by_showing.api import create_app
from teach_by_showing.crane import CraneParams
from teach_by_showing.live import LiveSession
from tests.support import ROOT

STATIC: Path = ROOT / "src" / "teach_by_showing" / "static"


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(lambda: LiveSession.start(np.random.default_rng(0)))) as c:
        yield c


def test_page_carries_the_funding_notice_verbatim(client: TestClient) -> None:
    page: str = client.get("/").text
    assert "<!--FUNDING-->" not in page
    assert html.escape(funding.FUNDING_TEXT) in page
    assert f'href="{funding.FUNDING_URL}"' in page
    for name, _ in funding.FUNDING_LOGOS:
        assert f'src="/funding/{name}"' in page


@pytest.mark.parametrize("name", [n for n, _ in funding.FUNDING_LOGOS])
def test_logos_are_served_unmodified(client: TestClient, name: str) -> None:
    r = client.get(f"/funding/{name}")
    assert r.status_code == 200
    assert r.content == funding.funding_asset(name)
    assert r.headers["content-type"].startswith("image/")


@pytest.mark.parametrize("name", ["api.py", "..%2Fapi.py", "LICENSE", ""])
def test_only_the_listed_logos_are_served(client: TestClient, name: str) -> None:
    assert client.get(f"/funding/{name}").status_code == 404


def test_every_file_the_page_references_is_served(client: TestClient) -> None:
    page: str = client.get("/").text
    refs: list[str] = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', page)
    assert {"scene.js", "input.js", "main.js", "style.css"} <= set(refs)
    for ref in refs:
        assert client.get(f"/{ref}").status_code == 200, ref


def test_page_needs_nothing_from_the_network() -> None:
    """Offline at runtime (§3): no CDN, no external fonts or scripts."""
    for path in STATIC.iterdir():
        text: str = path.read_text(encoding="utf-8")
        assert not re.search(r"(?:src|href)=\"https?://", text), path.name
        assert "@import" not in text and "fonts.googleapis" not in text, path.name


def test_scripts_hold_no_crane_parameters() -> None:
    """The frontend draws to the scale the config message gives it (§8, §9).
    A physical parameter copied into JavaScript would silently disagree with
    the simulation after the next tuning."""
    # Only the distinctive values: small round numbers such as 0.05 occur
    # legitimately as drawing sizes in metres.
    crane: CraneParams = CraneParams()
    values: set[float] = {
        crane.gravity,
        crane.rope_length,
        crane.max_speed,
        crane.max_accel,
    }
    for path in STATIC.glob("*.js"):
        numbers: set[float] = {
            float(n) for n in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", path.read_text())
        }
        assert not numbers & values, f"{path.name}: {numbers & values}"


def test_gamepad_probe_is_served(client: TestClient) -> None:
    """The staff page that shows raw Gamepad API values (§12)."""
    r = client.get("/gamepad.html")
    assert r.status_code == 200
    assert "navigator.getGamepads()" in r.text


def test_no_input_reacts_to_any_button() -> None:
    """The GameSir's triggers report as pressed at rest in Firefox (§12), so
    "press any button" would fire on its own. Buttons are read by index."""
    for path in STATIC.glob("*.js"):
        text: str = path.read_text(encoding="utf-8")
        assert not re.search(r"buttons\.(?:some|find|filter|every)\(", text), path.name


def test_the_ghost_is_drawn_and_its_texts_come_from_the_server() -> None:
    """§17, §18: the scripts draw and interpolate the ghost, and the German
    texts arrive from the server, in the config or with each state, not in
    the scripts."""
    scene: str = (STATIC / "scene.js").read_text(encoding="utf-8")
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    assert "view.ghost" in scene and "GHOST_ALPHA" in scene
    assert "a.ghost" in main and "view.banner" in main
    assert "showTexts(config)" in main and "showTexts(view.texts)" in main
    for key in ("headline", "hint", "prompt", "elapsed_label", "ghost_label"):
        assert f"{key}:" in main, key
    assert "view.ghost.command" in main and "ghost_command_label" in main
    for wording in ("Schatten", "Ihre Fahrt", "Vorbild", "Drücken Sie"):
        assert wording not in main, wording
