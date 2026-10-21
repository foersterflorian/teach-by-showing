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


def test_the_style_profile_is_shown_as_the_server_words_it() -> None:
    """§22: the page has a place for the profile, and the script fills it
    with the server's cells; no figure or label is computed or worded in it."""
    page: str = (STATIC / "index.html").read_text(encoding="utf-8")
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    for name in ("profile-panel", "profile-title", "profile", "profile-note"):
        assert f'id="{name}"' in page, name
    assert "showProfile(newest.profile)" in main
    for wording in ("Steckbrief", "Fahrzeit", "Stickweg", "Restpendeln", "nicht erreicht"):
        assert wording not in main, wording


@pytest.mark.parametrize("path", ["/", "/style.css", "/main.js", "/curves.js"])
def test_every_file_is_revalidated_so_an_update_is_never_mixed_with_old_files(
    client: TestClient, path: str
) -> None:
    """§20: a browser kept an old stylesheet beside new scripts, and a canvas
    sized by the new scripts grew without bound."""
    r = client.get(path)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_a_canvas_never_grows_beyond_the_window() -> None:
    """§20: under a stale stylesheet a canvas took its size from its own
    pixels and grew with every frame until Firefox stopped drawing. Bounded
    by the window, a missing stylesheet misplaces the page but cannot freeze
    it."""
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    fit: str = main[main.index("function fitCanvas") :]
    fit = fit[: fit.index("\n}\n")]
    assert "Math.min(box.width, window.innerWidth)" in fit
    assert "Math.min(box.height, window.innerHeight)" in fit
