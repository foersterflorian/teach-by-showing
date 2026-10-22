"""Colours and the screen's theme (DECISIONS.md §25): one meaning per colour,
every colour a token in both sets, readable contrast, the staff's key that
switches between dark and light, and the theme kept on disk."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teach_by_showing import api
from teach_by_showing.choices import load_choice, save_choice
from teach_by_showing.stand import StandSession
from teach_by_showing.texts import ENGLISH, GERMAN, Language
from teach_by_showing.theme import (
    DEFAULT_THEME,
    THEME_FILE,
    Theme,
    load_theme,
    save_theme,
)
from tests.support import ROOT
from tests.test_stand import SETTINGS, showcase

STATIC: Path = ROOT / "src" / "teach_by_showing" / "static"
STYLE: str = (STATIC / "style.css").read_text(encoding="utf-8")
# Tokens drawn on the dark or light ground as graphics, and as text.
GRAPHICS: tuple[str, ...] = (
    "--human",
    "--agent",
    "--naive",
    "--rival",
    "--target",
    "--girder",
    "--trolley",
    "--rope",
    "--weight-pos",
    "--weight-neg",
    "--node",
)
TEXT: tuple[str, ...] = ("--fg", "--muted", "--label")


def tokens(selector: str) -> dict[str, str]:
    """The custom properties of one rule of the stylesheet."""
    body: str = STYLE.split(selector + " {", 1)[1].split("\n}", 1)[0]
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body))


DARK: dict[str, str] = tokens(":root")
LIGHT: dict[str, str] = tokens(':root[data-theme="light"]')


def luminance(colour: str) -> float:
    """WCAG relative luminance of a #rgb or #rrggbb colour."""
    hexes: str = colour.lstrip("#")
    if len(hexes) == 3:
        hexes = "".join(c * 2 for c in hexes)
    channels: list[float] = [int(hexes[k : k + 2], 16) / 255 for k in (0, 2, 4)]
    linear: list[float] = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_both_themes_name_the_same_colours() -> None:
    colours: set[str] = {
        k for k, v in DARK.items() if v.startswith("#") or k == "--funding-edge"
    }
    assert colours <= LIGHT.keys(), colours - LIGHT.keys()
    assert {k for k, v in LIGHT.items() if v.startswith("#")} <= DARK.keys()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_colour_reads_on_its_ground(theme: str) -> None:
    """WCAG: 3:1 for graphics, 4.5:1 for text, computed from the tokens."""
    t: dict[str, str] = DARK if theme == "dark" else LIGHT
    for name in GRAPHICS:
        assert contrast(t[name], t["--bg"]) >= 3.0, name
    for name in TEXT:
        assert contrast(t[name], t["--bg"]) >= 4.5, name


def test_the_scripts_hold_no_colour_of_their_own() -> None:
    for name in ("scene.js", "curves.js", "model.js", "main.js"):
        script: str = (STATIC / name).read_text(encoding="utf-8")
        assert not re.search(r"#[0-9a-fA-F]{3,6}\b", script), name
    scene: str = (STATIC / "scene.js").read_text(encoding="utf-8")
    assert "getComputedStyle(document.documentElement)" in scene


def test_green_and_red_mean_only_the_drivers() -> None:
    """Concept §8, decided 2026-10-22: the banner keeps the text's colour,
    the target zone fills in the target's, and the command bar takes the
    colour of whoever drives."""
    assert "#banner.success" not in STYLE and "#banner.timeout" not in STYLE
    assert "--ok" not in STYLE and "--bad" not in STYLE
    assert (
        '#readout[data-driver="agent"] #command-fill { background: var(--agent); }' in STYLE
    )
    scene: str = (STATIC / "scene.js").read_text(encoding="utf-8")
    assert "COLORS.calm" not in scene
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    assert 'el("readout").dataset.driver = view.driver;' in main


def test_the_page_switches_its_colours_without_a_reload() -> None:
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    switch: str = main[main.index("message.theme !==") :]
    switch = switch[: switch.index("}")]
    assert "dataset.theme = message.theme" in switch and "readColors()" in switch
    assert "reload" not in switch


@pytest.mark.parametrize("theme", list(Theme))
def test_the_page_comes_in_the_stands_theme(theme: Theme) -> None:
    assert f'data-theme="{theme}"' in api.index_page(Language.EN, theme)


def test_dark_is_the_default_and_the_theme_survives_a_restart(tmp_path: Path) -> None:
    assert DEFAULT_THEME is Theme.DARK and load_theme(tmp_path) is Theme.DARK
    save_theme(tmp_path, Theme.LIGHT)
    assert load_theme(tmp_path) is Theme.LIGHT
    assert json.loads((tmp_path / THEME_FILE).read_text(encoding="utf-8")) == {
        "theme": "light"
    }


@pytest.mark.parametrize("content", ['{"theme": "sepia"}', "[]", "not json", "{}"])
def test_an_unreadable_choice_falls_back_to_its_default(tmp_path: Path, content: str) -> None:
    path: Path = tmp_path / "choice.json"
    path.write_text(content, encoding="utf-8")
    assert load_choice(path, "theme", Theme, Theme.DARK) is Theme.DARK


def test_a_choice_is_written_whole(tmp_path: Path) -> None:
    save_choice(tmp_path / "deep" / "choice.json", "theme", Theme.LIGHT)
    assert not list((tmp_path / "deep").glob("*.partial"))


def stand_saving(chosen: list[Theme], language: Language = Language.EN) -> StandSession:
    return StandSession.start(
        np.random.default_rng(0),
        SETTINGS,
        showcase(),
        lambda: "test",
        lambda _: None,
        seed=0,
        language=language,
        on_theme=chosen.append,
    )


def test_the_staffs_key_switches_the_theme_once_per_press() -> None:
    chosen: list[Theme] = []
    s: StandSession = stand_saving(chosen)
    assert s.state_message()["theme"] == "dark"
    s.set_theme_key(True)
    s.tick()
    m: dict[str, Any] = s.state_message()
    assert s.theme is Theme.LIGHT and chosen == [Theme.LIGHT]
    assert m["theme"] == "light" and m["banner"] == ENGLISH.light_notice
    s.set_theme_key(True)  # still held: no second switch
    s.tick()
    assert chosen == [Theme.LIGHT]
    s.set_theme_key(False)
    s.set_theme_key(True)
    s.tick()
    assert s.theme is Theme.DARK and chosen == [Theme.LIGHT, Theme.DARK]


def test_the_theme_notice_speaks_the_screens_language() -> None:
    s: StandSession = stand_saving([], Language.DE)
    s.set_theme_key(True)
    s.tick()
    assert s.state_message()["banner"] == GERMAN.light_notice
    # Switching the language while it shows words it anew.
    s.set_language_key(True)
    s.tick()
    assert s.state_message()["banner"] == ENGLISH.language_notice


def test_the_theme_key_arrives_with_the_input_and_only_true_counts() -> None:
    s: StandSession = stand_saving([])
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "theme": "yes"}))
    s.tick()
    assert s.theme is Theme.DARK
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "theme": True}))
    s.tick()
    assert s.theme is Theme.LIGHT
