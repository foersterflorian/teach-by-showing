"""The screen's language (DECISIONS.md §24): one catalogue for both
languages, the staff's key that switches between them, the language kept on
disk, and a page that speaks it."""

from __future__ import annotations

import ast
import json
import re
import string
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teach_by_showing import api, funding, gallery
from teach_by_showing.stand import Phase, StandSession
from teach_by_showing.texts import (
    DEFAULT_LANGUAGE,
    ENGLISH,
    GERMAN,
    LANGUAGE_FILE,
    TEXTS,
    Language,
    Texts,
    load_language,
    number,
    save_language,
)
from tests.support import ROOT
from tests.test_stand import SETTINGS, demonstrate, press, showcase

SRC: Path = ROOT / "src" / "teach_by_showing"
STATIC: Path = SRC / "static"
# Where German may stand in the code: the catalogue, the funder's wording,
# and the name a visitor's agent is stored under (§23), which is its identity
# in the gallery, not a text on the screen.
GERMAN_ALLOWED: frozenset[str] = frozenset({"texts.py", "funding.py"})
STORED_NAMES: frozenset[str] = frozenset({gallery.VISITOR_NAME})


def placeholders(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}


def wordings(words: Texts) -> dict[str, str]:
    """Every text of a language by field name, tuples spelt out."""
    out: dict[str, str] = {}
    for f in fields(Texts):
        value: str | tuple[str, ...] = getattr(words, f.name)
        if isinstance(value, tuple):
            out.update({f"{f.name}[{k}]": v for k, v in enumerate(value)})
        else:
            out[f.name] = value
    return out


def test_both_languages_fill_the_same_placeholders() -> None:
    """A template that names {n} in German must name it in English too, or
    format() fails on the stand in one language only."""
    german, english = wordings(GERMAN), wordings(ENGLISH)
    assert german.keys() == english.keys()
    for name in german:
        assert placeholders(german[name]) == placeholders(english[name]), name


def test_every_text_is_worded_and_differs_where_it_should() -> None:
    assert all(wordings(GERMAN).values()) and all(wordings(ENGLISH).values())
    assert GERMAN.decimal == "," and ENGLISH.decimal == "."
    assert GERMAN.language_notice == "Deutsch" and ENGLISH.language_notice == "English"


def test_numbers_follow_the_language() -> None:
    assert number(3.25, 1, GERMAN, " s") == "3,2 s"
    assert number(3.25, 1, ENGLISH, " s") == "3.2 s"
    assert number(0.1, 2, GERMAN) == "0,10"


def test_english_is_the_default_until_the_staff_switch(tmp_path: Path) -> None:
    assert DEFAULT_LANGUAGE is Language.EN
    assert load_language(tmp_path) is Language.EN


def test_the_language_survives_a_restart(tmp_path: Path) -> None:
    save_language(tmp_path / "data", Language.DE)
    assert load_language(tmp_path / "data") is Language.DE
    assert not list((tmp_path / "data").glob("*.partial"))


@pytest.mark.parametrize("content", ['{"language": "fr"}', "[]", "not json", "{}"])
def test_an_unreadable_file_falls_back_to_the_default(tmp_path: Path, content: str) -> None:
    (tmp_path / LANGUAGE_FILE).write_text(content, encoding="utf-8")
    assert load_language(tmp_path) is DEFAULT_LANGUAGE


def _german_literals(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"[äöüÄÖÜß„]", node.value) and node.value not in STORED_NAMES:
                found.append(node.value)
    return found


@pytest.mark.parametrize(
    "path", sorted(p for p in SRC.glob("*.py") if p.name not in GERMAN_ALLOWED), ids=str
)
def test_no_german_text_bypasses_the_catalogue(path: Path) -> None:
    """A cheap net: a new text with an umlaut outside texts.py would speak
    German on an English screen. Docstrings name no German words."""
    assert _german_literals(path) == []


# The gamepad test page is for the staff, not for visitors (§12), and stays
# German.
STAFF_PAGES: frozenset[str] = frozenset({"gamepad.html"})


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for p in STATIC.iterdir()
        if p.suffix in {".js", ".html", ".css"} and p.name not in STAFF_PAGES
    ),
    ids=lambda p: p.name,
)
def test_the_frontend_holds_no_german(path: Path) -> None:
    assert not re.search(r"[äöüÄÖÜß„]", path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("language", list(Language))
def test_the_page_speaks_its_language(language: Language) -> None:
    page: str = api.index_page(language)
    words: Texts = TEXTS[language]
    assert f'<html lang="{language}">' in page
    assert "{{" not in page
    assert f'data-decimal="{words.decimal}"' in page
    assert f">{words.page_headline}<" in page
    assert funding.FUNDING_TEXT[language].split("“")[0].split("„")[0] in page


def test_the_page_reloads_when_the_server_speaks_another_language() -> None:
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    assert "message.language !== document.documentElement.lang" in main
    assert "location.reload()" in main
    assert "language: buttons.language" in main


def test_every_staff_key_is_documented_in_the_readme() -> None:
    """Decided 2026-10-22: every keyboard shortcut is written down."""
    script: str = (STATIC / "input.js").read_text(encoding="utf-8")
    readme: str = (ROOT / "README.md").read_text(encoding="utf-8")
    keys: list[str] = re.findall(
        r'e\.ctrlKey && e\.altKey && e\.key\.toLowerCase\(\) === "(\w)"', script
    )
    assert sorted(keys) == ["g", "l"]
    for key in keys:
        assert f"Ctrl+Alt+{key.upper()}" in readme, key


def stand_saving(chosen: list[Language], language: Language = Language.EN) -> StandSession:
    return StandSession.start(
        np.random.default_rng(0),
        SETTINGS,
        showcase(),
        lambda: "test",
        lambda _: None,
        seed=0,
        language=language,
        on_language=chosen.append,
    )


def test_the_staffs_key_switches_once_per_press_and_is_kept() -> None:
    chosen: list[Language] = []
    s: StandSession = stand_saving(chosen)
    assert s.state_message()["texts"]["prompt"] == ENGLISH.idle_prompt
    s.set_language_key(True)
    s.tick()
    m: dict[str, Any] = s.state_message()
    assert s.language is Language.DE and chosen == [Language.DE]
    # The same frame speaks German, and the banner confirms in German.
    assert m["language"] == "de" and m["texts"]["prompt"] == GERMAN.idle_prompt
    assert m["banner"] == GERMAN.language_notice
    assert m["texts"]["ghost_label"] == GERMAN.teacher_earlier
    s.set_language_key(True)  # still held: no second switch
    s.tick()
    assert s.language is Language.DE and chosen == [Language.DE]
    s.set_language_key(False)
    s.set_language_key(True)
    s.tick()
    assert s.language is Language.EN and chosen == [Language.DE, Language.EN]
    assert s.state_message()["banner"] == ENGLISH.language_notice


def test_a_switch_while_the_agent_drives_reaches_its_replay() -> None:
    s: StandSession = stand_saving([], Language.DE)
    press(s, a=True)
    demonstrate(s)
    press(s, a=True)
    assert s.phase is Phase.AGENT and s.replay is not None
    assert s.state_message()["profile"]["columns"][1] == GERMAN.profile_figures[0]
    s.set_language_key(True)
    s.tick()
    m: dict[str, Any] = s.state_message()
    assert s.replay.words is ENGLISH
    assert m["profile"]["columns"][1:] == list(ENGLISH.profile_figures)
    assert m["texts"]["ghost_label"] == ENGLISH.teacher_self
    assert m["model"]["formula"].startswith(ENGLISH.command_label)
    json.dumps(m)


def test_the_staffs_key_arrives_with_the_input_and_only_true_counts() -> None:
    s: StandSession = stand_saving([])
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "language": "yes"}))
    s.tick()
    assert s.language is Language.EN
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "language": True}))
    s.tick()
    assert s.language is Language.DE


def test_the_layout_does_not_depend_on_the_language() -> None:
    """Measured on 2026-10-22: English headers wrapped into four lines and
    the profile's row changed height as its cells filled, shrinking the
    scene; the model's top weight was cut off."""
    style: str = (STATIC / "style.css").read_text(encoding="utf-8")
    model: str = (STATIC / "model.js").read_text(encoding="utf-8")
    assert (
        "#profile th:not(:first-child), #profile td:not(:first-child) {\n  width: 8vw;"
        in style
    )
    assert "#profile-title { color: var(--fg); min-height: calc(3 * 1.15em); }" in style
    assert "Math.max(font * 0.6, y)" in model
    # The English headers no longer than the German ones, by characters.
    for german, english in zip(GERMAN.profile_figures, ENGLISH.profile_figures, strict=True):
        assert len(english) <= len(german) + 6, english
