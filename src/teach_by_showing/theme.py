"""The screen's theme, dark or light (DECISIONS.md §25), and which of the two
the stand shows. The colours themselves are tokens in the stylesheet; the
server only says which set applies. Knows nothing about HTTP.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

from teach_by_showing.choices import load_choice, save_choice


class Theme(StrEnum):
    DARK = "dark"
    LIGHT = "light"


# Decided with the maintainer on 2026-10-22: dark, as beside feature-viz,
# until the staff switch.
DEFAULT_THEME: Final[Theme] = Theme.DARK
THEME_FILE: Final[str] = "theme.json"


def other_theme(theme: Theme) -> Theme:
    """The theme the staff's key switches to."""
    return Theme.LIGHT if theme is Theme.DARK else Theme.DARK


def load_theme(directory: Path) -> Theme:
    return load_choice(directory / THEME_FILE, "theme", Theme, DEFAULT_THEME)


def save_theme(directory: Path, theme: Theme) -> None:
    save_choice(directory / THEME_FILE, "theme", theme)
