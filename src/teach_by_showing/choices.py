"""The staff's choices that survive a restart: the screen's language (§24)
and its theme (§25), each one small JSON file in the data directory, as
the gallery keeps its filter (§23). Knows nothing about HTTP.
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path

log: logging.Logger = logging.getLogger("teach_by_showing")


def load_choice[E: StrEnum](path: Path, key: str, kind: type[E], default: E) -> E:
    """The choice as last stored; the default when nothing, or nothing
    readable, is stored, so that a broken file never stops the stand."""
    if not path.exists():
        return default
    try:
        return kind(json.loads(path.read_text(encoding="utf-8"))[key])
    except (ValueError, KeyError, TypeError):
        log.warning("unreadable %s; using %s", path, default)
        return default


def save_choice(path: Path, key: str, value: StrEnum) -> None:
    """Written whole and then renamed, so that a crash never leaves half a
    file behind (as gallery.py stores agents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".partial")
    partial.write_text(json.dumps({key: str(value)}), encoding="utf-8")
    partial.replace(path)
