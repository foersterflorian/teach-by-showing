"""Constants and helpers shared by the tests. Kept out of conftest.py, which
pytest loads as a plugin and which is not meant to be imported from."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
