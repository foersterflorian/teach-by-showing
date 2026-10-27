"""The endurance test's tool (DECISIONS.md §27). Only what it stores and how
its visitor drives; the figures come from runs against the stand and are
recorded in DECISIONS.md with their machine."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from teach_by_showing.gallery import VISITOR_NAME, GalleryStore
from tests.support import ROOT


def load_endurance() -> ModuleType:
    # tools/ is not a package; the script must stay runnable on its own.
    path = ROOT / "tools" / "endurance.py"
    spec = importlib.util.spec_from_file_location("endurance", path)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["endurance"] = module
    spec.loader.exec_module(module)
    return module


endurance: ModuleType = load_endurance()


def test_seeded_agents_are_numbered_as_the_stand_numbers_them(tmp_path: Path) -> None:
    endurance.seed(tmp_path, 3)
    store: GalleryStore = GalleryStore(tmp_path)
    assert [a.name for a in store.agents()] == [VISITOR_NAME.format(n) for n in (1, 2, 3)]
    assert store.next_number() == 4


def test_seeding_refuses_a_gallery_that_holds_agents(tmp_path: Path) -> None:
    endurance.seed(tmp_path, 1)
    try:
        endurance.seed(tmp_path, 1)
    except SystemExit:
        return
    raise AssertionError("seeded twice")


def test_the_visitor_drives_towards_the_target_capped_and_damps_the_swing() -> None:
    style = endurance.Style(0.5)
    assert style.command(0.0, 0.0, 10.0) == 0.5
    assert style.command(10.0, 0.0, 0.0) == -0.5
    # Load swung to +x: the trolley follows it.
    assert style.command(5.0, 0.1, 5.0) > 0
