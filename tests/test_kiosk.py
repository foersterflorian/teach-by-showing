"""Kiosk placement script (DECISIONS.md §11). Only the parts that need no
desktop session: parsing, matching, the command line and the profile prefs.
Placement itself was verified by eye on the stand machine, see §11."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.support import ROOT


def load_kiosk() -> ModuleType:
    # tools/ is not a package; the script must stay runnable on its own.
    spec = importlib.util.spec_from_file_location("kiosk", ROOT / "tools" / "kiosk.py")
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["kiosk"] = module
    spec.loader.exec_module(module)
    return module


kiosk: ModuleType = load_kiosk()

# Shortened output of `cosmic-randr list --kdl` on the stand machine,
# 2026-10-02: two monitors of the same model, listed right one first.
KDL: str = """output "HDMI-A-2" enabled=#true {
  description make="PNP(AOC)" model="U2790B"
  physical 600 340
  position 2560 0
  scale 1.50
  serial_number "0x0001BE1B"
  modes {
    mode 3840 2160 59997 current=#true preferred=#true
  }
}
output "DP-4" enabled=#true {
  description make="PNP(AOC)" model="U2790B"
  position 0 0
  serial_number "0x0001BE22"
}
output "HDMI-A-1" enabled=#false {
  description make="X" model="Y"
  position 0 0
  serial_number "0x1"
}
"""


def test_parses_enabled_outputs_with_serial_and_position() -> None:
    outputs = kiosk.parse_cosmic_randr(KDL)
    assert [(o.connector, o.serial, o.x, o.y) for o in outputs] == [
        ("HDMI-A-2", 114203, 2560, 0),
        ("DP-4", 114210, 0, 0),
    ]
    assert outputs[0].model == "U2790B"


def test_monitor_number_comes_from_gtk_order_not_list_order() -> None:
    """Firefox numbers monitors as GTK does; on the stand machine that put
    the right monitor first (§11)."""
    outputs = kiosk.parse_cosmic_randr(KDL)
    gtk_order: list[tuple[int, int]] = [(2560, 0), (0, 0)]
    by_serial = {o.serial: kiosk.monitor_index(o, gtk_order) for o in outputs}
    assert by_serial == {114203: 0, 114210: 1}
    assert kiosk.monitor_index(outputs[1], [(0, 0), (2560, 0)]) == 0


def test_unknown_or_ambiguous_monitors_stop_the_script() -> None:
    outputs = kiosk.parse_cosmic_randr(KDL)
    with pytest.raises(SystemExit):
        kiosk.choose(outputs, 999)
    with pytest.raises(SystemExit):
        kiosk.monitor_index(outputs[0], [(0, 0)])


def test_firefox_runs_as_a_separate_kiosk_on_the_chosen_monitor() -> None:
    cmd: list[str] = kiosk.firefox_command(
        "firefox", Path("/tmp/p"), 1, "http://localhost:8081/"
    )
    assert cmd[0] == "firefox"
    assert "--no-remote" in cmd and "--kiosk" in cmd
    assert cmd[cmd.index("--profile") + 1] == "/tmp/p"
    assert cmd[cmd.index("--kiosk-monitor") + 1] == "1"
    assert cmd[-1] == "http://localhost:8081/"


def test_profile_prefs_suppress_first_run_screens() -> None:
    text: str = kiosk.user_js()
    assert 'user_pref("browser.aboutwelcome.enabled", false);' in text
    assert 'user_pref("browser.startup.homepage_override.mstone", "ignore");' in text
    assert text.count("user_pref(") == len(kiosk.USER_PREFS)


def test_script_needs_only_the_standard_library_at_import() -> None:
    """It runs with the system Python on the host, without the venv. GTK is
    imported inside the one function that needs it."""
    import ast

    tree = ast.parse((ROOT / "tools" / "kiosk.py").read_text())
    top: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= set(sys.stdlib_module_names) | {"__future__"}
