"""udev rule for the GameSir's keyboard and mouse interface (DECISIONS.md
§12). Its effect was verified on the stand machine; this guards that the
rule keeps switching off only that interface and never the gamepad."""

from __future__ import annotations

import re
from pathlib import Path

from tests.support import ROOT

RULE_FILE: Path = ROOT / "tools" / "udev" / "70-teach-by-showing-gamepad.rules"


def rules() -> list[dict[str, str]]:
    """Each rule line as {key: value}, both match (==) and assignment (=)."""
    parsed: list[dict[str, str]] = []
    for line in RULE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            parsed.append(dict(re.findall(r'([A-Za-z]+(?:\{\w+\})?)=?="([^"]*)"', line)))
    return parsed


def test_one_rule_deauthorises_interface_01_of_the_gamesir_only() -> None:
    [rule] = rules()
    assert rule["ATTRS{idVendor}"] == "3537"
    assert rule["ATTRS{idProduct}"] == "1082"
    assert rule["ATTR{bInterfaceNumber}"] == "01"
    assert rule["ENV{DEVTYPE}"] == "usb_interface"
    assert rule["ATTR{authorized}"] == "0"


def test_the_rule_never_touches_the_whole_device() -> None:
    """Deauthorising the device instead of the interface would take the
    gamepad with it."""
    text: str = RULE_FILE.read_text(encoding="utf-8")
    assert 'ENV{DEVTYPE}=="usb_interface"' in text
    assert 'ENV{DEVTYPE}=="usb_device"' not in text
