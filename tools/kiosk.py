#!/usr/bin/env python3
"""Open a page full screen on a chosen monitor, in Firefox (DECISIONS.md §11).

Runs on the host, not in the container, with the system Python, so that it
needs neither the venv nor a build:

    python3 tools/kiosk.py list
    python3 tools/kiosk.py open --serial 114203
    python3 tools/kiosk.py open --serial 114210 --url http://localhost:8080/

A monitor is chosen by the serial number in its EDID, because connector names
differ between the office (DP-4, HDMI-A-2, docking station) and the venue
(both DisplayPort), and two monitors of the same model differ only there.

Written for the stand machine's desktop, COSMIC on Wayland. It needs
`cosmic-randr` (part of COSMIC) and the GTK 3 bindings for Python
(`python3-gi`, `gir1.2-gtk-3.0`), both installed with the desktop.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_URL: str = "http://localhost:8081/"


@dataclass(frozen=True)
class Output:
    """One monitor as the compositor reports it. The position is in the
    compositor's layout coordinates, which are scaled: a 3840-pixel monitor
    at 150 % is 2560 wide."""

    connector: str
    make: str
    model: str
    serial: int
    x: int
    y: int


OUTPUT_BLOCK: re.Pattern[str] = re.compile(
    r'^output "(?P<connector>[^"]+)" enabled=#true \{(?P<body>.*?)^\}', re.M | re.S
)


def parse_cosmic_randr(kdl: str) -> list[Output]:
    """Enabled outputs from `cosmic-randr list --kdl`."""
    outputs: list[Output] = []
    for block in OUTPUT_BLOCK.finditer(kdl):
        body: str = block["body"]
        make: re.Match[str] | None = re.search(r'make="([^"]*)"', body)
        model: re.Match[str] | None = re.search(r'model="([^"]*)"', body)
        serial: re.Match[str] | None = re.search(
            r'serial_number "(0x[0-9A-Fa-f]+|\d+)"', body
        )
        pos: re.Match[str] | None = re.search(r"^\s*position (-?\d+) (-?\d+)$", body, re.M)
        if serial is None or pos is None:
            continue
        outputs.append(
            Output(
                connector=block["connector"],
                make=make[1] if make else "",
                model=model[1] if model else "",
                serial=int(serial[1], 0),
                x=int(pos[1]),
                y=int(pos[2]),
            )
        )
    return outputs


def cosmic_outputs() -> list[Output]:
    kdl: str = subprocess.run(
        ["cosmic-randr", "list", "--kdl"], capture_output=True, text=True, check=True
    ).stdout
    return parse_cosmic_randr(kdl)


def gtk_monitor_origins() -> list[tuple[int, int]]:
    """Layout origin of every monitor, in the order GTK 3 numbers them.

    Firefox's --kiosk-monitor counts in exactly this order, which is not the
    order cosmic-randr lists them in (§11), so the number has to come from
    GTK itself. Imported here: only `open` needs it, and the tests do not."""
    # System package (python3-gi), not in the venv: no types for mypy there.
    import gi  # type: ignore[import-not-found]

    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk  # type: ignore[import-not-found]

    display = Gdk.Display.get_default()
    if display is None:
        raise SystemExit("kiosk: no display; run this inside the desktop session")
    origins: list[tuple[int, int]] = []
    for i in range(display.get_n_monitors()):
        g = display.get_monitor(i).get_geometry()
        origins.append((g.x, g.y))
    return origins


def monitor_index(target: Output, origins: list[tuple[int, int]]) -> int:
    """Firefox's number for the target, matched by layout origin."""
    matches: list[int] = [i for i, o in enumerate(origins) if o == (target.x, target.y)]
    if len(matches) != 1:
        raise SystemExit(f"kiosk: no unique GTK monitor at {target.x},{target.y}: {origins}")
    return matches[0]


def choose(outputs: list[Output], serial: int) -> Output:
    matches: list[Output] = [o for o in outputs if o.serial == serial]
    if len(matches) != 1:
        raise SystemExit(
            f"kiosk: {len(matches)} monitors with serial {serial}; see `kiosk.py list`"
        )
    return matches[0]


# Prefs for the fresh profile: nothing may appear in front of the page on
# the stand - no welcome page, no data-reporting notice, no default-browser
# question, no session restore after a crash, no translation offer.
USER_PREFS: dict[str, bool | str] = {
    "browser.aboutwelcome.enabled": False,
    "trailhead.firstrun.didSeeAboutWelcome": True,
    "datareporting.policy.dataSubmissionPolicyBypassNotification": True,
    "toolkit.telemetry.reportingpolicy.firstRun": False,
    "browser.shell.checkDefaultBrowser": False,
    "browser.startup.homepage_override.mstone": "ignore",
    "browser.sessionstore.resume_from_crash": False,
    "browser.translations.automaticallyPopup": False,
}


def user_js() -> str:
    lines: list[str] = []
    for key, value in USER_PREFS.items():
        literal: str = (
            ("true" if value else "false") if isinstance(value, bool) else f'"{value}"'
        )
        lines.append(f'user_pref("{key}", {literal});')
    return "\n".join(lines) + "\n"


def firefox_command(firefox: str, profile: Path, index: int, url: str) -> list[str]:
    """--profile with --no-remote: a fresh, separate instance, so that two
    kiosks (crane and feature-viz) never merge into one process or window,
    and nothing from an earlier run is restored."""
    return [
        firefox,
        "--no-remote",
        "--profile",
        str(profile),
        "--kiosk",
        "--kiosk-monitor",
        str(index),
        url,
    ]


def cmd_list() -> int:
    for o in cosmic_outputs():
        print(f"{o.connector:10} {o.make} {o.model:10} serial {o.serial:<10} at {o.x},{o.y}")
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    target: Output = choose(cosmic_outputs(), args.serial)
    index: int = monitor_index(target, gtk_monitor_origins())
    firefox: str | None = shutil.which("firefox")
    if firefox is None:
        raise SystemExit("kiosk: firefox not found")
    profile: Path = Path(tempfile.mkdtemp(prefix="teach-by-showing-kiosk-"))
    (profile / "user.js").write_text(user_js(), encoding="utf-8")
    print(
        f"kiosk: {target.connector} ({target.model}, serial {target.serial})"
        f" -> monitor {index}"
    )
    # Native Wayland: --kiosk-monitor names the output in the fullscreen
    # request, and only a Wayland client can name it there.
    env: dict[str, str] = {**os.environ, "MOZ_ENABLE_WAYLAND": "1"}
    try:
        command: list[str] = firefox_command(firefox, profile, index, args.url)
        return subprocess.run(command, env=env).returncode
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="connected monitors with serial and position")
    p_open = sub.add_parser("open", help="open a page full screen on one monitor")
    p_open.add_argument("--serial", type=int, required=True, help="EDID serial number")
    p_open.add_argument("--url", default=DEFAULT_URL)
    args: argparse.Namespace = parser.parse_args(argv)
    return cmd_list() if args.command == "list" else cmd_open(args)


if __name__ == "__main__":
    sys.exit(main())
