"""The stand's start script and its installer (DECISIONS.md §26). Only the
parts that need no desktop session: waiting for the servers, reopening a
kiosk that exits, the command line, and what the installer may touch. The
kiosks themselves were checked by eye on the stand machine, see §26."""

from __future__ import annotations

import ast
import importlib.util
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.support import ROOT

SETUP: Path = ROOT / "tools" / "stand-setup"


def load_stand() -> ModuleType:
    # tools/ is not a package; the script must stay runnable on its own.
    spec = importlib.util.spec_from_file_location("stand_script", ROOT / "tools" / "stand.py")
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["stand_script"] = module
    spec.loader.exec_module(module)
    return module


stand: ModuleType = load_stand()


class Clock:
    """A clock that advances only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_it_waits_until_every_server_is_healthy() -> None:
    clock: Clock = Clock()
    ready_at: dict[str, float] = {"crane": 3.0, "viz": 5.0}
    missing: list[str] = stand.wait_for(
        ["crane", "viz"],
        120.0,
        check=lambda u: clock.now >= ready_at[u],
        clock=clock,
        sleep=clock.sleep,
    )
    assert missing == [] and clock.now == 5.0


def test_it_gives_up_waiting_and_names_who_is_missing() -> None:
    clock: Clock = Clock()
    missing: list[str] = stand.wait_for(
        ["crane", "viz"], 10.0, check=lambda u: u == "crane", clock=clock, sleep=clock.sleep
    )
    assert missing == ["viz"] and clock.now == 10.0


@pytest.fixture
def server() -> Iterator[tuple[str, list[int]]]:
    """A local server whose /healthz answers the next status in the list."""
    statuses: list[int] = [503, 200]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(statuses.pop(0) if self.path == "/healthz" else 404)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    httpd: HTTPServer = HTTPServer(("127.0.0.1", 0), Handler)
    thread: threading.Thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", statuses
    httpd.shutdown()


def test_healthy_means_200_from_healthz(server: tuple[str, list[int]]) -> None:
    url, _ = server
    assert not stand.healthy(url)  # 503: the loop is not ticking
    assert stand.healthy(url)
    assert not stand.healthy("http://127.0.0.1:1/")  # nobody there


class FakeProcess:
    """Exits after `lives` polls; returncode as Popen has it."""

    def __init__(self, lives: int) -> None:
        self.lives: int = lives
        self.returncode: int | None = None
        self.pid: int = 0

    def poll(self) -> int | None:
        self.lives -= 1
        if self.lives < 0:
            self.returncode = 0
        return self.returncode


def test_a_kiosk_that_exits_is_opened_again() -> None:
    clock: Clock = Clock()
    k = stand.Kiosk("crane", 1, "u")
    opened: list[str] = []

    def start(kiosk: Any) -> FakeProcess:
        opened.append(kiosk.name)
        # The first one closes after a moment, the second stays.
        return FakeProcess(2 if len(opened) == 1 else 10**6)

    rounds: list[int] = []

    def running() -> bool:
        rounds.append(1)
        return len(rounds) < 20

    stand.supervise(
        [k],
        start,
        running,
        clock=clock,
        sleep=clock.sleep,
    )
    assert opened == ["crane", "crane"] and not k.given_up


def test_a_kiosk_that_keeps_exiting_is_left_closed() -> None:
    """At most MAX_RESTARTS within RESTART_WINDOW, so a broken setup does
    not spin."""
    clock: Clock = Clock()
    k = stand.Kiosk("crane", 1, "u")
    opened: list[int] = []

    def start(kiosk: Any) -> FakeProcess:
        opened.append(1)
        return FakeProcess(0)

    stand.supervise([k], start, lambda: True, clock=clock, sleep=clock.sleep)
    assert k.given_up and len(opened) == 1 + stand.MAX_RESTARTS
    assert clock.now < stand.RESTART_WINDOW


def test_the_kiosks_are_opened_through_kiosk_py_with_the_offices_defaults() -> None:
    k = stand.Kiosk("crane", stand.CRANE_SERIAL, stand.CRANE_URL)
    command: list[str] = k.command()
    assert command[1].endswith("kiosk.py") and command[2:5] == ["open", "--serial", "114210"]
    assert stand.VIZ_SERIAL == 114203 and stand.VIZ_URL == "http://localhost:8080/"
    assert stand.KIOSK.name == "kiosk.py" and stand.KIOSK.parent == ROOT / "tools"


def test_the_script_needs_only_the_standard_library() -> None:
    tree: ast.Module = ast.parse((ROOT / "tools" / "stand.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


def test_the_installer_copies_and_grants_nothing_else() -> None:
    """The stand account must never get Docker (§26): no group change."""
    script: str = (SETUP / "install.sh").read_text(encoding="utf-8")
    commands: str = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    for word in ("usermod", "gpasswd", "adduser", "useradd", "docker", "sudo", "chmod +s"):
        assert word not in commands, word
    assert '"$here/../kiosk.py" "$here/../stand.py" /opt/kmi-stand/' in commands


def test_the_autostart_entry_starts_the_installed_script() -> None:
    entry: str = (SETUP / "kmi-stand.desktop").read_text(encoding="utf-8")
    assert "Exec=/usr/bin/python3 /opt/kmi-stand/stand.py" in entry
    assert "Type=Application" in entry
