#!/usr/bin/env python3
"""Open the stand: wait for both demonstrators, then a kiosk on each monitor
(DECISIONS.md §26).

Runs in the stand account's desktop session, started by its XDG autostart
entry, with the system Python and the standard library only, like kiosk.py
beside it, which places each window (§11). The containers are not started
here: they run under the maintainer's account and come back at every boot by
themselves (§10); the stand account has no Docker.

    python3 tools/stand.py
    python3 tools/stand.py --crane-serial 114203 --viz-serial 114210

Each kiosk is a child process; one that exits - closed by mistake, crashed -
is opened again, at most a few times a minute, so that a broken setup does
not spin.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

KIOSK: Path = Path(__file__).with_name("kiosk.py")
CRANE_URL: str = "http://localhost:8081/"
VIZ_URL: str = "http://localhost:8080/"
# The office's monitors (§11): the crane on the left one, feature-viz on the
# right one; swapped at the venue by argument, not by editing this file.
CRANE_SERIAL: int = 114210
VIZ_SERIAL: int = 114203
# The containers are healthy about 5 s after the daemon starts them (§10);
# after a cold boot the daemon itself takes a while.
WAIT_SECONDS: float = 120.0
POLL_SECONDS: float = 1.0
# A kiosk that exits this often within the window is left closed.
MAX_RESTARTS: int = 3
RESTART_WINDOW: float = 60.0
RESTART_PAUSE: float = 2.0


def healthy(url: str, timeout: float = 2.0) -> bool:
    """Whether the server's /healthz answers 200."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/healthz", timeout=timeout) as r:
            return bool(r.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for(
    urls: Sequence[str],
    limit: float,
    poll: float = POLL_SECONDS,
    check: Callable[[str], bool] = healthy,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """Wait until every server is healthy or `limit` seconds have passed;
    returns those still not healthy (empty when all are)."""
    end: float = clock() + limit
    waiting: list[str] = list(urls)
    while True:
        waiting = [u for u in waiting if not check(u)]
        if not waiting or clock() >= end:
            return waiting
        print(f"stand: waiting for {', '.join(waiting)}", flush=True)
        sleep(poll)


@dataclass
class Kiosk:
    """One monitor's kiosk, opened through kiosk.py and opened again when it
    exits, within the restart limit."""

    name: str
    serial: int
    url: str
    process: subprocess.Popen[bytes] | None = None
    restarts: deque[float] = field(default_factory=deque)
    given_up: bool = False

    def command(self) -> list[str]:
        return [
            sys.executable,
            str(KIOSK),
            "open",
            "--serial",
            str(self.serial),
            "--url",
            self.url,
        ]

    def open(self) -> subprocess.Popen[bytes]:
        """In a process group of its own, so that closing it ends Firefox
        too, not only kiosk.py waiting for it."""
        return subprocess.Popen(self.command(), start_new_session=True)

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)

    def may_restart(self, now: float) -> bool:
        """At most MAX_RESTARTS within RESTART_WINDOW seconds."""
        while self.restarts and now - self.restarts[0] > RESTART_WINDOW:
            self.restarts.popleft()
        return len(self.restarts) < MAX_RESTARTS


def supervise(
    kiosks: Sequence[Kiosk],
    start: Callable[[Kiosk], subprocess.Popen[bytes]],
    running: Callable[[], bool],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Open every kiosk, then open again each one that exits, until
    `running()` turns false or all have been given up."""
    for k in kiosks:
        k.process = start(k)
    while running() and not all(k.given_up for k in kiosks):
        for k in kiosks:
            if k.given_up or k.process is None or k.process.poll() is None:
                continue
            now: float = clock()
            if not k.may_restart(now):
                print(f"stand: {k.name} keeps exiting; left closed", flush=True)
                k.given_up = True
                continue
            print(
                f"stand: {k.name} exited ({k.process.returncode}); opening again", flush=True
            )
            k.restarts.append(now)
            sleep(RESTART_PAUSE)
            k.process = start(k)
        sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--crane-serial", type=int, default=CRANE_SERIAL)
    parser.add_argument("--viz-serial", type=int, default=VIZ_SERIAL)
    parser.add_argument("--crane-url", default=CRANE_URL)
    parser.add_argument("--viz-url", default=VIZ_URL)
    parser.add_argument(
        "--wait", type=float, default=WAIT_SECONDS, help="seconds to wait for the servers"
    )
    args: argparse.Namespace = parser.parse_args(argv)

    missing: list[str] = wait_for([args.crane_url, args.viz_url], args.wait)
    if missing:
        # Opened anyway: the crane's page shows "connecting …" and reconnects
        # by itself once its container is up (§9).
        print(f"stand: not healthy after {args.wait:.0f} s: {', '.join(missing)}", flush=True)

    kiosks: list[Kiosk] = [
        Kiosk("crane", args.crane_serial, args.crane_url),
        Kiosk("feature-viz", args.viz_serial, args.viz_url),
    ]
    stop: list[bool] = []

    def on_signal(signum: int, frame: FrameType | None) -> None:
        stop.append(True)

    # Logging out ends the session's processes with SIGTERM.
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        supervise(kiosks, Kiosk.open, lambda: not stop)
    finally:
        for k in kiosks:
            k.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
