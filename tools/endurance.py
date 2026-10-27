#!/usr/bin/env python3
"""The stand's endurance test (DECISIONS.md §27): a gallery of many agents,
scripted visitors for hours, and a minute-by-minute log of what could wear
out. Three commands:

    # inside the container, as its own user, into an empty test volume
    docker run --rm -v tbs-endurance_data:/data -v ./tools:/tools:ro \\
        --entrypoint python teach-by-showing:0.2.1 /tools/endurance.py seed 500
    # beside the running container, from the venv
    pdm run python tools/endurance.py visit --hours 8
    pdm run python tools/endurance.py sample --out endurance.csv --hours 8

`seed` stores agents as the stand stores a visitor's, cloned from the two
shipped teaching drives. `visit` presses A, drives the three transports
with a simple controller, and lets the stand run through learning, the
agent and the gallery before it begins again; it must be the only client
that sends, because the latest message wins (§5), so no kiosk may be open.
`sample` only listens, and may run beside the kiosks.

Every figure recorded from this tool names the machine it ran on.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Final

DEFAULT_URL: Final[str] = "ws://127.0.0.1:8081/ws"
DEFAULT_HEALTH: Final[str] = "http://127.0.0.1:8081/healthz"
CRANE_CONTAINER: Final[str] = "tbs-endurance-teach-by-showing-1"
VIZ_CONTAINER: Final[str] = "feature-viz-feature-viz-1"
# Seeded agents date from the day before the event, so that a filter set
# on its first morning ("since now") would hide them, as it would the
# rehearsal's.
SEED_START: Final[datetime] = datetime(2026, 11, 2, 8, 0, 0)
# Distinct policies the seeded agents cycle through: each teaching drive
# with this many training seeds.
SEED_POLICIES: Final[int] = 10
FRAME: Final[float] = 1 / 60
# A message later than this after the previous one is a hitch worth
# naming: one frame lost and more.
HITCH_MS: Final[float] = 25.0


# --- seed -----------------------------------------------------------------


def seed(data_dir: Path, count: int) -> list[str]:
    """Store `count` agents numbered from 1 into an empty gallery. Runs
    inside the image: the package is there, and the files belong to the
    container's user."""
    import numpy as np

    from teach_by_showing.api import MODELS
    from teach_by_showing.cloning import Policy, clone
    from teach_by_showing.gallery import VISITOR_NAME, GalleryStore, make_agent
    from teach_by_showing.live import Settings
    from teach_by_showing.recording import load_run

    store: GalleryStore = GalleryStore(data_dir)
    if store.agents():
        raise SystemExit(f"{data_dir} holds agents already; seed only an empty volume")
    settings: Settings = Settings.from_dict({})
    showcase: Path = Path(str(resources.files("teach_by_showing"))) / "showcase"
    policies: list[tuple[str, Policy]] = []
    for name, _ in MODELS:
        run = load_run(showcase / name)
        for s in range(SEED_POLICIES):
            trained: Policy = clone([run], np.random.default_rng(s))[0]
            policies.append((run.label, trained))
    start: float = time.perf_counter()
    for number in range(1, count + 1):
        label, policy = policies[number % len(policies)]
        created: str = (SEED_START + timedelta(seconds=number)).strftime("%Y%m%d-%H%M%S")
        store.add(
            make_agent(
                number,
                VISITOR_NAME.format(number),
                created,
                f"{created}-{label}.json",
                policy,
                settings.crane,
                settings.task,
            )
        )
    took: float = time.perf_counter() - start
    return [f"seeded {count} agents into {data_dir} in {took:.1f} s"]


# --- visit ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Style:
    """The scripted visitor's driving: a speed command towards the target,
    capped at `cap`, plus a term in the swing that damps it. Tuned offline
    on the stand's physics: 30 of 30 targets, 9 s each at cap 1.0 and
    about 20 s at 0.5."""

    cap: float
    gain: float = 1.0
    swing: float = 2.0

    def command(self, x: float, theta: float, target: float) -> float:
        """Accelerating towards the side the load swings to damps it
        (θ̈ = −(g·θ + a)/l), so the speed command follows θ."""
        u: float = (
            max(-self.cap, min(self.cap, self.gain * (target - x))) + self.swing * theta
        )
        return max(-1.0, min(1.0, u))


@dataclass(slots=True)
class Visitor:
    """What the visitor sees of the latest state, and what it sends."""

    rng: random.Random
    state: dict[str, Any] = field(default_factory=dict)
    style: Style = Style(1.0)
    press_frames: int = 0
    walk_up_until: float = 0.0

    def message(self, now: float) -> dict[str, Any]:
        phase: str | None = self.state.get("phase")
        a: bool = False
        u: float = 0.0
        if phase == "idle" and now >= self.walk_up_until:
            # A press is a change from not held to held (§18).
            self.press_frames += 1
            a = self.press_frames % 6 < 3
        elif phase == "teach":
            self.press_frames = 0
            u = self.style.command(
                float(self.state["x"][0]),
                float(self.state["theta"][0]),
                float(self.state["target"][0]),
            )
        return {"type": "input", "u": u, "a": a, "b": False}

    def next_visit(self, now: float) -> None:
        self.style = Style(self.rng.uniform(0.45, 1.0))
        # The next visitor watches the waiting stand for a while first.
        self.walk_up_until = now + self.rng.uniform(5.0, 20.0)


async def visit(url: str, hours: float, rng: random.Random) -> list[str]:
    visitor: Visitor = Visitor(rng)
    visitor.next_visit(time.monotonic())
    visits: list[float] = []
    begun: float = time.monotonic()
    end: float = begun + hours * 3600
    async with websockets_connect(url) as ws:
        await ws.recv()  # config

        async def receive() -> None:
            async for raw in ws:
                message: dict[str, Any] = json.loads(raw)
                if message.get("type") == "state":
                    visitor.state = message

        receiver: asyncio.Task[None] = asyncio.create_task(receive())
        phase: str | None = None
        started: float = time.monotonic()
        while (now := time.monotonic()) < end:
            await ws.send(json.dumps(visitor.message(now)))
            new: str | None = visitor.state.get("phase")
            if new != phase:
                if new == "teach":
                    started = now
                if phase == "gallery" and new == "idle":
                    visits.append(now - started)
                    print(
                        f"{time.strftime('%H:%M:%S')} visit {len(visits)}"
                        f" in {visits[-1]:.0f} s, cap {visitor.style.cap:.2f}",
                        flush=True,
                    )
                    visitor.next_visit(now)
                elif new == "idle" and phase is not None:
                    print(
                        f"{time.strftime('%H:%M:%S')} back to idle from {phase}", flush=True
                    )
                    visitor.next_visit(now)
                phase = new
            await asyncio.sleep(FRAME)
        receiver.cancel()
    if not visits:
        return ["no visit completed"]
    return [
        f"{len(visits)} visits in {(time.monotonic() - begun) / 3600:.2f} h,"
        f" median {statistics.median(visits):.0f} s"
    ]


def websockets_connect(url: str) -> Any:
    # Imported here, so that `seed` runs in the image without the tools'
    # dependencies being part of its contract.
    import websockets

    return websockets.connect(url, max_size=None)


# --- sample ---------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in 0..100, as measure_ws.py."""
    ordered: list[float] = sorted(values)
    rank: int = max(0, min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1))))
    return ordered[rank]


@dataclass(slots=True)
class Window:
    """The state messages of one sampling interval."""

    intervals: list[float] = field(default_factory=list)
    frames: int = 0
    hitches: list[str] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        iv: list[float] = self.intervals or [0.0]
        return {
            "messages": len(self.intervals),
            "skipped": self.frames - len(self.intervals),
            "median_ms": round(statistics.median(iv), 2),
            "p99_ms": round(percentile(iv, 99), 2),
            "max_ms": round(max(iv), 2),
            "hitches": " ".join(self.hitches),
            "transitions": " ".join(self.transitions),
        }


def docker_stats(names: Sequence[str]) -> dict[str, str]:
    out: str = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    found: dict[str, str] = {}
    for line in out.splitlines():
        name, memory, cpu = line.split("\t")
        if name in names:
            short: str = "crane" if name == names[0] else "viz"
            found[f"{short}_mem"] = memory.split(" / ")[0]
            found[f"{short}_cpu"] = cpu
    return found


def firefox_rss_mb() -> float:
    """Every Firefox process together: the two kiosks and their content
    processes."""
    out: str = subprocess.run(
        ["ps", "-C", "firefox,firefox-bin", "-o", "rss="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return round(sum(int(v) for v in out.split()) / 1024, 1)


def volume_figures(container: str) -> dict[str, Any]:
    """Agents and the data's size, read inside the container, whose user
    owns the volume."""
    script: str = (
        "import pathlib, json;"
        "d = pathlib.Path('/data');"
        "print(json.dumps({'agents': len(list(d.glob('agents/agent-*.json'))),"
        " 'runs': len(list(d.glob('runs/*.json'))),"
        " 'data_mb': round(sum(p.stat().st_size"
        " for p in d.rglob('*') if p.is_file()) / 1e6, 1)}))"
    )
    out: subprocess.CompletedProcess[str] = subprocess.run(
        ["docker", "exec", container, "python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(out.stdout) if out.returncode == 0 else {"agents": "?"}


def log_problems(container: str, since: str) -> int:
    out: subprocess.CompletedProcess[str] = subprocess.run(
        ["docker", "logs", "--since", since, container],
        capture_output=True,
        text=True,
        check=False,
    )
    lines: list[str] = (out.stdout + out.stderr).splitlines()
    return sum(
        1 for line in lines if "ERROR" in line or "Traceback" in line or "WARNING" in line
    )


def health(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body: dict[str, Any] = json.loads(response.read())
            return body
    except OSError as error:
        return {"status": f"down: {error}"}


async def sample(
    url: str,
    health_url: str,
    out: Path,
    hours: float,
    every: float,
    containers: Sequence[str],
) -> list[str]:
    window: Window = Window()
    rows: int = 0
    end: float = time.monotonic() + hours * 3600
    fields: list[str] = [
        "time",
        "status",
        "phase",
        "ticks",
        "overruns",
        "clients",
        "messages",
        "skipped",
        "median_ms",
        "p99_ms",
        "max_ms",
        "crane_mem",
        "crane_cpu",
        "viz_mem",
        "viz_cpu",
        "firefox_mb",
        "agents",
        "runs",
        "data_mb",
        "log_problems",
        "hitches",
        "transitions",
    ]
    with out.open("a", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        if f.tell() == 0:
            writer.writeheader()
        while time.monotonic() < end:
            window_end: float = time.monotonic() + every
            phase: str | None = None
            try:
                async with websockets_connect(url) as ws:
                    await ws.recv()  # config
                    last: tuple[float, int] | None = None
                    while (now := time.monotonic()) < window_end:
                        message: dict[str, Any] = json.loads(
                            await asyncio.wait_for(ws.recv(), timeout=5)
                        )
                        if message.get("type") != "state":
                            continue
                        new: str = message.get("phase", "-")
                        if last is not None:
                            ms: float = (now - last[0]) * 1000
                            window.intervals.append(ms)
                            window.frames += message["frame"] - last[1]
                            if ms > HITCH_MS:
                                window.hitches.append(f"{ms:.0f}ms@{new}")
                        if phase is not None and new != phase:
                            window.transitions.append(f"{phase}>{new}")
                        phase = new
                        last = (now, message["frame"])
            except (OSError, TimeoutError, asyncio.TimeoutError) as error:
                window.hitches.append(f"connection:{type(error).__name__}")
                await asyncio.sleep(max(0.0, window_end - time.monotonic()))
            body: dict[str, Any] = health(health_url)
            row: dict[str, Any] = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": body.get("status"),
                "phase": phase,
                "ticks": body.get("ticks"),
                "overruns": body.get("overruns"),
                "clients": body.get("clients"),
                **window.row(),
                **docker_stats(containers),
                "firefox_mb": firefox_rss_mb(),
                **volume_figures(containers[0]),
                "log_problems": log_problems(containers[0], f"{int(every)}s"),
            }
            writer.writerow(row)
            f.flush()
            rows += 1
            window = Window()
    return [f"{rows} rows written to {out}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_seed = sub.add_parser(
        "seed", help="store N agents into an empty gallery (in the image)"
    )
    p_seed.add_argument("count", type=int)
    p_seed.add_argument("--data-dir", type=Path, default=Path("/data"))
    p_visit = sub.add_parser("visit", help="scripted visitors, one after another")
    p_visit.add_argument("--url", default=DEFAULT_URL)
    p_visit.add_argument("--hours", type=float, default=8.0)
    p_visit.add_argument("--seed", type=int, default=0)
    p_sample = sub.add_parser("sample", help="one CSV row per interval; only listens")
    p_sample.add_argument("--url", default=DEFAULT_URL)
    p_sample.add_argument("--health", default=DEFAULT_HEALTH)
    p_sample.add_argument("--out", type=Path, required=True)
    p_sample.add_argument("--hours", type=float, default=8.0)
    p_sample.add_argument("--every", type=float, default=60.0, help="seconds per row")
    p_sample.add_argument("--crane", default=CRANE_CONTAINER)
    p_sample.add_argument("--viz", default=VIZ_CONTAINER)
    args = parser.parse_args()
    lines: list[str]
    if args.command == "seed":
        lines = seed(args.data_dir, args.count)
    elif args.command == "visit":
        lines = asyncio.run(visit(args.url, args.hours, random.Random(args.seed)))
    else:
        lines = asyncio.run(
            sample(
                args.url,
                args.health,
                args.out,
                args.hours,
                args.every,
                (args.crane, args.viz),
            )
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
