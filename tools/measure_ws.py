#!/usr/bin/env python3
"""Measure the live server from the outside, as a WebSocket client
(DECISIONS.md §8, §12). Runs in the venv, beside a running server:

    pdm run python tools/measure_ws.py timing --seconds 30
    pdm run python tools/measure_ws.py roundtrip --seconds 10
    pdm run python tools/measure_ws.py trace --seconds 20

`timing` listens only: message rate, intervals, simulated against wall-clock
time. `roundtrip` sends distinctive commands and times until a state shows
them; it needs every other browser closed, because any open page sends its
own command every frame and the latest message wins (§5), and a server
started with `--free`, because in the stand's steps an agent drives while
the stand waits (§18). `trace` listens
only and prints the command the server applies, per time bin: what a browser
and its gamepad deliver, judged at the server and not on the screen.

Every figure recorded from this tool names the machine it ran on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import websockets

from teach_by_showing.live import InputParams

DEFAULT_URL: str = "ws://127.0.0.1:8081/ws"


@dataclass(frozen=True)
class Sample:
    """One state message: when it arrived and what it said."""

    arrival: float  # monotonic seconds
    frame: int
    command: float


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile, q in 0..100; enough for a few thousand values."""
    ordered: list[float] = sorted(values)
    rank: int = max(0, min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1))))
    return ordered[rank]


def timing_report(samples: Sequence[Sample], hz: float) -> list[str]:
    span: float = samples[-1].arrival - samples[0].arrival
    intervals: list[float] = [
        (b.arrival - a.arrival) * 1000 for a, b in zip(samples, samples[1:], strict=False)
    ]
    frames: int = samples[-1].frame - samples[0].frame
    skipped: int = frames - (len(samples) - 1)
    drift_ms: float = (frames / hz - span) * 1000
    rate: float = (len(samples) - 1) / span
    p1: float = percentile(intervals, 1)
    p99: float = percentile(intervals, 99)
    return [
        f"state messages   {len(samples)} in {span:.1f} s, {rate:.2f} Hz,"
        f" {skipped} frames skipped",
        f"interval         median {statistics.median(intervals):.2f} ms,"
        f" 1st-99th percentile {p1:.2f}-{p99:.2f} ms,"
        f" max {max(intervals):.2f} ms",
        f"sim vs wall      {drift_ms:+.1f} ms over {span:.1f} s",
    ]


def trace_report(samples: Sequence[Sample], bin_seconds: float) -> list[str]:
    """Min and max of the applied command per bin. A browser that stops
    sending shows as 0 after the server's input timeout (§8)."""
    t0: float = samples[0].arrival
    bins: dict[int, list[float]] = {}
    for s in samples:
        bins.setdefault(int((s.arrival - t0) / bin_seconds), []).append(s.command)
    return [
        f"{k * bin_seconds:6.1f} s  {min(v):+.2f} .. {max(v):+.2f}"
        for k, v in sorted(bins.items())
    ]


async def listen(url: str, seconds: float) -> tuple[float, list[Sample]]:
    samples: list[Sample] = []
    async with websockets.connect(url) as ws:
        config: dict[str, Any] = json.loads(await ws.recv())
        end: float = time.monotonic() + seconds
        while (now := time.monotonic()) < end:
            message: dict[str, Any] = json.loads(await ws.recv())
            if message["type"] == "state":
                samples.append(Sample(now, message["frame"], message["command"]))
    return float(config["hz"]), samples


def expected_command(config: dict[str, Any], u: float) -> float:
    """The command a state shows for the deflection `u`: the server shapes
    it with its input curve (§14), so the tool shapes it the same way, with
    the server's own parameters from the config message."""
    if config.get("mode") == "stand":
        raise SystemExit(
            "roundtrip: the server runs the stand's steps, where an agent drives"
            " while the stand waits; start it with --free (DECISIONS.md §18)"
        )
    return InputParams(**config.get("input", {})).shape(u)


async def roundtrip(url: str, seconds: float) -> list[str]:
    """Alternate between two commands every 0.25 s and time each until the
    first state that carries it. The commands are distinctive, so a stale
    state is never mistaken for the answer. Each send waits a random part of
    a frame first: a client in step with the server's tick would otherwise
    always hit the same phase and report a best or worst case, not the
    spread a stick sees."""
    rng: random.Random = random.Random(0)
    delays: list[float] = []
    async with websockets.connect(url) as ws:
        config: dict[str, Any] = json.loads(await ws.recv())
        end: float = time.monotonic() + seconds
        values: tuple[float, float] = (0.3125, -0.4375)
        i: int = 0
        while time.monotonic() < end:
            u: float = values[i % 2]
            await asyncio.sleep(rng.uniform(0, 1 / 60))
            sent: float = time.monotonic()
            await ws.send(json.dumps({"type": "input", "u": u}))
            shown: float = expected_command(config, u)
            while True:
                message: dict[str, Any] = json.loads(await ws.recv())
                if message["type"] == "state" and message["command"] == shown:
                    delays.append((time.monotonic() - sent) * 1000)
                    break
            i += 1
            # Keep sending inside the gap, or the input timeout zeroes it.
            for _ in range(15):
                await ws.send(json.dumps({"type": "input", "u": u}))
                await asyncio.sleep(1 / 60)
        await ws.send(json.dumps({"type": "input", "u": 0.0}))
    return [
        f"round trip       {len(delays)} commands, median {statistics.median(delays):.1f} ms,"
        f" p95 {percentile(delays, 95):.1f} ms, max {max(delays):.1f} ms"
    ]


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument("mode", choices=["timing", "roundtrip", "trace"])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--bin", type=float, default=0.5, help="trace: seconds per line")
    args: argparse.Namespace = parser.parse_args()

    lines: list[str]
    if args.mode == "roundtrip":
        lines = asyncio.run(roundtrip(args.url, args.seconds))
    else:
        hz, samples = asyncio.run(listen(args.url, args.seconds))
        lines = (
            timing_report(samples, hz)
            if args.mode == "timing"
            else trace_report(samples, args.bin)
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
