"""Scripted teachers standing in for people: they drive the live crane and
are recorded the same way a visitor is (DECISIONS.md §13, §15)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from teach_by_showing.live import InputParams, LiveSession, RecordingRequest
from teach_by_showing.recording import Run

type Driver = Callable[[LiveSession], float]


def cautious(s: LiveSession) -> float:
    """Slow, with a gentle approach and little swing."""
    gap: float = float(s.transport.target[0] - s.state.x[0])
    return float(np.clip(0.25 * gap, -0.35, 0.35) + 1.0 * float(s.state.theta[0]))


def quick(s: LiveSession) -> float:
    """Full speed until close, then damps the swing it has built up."""
    gap: float = float(s.transport.target[0] - s.state.x[0])
    return float(np.clip(1.0 * gap, -1.0, 1.0) + 2.0 * float(s.state.theta[0]))


def teach(driver: Driver, transports: int = 3, seed: int = 0) -> Run:
    """A run of `transports` transports by `driver`, with a linear stick so
    that the scripted command is the command applied."""
    s: LiveSession = LiveSession.start(
        np.random.default_rng(seed), input=InputParams(expo=0.0)
    )
    runs: list[Run] = []
    s.request_recording(
        RecordingRequest(driver.__name__, seed, "test", transports, runs.append)
    )
    while not runs:
        s.set_input(max(-1.0, min(1.0, driver(s))))
        s.tick()
        if s.state.frame > transports * 2 * s.task.limit_frames:
            raise AssertionError(f"{driver.__name__} did not finish {transports} transports")
    return runs[0]
