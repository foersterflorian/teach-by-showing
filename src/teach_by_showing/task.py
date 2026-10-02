"""The transport task, level 1: bring the load from a start to a target and let
it hang calm there (DECISIONS.md §7).

Batched like the crane: one transport per crane. The random generator is
always passed in, never global, so that a seed reproduces a run.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from teach_by_showing.crane import (
    SIM_HZ,
    CraneParams,
    CraneState,
    FloatArray,
    swing_amplitude,
)

type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TaskParams:
    """When a transport counts as done. First set for the spike."""

    # Start and target keep this distance from the end stops, so that the
    # drive's early braking (crane.step) never decides the outcome.
    margin: float = 0.5
    # Shortest transport. Too short and there is no time to build up a swing
    # worth controlling.
    min_distance: float = 4.0
    position_tolerance: float = 0.15  # m, trolley to target
    speed_tolerance: float = 0.05  # m/s
    swing_tolerance: float = math.radians(2.0)  # of crane.swing_amplitude
    calm_seconds: float = 1.0
    time_limit: float = 60.0

    @property
    def calm_frames(self) -> int:
        return round(self.calm_seconds * SIM_HZ)

    @property
    def limit_frames(self) -> int:
        return round(self.time_limit * SIM_HZ)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Self:
        return cls(**data)


def sample_transports(
    rng: np.random.Generator, crane: CraneParams, task: TaskParams, n: int
) -> tuple[FloatArray, FloatArray]:
    """n pairs of start and target, uniform over the track and at least
    min_distance apart, in either direction."""
    lo: float = crane.track_min + task.margin
    hi: float = crane.track_max - task.margin
    d: float = task.min_distance
    if hi - lo < 2.0 * d:
        # Otherwise a start near the middle has no admissible target.
        raise ValueError(f"track {lo}..{hi} too short for min_distance {d}")

    start: FloatArray = rng.uniform(lo, hi, size=n)
    # Admissible targets form [lo, start − d] and [start + d, hi]; draw a point
    # on their joint length and map it back.
    left: FloatArray = np.maximum(start - d - lo, 0.0)
    right: FloatArray = np.maximum(hi - start - d, 0.0)
    r: FloatArray = rng.uniform(0.0, 1.0, size=n) * (left + right)
    target: FloatArray = np.where(r < left, lo + r, start + d + (r - left))
    return start, target


def is_calm(
    state: CraneState, target: FloatArray, crane: CraneParams, task: TaskParams
) -> BoolArray:
    """Trolley stands at the target and the load hangs still."""
    return (
        (np.abs(state.x - target) <= task.position_tolerance)
        & (np.abs(state.v) <= task.speed_tolerance)
        & (swing_amplitude(state, crane) <= task.swing_tolerance)
    )


@dataclass(frozen=True, slots=True)
class Transport:
    """Progress of n transports. Never mutated; advance() returns a new one."""

    start_frame: int
    target: FloatArray = field(repr=False)
    # Consecutive calm frames so far; frozen once the transport has succeeded.
    calm_frames: IntArray = field(repr=False)
    # Frame at which the load had hung calm for calm_seconds; −1 until then.
    success_frame: IntArray = field(repr=False)

    @classmethod
    def begin(cls, target: FloatArray, start_frame: int) -> Self:
        target = np.asarray(target, dtype=np.float64).copy()
        n: int = int(target.shape[0])
        return cls(
            start_frame, target, np.zeros(n, dtype=np.int64), np.full(n, -1, dtype=np.int64)
        )

    @property
    def succeeded(self) -> BoolArray:
        return self.success_frame >= 0

    def timed_out(self, frame: int, task: TaskParams) -> BoolArray:
        return ~self.succeeded & (frame - self.start_frame >= task.limit_frames)

    def done(self, frame: int, task: TaskParams) -> BoolArray:
        return self.succeeded | self.timed_out(frame, task)

    def duration_frames(self) -> IntArray:
        """Frames from start until the load hung calm, the calm second
        included; −1 for transports that have not succeeded."""
        return np.where(self.succeeded, self.success_frame - self.start_frame, -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "target": self.target.tolist(),
            "calm_frames": self.calm_frames.tolist(),
            "success_frame": self.success_frame.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            int(data["start_frame"]),
            np.asarray(data["target"], dtype=np.float64),
            np.asarray(data["calm_frames"], dtype=np.int64),
            np.asarray(data["success_frame"], dtype=np.int64),
        )


def advance(
    transport: Transport, state: CraneState, crane: CraneParams, task: TaskParams
) -> Transport:
    """Account for one new frame `state`. Finished transports, succeeded or
    timed out, no longer change."""
    active: BoolArray = ~transport.done(state.frame, task)
    calm: BoolArray = is_calm(state, transport.target, crane, task)
    calm_frames: IntArray = np.where(
        active, np.where(calm, transport.calm_frames + 1, 0), transport.calm_frames
    )
    newly_done: BoolArray = active & (calm_frames >= task.calm_frames)
    success_frame: IntArray = np.where(newly_done, state.frame, transport.success_frame)
    return Transport(transport.start_frame, transport.target, calm_frames, success_frame)
