"""Crane physics, level 1: a trolley on a girder with a load swinging on a rope
of fixed length (DECISIONS.md §6).

Every quantity carries a leading batch axis of n cranes. A single crane is the
case n = 1, the evolution swarm is n = 100; there is no second implementation.
The module knows nothing about tasks, learning, HTTP or the frontend.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

SIM_HZ: Final[int] = 60
SUBSTEPS: Final[int] = 4
DT: Final[float] = 1.0 / SIM_HZ


@dataclass(frozen=True, slots=True)
class CraneParams:
    """Physical parameters. First set for the spike; tuned on the gamepad
    (concept §15 item 7)."""

    gravity: float = 9.81
    # g·T²/(4π²) for a swing period T of 2.5 s: long enough to see the swing,
    # short enough for transports of 20–30 s.
    rope_length: float = 1.55
    # Rope and air friction, 1/s. Amplitude halves in about 2·ln 2 / c ≈ 28 s,
    # so an uncontrolled load keeps swinging for the whole transport.
    damping: float = 0.05
    track_min: float = 0.0
    track_max: float = 10.0
    # Faster and harder than a real crane (about 0.3–0.5 m/s²), on purpose:
    # the naive driver must leave a swing a visitor sees at a glance. Set B of
    # the comparison in DECISIONS.md §6; fine-tuning remains open.
    max_speed: float = 1.2
    max_accel: float = 1.5

    @property
    def omega0(self) -> float:
        """Small-angle angular frequency of the pendulum, rad/s."""
        return math.sqrt(self.gravity / self.rope_length)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> Self:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CraneState:
    """State of n cranes at one instant. Arrays are never mutated in place;
    step() returns a new state, so recorded states stay valid."""

    # Frames since the start, not seconds: an integer count cannot drift the
    # way repeated addition of 1/60 does.
    frame: int
    x: FloatArray = field(repr=False)  # trolley position, m
    v: FloatArray = field(repr=False)  # trolley velocity, m/s
    theta: FloatArray = field(repr=False)  # rope angle to the vertical, rad
    omega: FloatArray = field(repr=False)  # dθ/dt, rad/s

    @property
    def t(self) -> float:
        """Simulated time in seconds."""
        return self.frame * DT

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @classmethod
    def at_rest(cls, x: FloatArray) -> Self:
        """Cranes standing still at positions x with the load hanging calm."""
        x = np.asarray(x, dtype=np.float64).copy()
        return cls(0, x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x))

    def load_position(self, params: CraneParams) -> tuple[FloatArray, FloatArray]:
        """Horizontal and vertical load position relative to the girder; the
        vertical one is negative, since the load hangs below the trolley."""
        length: float = params.rope_length
        return self.x + length * np.sin(self.theta), -length * np.cos(self.theta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "x": self.x.tolist(),
            "v": self.v.tolist(),
            "theta": self.theta.tolist(),
            "omega": self.omega.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        def arr(key: str) -> FloatArray:
            return np.asarray(data[key], dtype=np.float64)

        return cls(int(data["frame"]), arr("x"), arr("v"), arr("theta"), arr("omega"))


def swing_amplitude(state: CraneState, params: CraneParams) -> FloatArray:
    """Amplitude the load would swing with if the trolley stood still from now
    on, in rad. Unlike |θ| it does not drop to zero as the load passes the
    vertical, so it measures how calm the load really is."""
    return np.hypot(state.theta, state.omega / params.omega0)


def _stoppable_speed(distance: FloatArray, params: CraneParams, h: float) -> FloatArray:
    """Highest speed from which the drive, braking at max_accel, still stops
    within `distance`, counted in substeps of length h.

    The continuous answer, sqrt(2·a·d), is not enough: within a substep the
    trolley moves after the speed is set, so near the end it would arrive too
    fast and be stopped by the hard clip, a jolt the drive cannot produce.
    Requiring v²/(2a) + v·h/2 ≤ d keeps braking feasible from every substep
    to the next. That bound is exact only when v is a whole multiple of a·h;
    otherwise the trolley rolls up to a·h²/8 farther, so the distance is
    shortened by a·h², which also absorbs rounding (test_crane.py)."""
    a: float = params.max_accel
    d: FloatArray = np.maximum(distance - a * h * h, 0.0)
    return a * (np.sqrt(h * h / 4.0 + 2.0 * d / a) - h / 2.0)


def step(state: CraneState, command: FloatArray, params: CraneParams) -> CraneState:
    """Advance all cranes by one frame of 1/SIM_HZ.

    `command` is the target trolley velocity as a fraction of max_speed, one
    value per crane, clipped to −1..1. The drive follows it with limited
    acceleration, like a crane on a frequency converter, and slows down early
    enough to stop at the end of the track. This is the safety layer under
    every driver, human or network: nothing it commands can leave the track
    or exceed the drive's limits.
    """
    h: float = DT / SUBSTEPS
    p: CraneParams = params
    u: FloatArray = np.clip(np.asarray(command, dtype=np.float64), -1.0, 1.0)
    x, v, theta, omega = state.x, state.v, state.theta, state.omega

    for _ in range(SUBSTEPS):
        room_right: FloatArray = _stoppable_speed(np.maximum(p.track_max - x, 0.0), p, h)
        room_left: FloatArray = _stoppable_speed(np.maximum(x - p.track_min, 0.0), p, h)
        v_target: FloatArray = np.clip(u * p.max_speed, -room_left, room_right)
        accel: FloatArray = np.clip((v_target - v) / h, -p.max_accel, p.max_accel)

        # Semi-implicit Euler: velocities first, positions from the new
        # velocities. Keeps the pendulum's energy bounded where explicit Euler
        # would pump it up (test_crane.py).
        v = v + accel * h
        x = x + v * h
        alpha: FloatArray = (
            -(p.gravity * np.sin(theta) + accel * np.cos(theta)) / p.rope_length
            - p.damping * omega
        )
        omega = omega + alpha * h
        theta = theta + omega * h

        # Last guard against rounding at the end stops.
        at_stop: NDArray[np.bool_] = (x < p.track_min) | (x > p.track_max)
        x = np.clip(x, p.track_min, p.track_max)
        v = np.where(at_stop, 0.0, v)

    return CraneState(state.frame + 1, x, v, theta, omega)
