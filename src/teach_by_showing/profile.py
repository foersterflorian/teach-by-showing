"""The style profile: the same few figures for every driver, human or agent,
so that they can be set side by side (concept §6.2, DECISIONS.md §22).

Per transport: seconds to calm, the largest swing, the residual swing from
reaching the target zone on, and the stick travel. They are counted frame by
frame in a Tally, batched like Transport, so that a clone driving many
transports, the replay driving one and a recorded drive played back are
counted by the same code. The module knows nothing about learning, HTTP or
the frontend.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from teach_by_showing.crane import (
    SIM_HZ,
    CraneParams,
    CraneState,
    FloatArray,
    swing_amplitude,
)
from teach_by_showing.recording import Run
from teach_by_showing.task import BoolArray, IntArray, TaskParams, Transport, advance


@dataclass(frozen=True, slots=True)
class Tally:
    """Running sums for n transports. Never mutated; update() returns a new
    one. Only frames in which a transport is still running count."""

    target: FloatArray = field(repr=False)
    max_swing: FloatArray = field(repr=False)  # rad
    # Swing amplitude summed over the frames since the trolley first came
    # within position_tolerance of the target, and the number of those frames.
    rest_sum: FloatArray = field(repr=False)  # rad
    rest_frames: IntArray = field(repr=False)
    reached_zone: BoolArray = field(repr=False)
    stick_travel: FloatArray = field(repr=False)
    # Every transport begins with the stick at the centre, so that a first
    # deflection counts and a command held over from the pause does not
    # depend on what came before.
    last_command: FloatArray = field(repr=False)

    @classmethod
    def begin(cls, target: FloatArray) -> Self:
        target = np.asarray(target, dtype=np.float64).copy()
        zeros: FloatArray = np.zeros_like(target)
        return cls(
            target,
            zeros,
            zeros,
            np.zeros(target.shape[0], dtype=np.int64),
            np.zeros(target.shape[0], dtype=np.bool_),
            zeros,
            zeros,
        )

    def update(
        self,
        command: FloatArray,
        state: CraneState,
        running: BoolArray,
        crane: CraneParams,
        task: TaskParams,
    ) -> Tally:
        """Account for the frame in which `command` was applied and led to
        `state`; `running` says which transports were still under way."""
        amp: FloatArray = swing_amplitude(state, crane)
        zone: BoolArray = running & (
            self.reached_zone | (np.abs(state.x - self.target) <= task.position_tolerance)
        )
        return Tally(
            self.target,
            np.where(running, np.maximum(self.max_swing, amp), self.max_swing),
            np.where(zone, self.rest_sum + amp, self.rest_sum),
            np.where(zone, self.rest_frames + 1, self.rest_frames),
            self.reached_zone | zone,
            np.where(
                running,
                self.stick_travel + np.abs(command - self.last_command),
                self.stick_travel,
            ),
            np.where(running, command, self.last_command),
        )


@dataclass(frozen=True, slots=True)
class TransportProfile:
    """The figures of one transport; seconds is None if it was not reached."""

    seconds: float | None
    max_swing_deg: float
    # None if the trolley never came within reach of the target.
    rest_deg: float | None
    stick_travel: float

    @property
    def reached(self) -> bool:
        return self.seconds is not None


def profiles(transport: Transport, tally: Tally) -> list[TransportProfile]:
    """One profile per transport of the batch."""
    frames: list[int] = transport.duration_frames().tolist()
    return [
        TransportProfile(
            frames[i] / SIM_HZ if frames[i] >= 0 else None,
            math.degrees(float(tally.max_swing[i])),
            math.degrees(float(tally.rest_sum[i]) / int(tally.rest_frames[i]))
            if tally.rest_frames[i] > 0
            else None,
            float(tally.stick_travel[i]),
        )
        for i in range(len(frames))
    ]


def run_profiles(run: Run) -> list[TransportProfile]:
    """The teacher's transports, counted row by row with the same advance()
    and Tally as a clone's."""
    command: FloatArray = run.column("command")
    result: list[TransportProfile] = []
    for i in range(run.transports):
        rows: IntArray = run.transport_rows(i)
        transport: Transport = Transport.begin(run.column("target")[rows[:1]], 0)
        tally: Tally = Tally.begin(transport.target)
        for k, row in enumerate(rows, start=1):
            state: CraneState = run.states(np.asarray([row]), k)
            running: BoolArray = ~transport.done(k - 1, run.task)
            tally = tally.update(command[row : row + 1], state, running, run.crane, run.task)
            transport = advance(transport, state, run.crane, run.task)
        result += profiles(transport, tally)
    return result


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """Means over the reached transports of a list; None where none was
    reached. A transport not reached is counted, not averaged."""

    transports: int
    reached: int
    seconds: float | None
    max_swing_deg: float | None
    rest_deg: float | None
    stick_travel: float | None


def summarize(items: list[TransportProfile]) -> ProfileSummary:
    reached: list[TransportProfile] = [p for p in items if p.reached]

    def mean(values: list[float | None]) -> float | None:
        present: list[float] = [v for v in values if v is not None]
        return statistics.fmean(present) if present else None

    return ProfileSummary(
        len(items),
        len(reached),
        mean([p.seconds for p in reached]),
        mean([p.max_swing_deg for p in reached]),
        mean([p.rest_deg for p in reached]),
        mean([p.stick_travel for p in reached]),
    )
