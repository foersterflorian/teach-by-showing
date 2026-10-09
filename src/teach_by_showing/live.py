"""The live crane on the stand: one crane driven by the visitor's input, one
transport after another, paced at SIM_HZ (DECISIONS.md §8).

This is the engine between the simulation core and the web layer. It knows
nothing about HTTP; api.py calls into it and nothing else. Everything here
counts in frames, never in wall-clock time, so a session is reproducible from
its seed and its input sequence.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any, Final, Self

import numpy as np

from teach_by_showing.crane import DT, SIM_HZ, CraneParams, CraneState, FloatArray, step
from teach_by_showing.recording import Run
from teach_by_showing.task import (
    TaskParams,
    Transport,
    advance,
    sample_targets,
    sample_transports,
)

# Without fresh input for this long the command falls back to zero and the
# trolley brakes: a closed browser tab, a lost connection or an unplugged
# gamepad must never leave the crane driving. Four frames at the browser's
# own 60 Hz would do; 0.25 s tolerates a hiccup.
INPUT_TIMEOUT_FRAMES: Final[int] = SIM_HZ // 4
# Between two transports, long enough to see the result.
PAUSE_FRAMES: Final[int] = round(1.5 * SIM_HZ)
# If the loop falls further behind than this, it gives up the lost time
# instead of catching up with a burst of steps the viewer would see as a jump.
MAX_LAG: Final[float] = 0.1


def parse_command(value: object) -> float | None:
    """A command from untrusted input: a finite number, clipped to −1..1, or
    None if it is not one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    u: float = float(value)
    if not math.isfinite(u):
        return None
    return min(1.0, max(-1.0, u))


@dataclass(frozen=True, slots=True)
class InputParams:
    """How the stick's deflection becomes a command (concept §15 item 6).
    Here, not in the browser, so that the frontend holds no logic (§3)."""

    # Blend of a linear and a cubic curve, 0 (linear) to 1 (cubic): small
    # deflections give finer commands, full deflection stays full speed.
    # Applies to the human's input only; an agent commands directly.
    # 0.7 chosen on the gamepad on 2026-10-09 (DECISIONS.md §14); fine-tuning
    # is open.
    expo: float = 0.7

    def shape(self, u: float) -> float:
        return (1.0 - self.expo) * u + self.expo * u**3

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Settings:
    """Crane, task and input parameters as one file, so that candidate sets
    can be compared on the gamepad without editing code (DECISIONS.md §14).
    A file names only what differs from the defaults."""

    crane: CraneParams
    task: TaskParams
    input: InputParams

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        unknown: set[str] = set(data) - {"note", "crane", "task", "input"}
        if unknown:
            raise ValueError(f"unknown settings {sorted(unknown)}")
        # replace() refuses unknown field names, so a typo fails loudly.
        return cls(
            replace(CraneParams(), **data.get("crane", {})),
            replace(TaskParams(), **data.get("task", {})),
            replace(InputParams(), **data.get("input", {})),
        )


@dataclass(frozen=True, slots=True)
class RecordingRequest:
    """A recording asked for but not yet begun (DECISIONS.md §13). `created`
    comes from the caller: this module counts frames, not wall-clock time."""

    label: str
    seed: int | None
    created: str
    transports: int
    on_done: Callable[[Run], None]


@dataclass(slots=True)
class LiveSession:
    """One crane on the stand. Mutable on purpose: it is the single owner of
    the live state, and only the pacing loop calls tick()."""

    rng: np.random.Generator
    crane: CraneParams
    task: TaskParams
    state: CraneState
    transport: Transport
    input: InputParams = InputParams()
    command: float = 0.0
    input_frame: int = -INPUT_TIMEOUT_FRAMES - 1
    pause_left: int = 0
    transports_done: int = 0
    # Asked for by request_recording(); `recording` is None until the
    # teacher's first deflection begins it.
    request: RecordingRequest | None = None
    recording: Run | None = None
    # transports_done when the recording began, so that its own transports
    # count from 0.
    recording_base: int = 0

    @classmethod
    def start(
        cls,
        rng: np.random.Generator,
        crane: CraneParams | None = None,
        task: TaskParams | None = None,
        input: InputParams | None = None,
    ) -> LiveSession:
        crane = crane or CraneParams()
        task = task or TaskParams()
        start, target = sample_transports(rng, crane, task, 1)
        state: CraneState = CraneState.at_rest(start)
        transport: Transport = Transport.begin(target, state.frame)
        return cls(rng, crane, task, state, transport, input or InputParams())

    def set_input(self, deflection: float) -> None:
        """The stick's deflection, −1..1, shaped by the input curve."""
        self.command = self.input.shape(deflection)
        self.input_frame = self.state.frame

    @property
    def effective_command(self) -> float:
        if self.state.frame - self.input_frame > INPUT_TIMEOUT_FRAMES:
            return 0.0
        return self.command

    def request_recording(self, request: RecordingRequest) -> None:
        """Record the next `request.transports` transports, beginning with the
        teacher's first stick deflection."""
        self.request = request
        self.recording = None

    def tick(self) -> None:
        """Advance one frame. After a transport has finished, the crane keeps
        responding during a short pause; then the next target is drawn from
        wherever the trolley stands."""
        command: float = self.effective_command
        if (
            self.request is not None
            and self.recording is None
            and command != 0.0
            and not self.transport.done(self.state.frame, self.task)[0]
        ):
            self._begin_recording(self.request)
        u: FloatArray = np.array([command])
        target: float = float(self.transport.target[0])
        # Judged before the step: a transport that times out in this frame
        # is still running in it, so that it is counted and paused after
        # like one that succeeds.
        running: bool = not self.transport.done(self.state.frame, self.task)[0]
        self.state = step(self.state, u, self.crane)
        # A pause belongs to the transport it follows, which is already counted.
        index: int = self.transports_done - self.recording_base - (0 if running else 1)
        if running:
            self.transport = advance(self.transport, self.state, self.crane, self.task)
            if self.transport.done(self.state.frame, self.task)[0]:
                self.pause_left = PAUSE_FRAMES
                self.transports_done += 1
        elif self.pause_left > 0:
            self.pause_left -= 1
        else:
            new_target: FloatArray = sample_targets(
                self.rng, self.state.x, self.crane, self.task
            )
            self.transport = Transport.begin(new_target, self.state.frame)
        if self.recording is not None:
            self._record(command, target, index, running)

    def _begin_recording(self, request: RecordingRequest) -> None:
        # The transport's clock restarts at the first deflection: the time a
        # teacher needs to pick up the gamepad is not part of the drive, and
        # frames of standing still before it would teach "do nothing".
        self.transport = Transport.begin(self.transport.target, self.state.frame)
        self.recording = Run(
            request.label, request.seed, request.created, self.crane, self.task, self.state
        )
        self.recording_base = self.transports_done

    def _record(self, command: float, target: float, index: int, running: bool) -> None:
        run: Run | None = self.recording
        request: RecordingRequest | None = self.request
        assert run is not None and request is not None
        run.append(command, self.state, target, index, running)
        # Ends with the frame that finished the last transport; its pause is
        # not part of the demonstration.
        if self.transports_done - self.recording_base >= request.transports:
            self.recording = None
            self.request = None
            request.on_done(run)

    def config_message(self) -> dict[str, Any]:
        """Sent once per connection: everything the frontend needs to draw to
        scale, so that it holds no physical constants of its own."""
        return {
            "type": "config",
            "hz": SIM_HZ,
            "crane": self.crane.to_dict(),
            "task": self.task.to_dict(),
            "input": self.input.to_dict(),
        }

    def state_message(self) -> dict[str, Any]:
        """Sent every frame. Lists, not scalars, because the core is batched
        and later messages carry ghosts and swarms in the same shape."""
        s: CraneState = self.state
        tr: Transport = self.transport
        frame: int = s.frame
        result: str = (
            "success"
            if tr.succeeded[0]
            else "timeout"
            if tr.timed_out(frame, self.task)[0]
            else "running"
        )
        elapsed: int = (
            int(tr.duration_frames()[0]) if tr.succeeded[0] else frame - tr.start_frame
        )
        return {
            "type": "state",
            "frame": frame,
            "x": s.x.tolist(),
            "theta": s.theta.tolist(),
            "target": tr.target.tolist(),
            "command": self.effective_command,
            "calm": min(1.0, int(tr.calm_frames[0]) / self.task.calm_frames),
            "result": result,
            "elapsed": elapsed * DT,
            "transports": self.transports_done,
        }


@dataclass(slots=True)
class LoopStats:
    """What the health check reads. Wall-clock times come from the loop's own
    monotonic clock."""

    last_tick: float = -math.inf
    ticks: int = 0
    # Times the loop fell more than MAX_LAG behind and dropped time.
    overruns: int = 0


async def run_paced(
    session: LiveSession,
    publish: Callable[[dict[str, Any]], None],
    stats: LoopStats,
) -> None:
    """Tick the session at SIM_HZ against absolute deadlines, so that
    sleep inaccuracy does not accumulate into drift. `publish` must not
    block; it is called once per frame with the state message."""
    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    deadline: float = loop.time()
    while True:
        session.tick()
        publish(session.state_message())
        now: float = loop.time()
        stats.last_tick = now
        stats.ticks += 1
        deadline += DT
        if now - deadline > MAX_LAG:
            stats.overruns += 1
            deadline = now
        await asyncio.sleep(max(0.0, deadline - now))
