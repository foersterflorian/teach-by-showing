"""Ghost replay: the clone drives its teacher's transports while the
teacher's recorded drive plays alongside, translucent (DECISIONS.md §17).

The agent is simulated with the live crane's step() and advance(); the ghost
is not simulated at all but plays the recorded rows, so it shows exactly
what the teacher did. Both begin each transport in the state the teacher
began it in, with the same target. After the teacher's transports the agent
may drive new targets of its own, without a ghost (§20). The module knows
nothing about HTTP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from teach_by_showing.cloning import Policy
from teach_by_showing.crane import SIM_HZ, CraneState, FloatArray, step
from teach_by_showing.features import features
from teach_by_showing.live import PAUSE_FRAMES
from teach_by_showing.naive import naive_command
from teach_by_showing.recording import Run
from teach_by_showing.task import Transport, advance, sample_targets

HEADLINE: Final[str] = "Ihr Agent fährt – Ihre eigene Fahrt läuft als Schatten mit"
HEADLINE_NEW: Final[str] = "Ihr Agent fährt ein neues Ziel – das hat er nie gesehen"
TEACHER_SELF: Final[str] = "Ihre Fahrt"
# The swing curve's full height. A quick teacher swings about 24°, the naive
# driver up to about 20° (§16, §19); beyond this the curve is clipped.
CURVE_SWING_DEG: Final[float] = 30.0
# The time axis is never shorter than this and grows in steps of it, so that
# it does not rescale with every frame while a drive runs long.
CURVE_STEP_SECONDS: Final[float] = 5.0
CURVE_LABELS: Final[dict[str, str]] = {
    "v": "Geschwindigkeit",
    "theta": "Pendelwinkel",
    "command": "Stellgröße",
}


def _seconds(frames: int) -> str:
    return f"{frames / SIM_HZ:.1f}".replace(".", ",") + " s"


@dataclass(slots=True)
class ReplaySession:
    """The teacher's transports one after another, each driven by the agent
    and played by the ghost, then from the start again. Same interface as
    live.LiveSession, so the pacing loop and the web layer serve it alike."""

    run: Run
    policy: Policy
    begins: CraneState
    targets: FloatArray
    # Frames since the session began. Each transport restarts from a
    # recorded state, but the frame count runs on, so that the browser's
    # interpolation (§9) never sees time go backwards.
    frame: int = 0
    index: int = 0
    state: CraneState | None = None
    transport: Transport | None = None
    rows: NDArray[np.int64] | None = None
    ghost_played: int = 0
    command: float = 0.0
    pause_left: int = 0
    transports_done: int = 0
    # Whose drive the ghost plays, as the readouts and the banner name it:
    # the visitor's own, or an earlier teacher's in the stand's idle step.
    teacher: str = TEACHER_SELF
    # The naive driver beside them, from the same start to the same target
    # (§19). It does not hold up the next transport: it rarely comes to rest
    # within the time limit, and the stand would wait a minute for it.
    with_naive: bool = False
    # Whether the teacher's drive is drawn. The ghost still plays, unseen,
    # so that the transports keep the teacher's pace (§19).
    show_ghost: bool = True
    naive: CraneState | None = None
    naive_transport: Transport | None = None
    # After the teacher's transports, this many new targets for the agent
    # alone, drawn with `rng` from wherever it stands (§20).
    new_targets: int = 0
    rng: np.random.Generator | None = None
    # Whether states carry the curves over time drawn under the scene (§20).
    show_curves: bool = False
    # Counts every transport begun, so that the frontend starts its curves
    # afresh even when the loop returns to the same index.
    segment: int = 0

    @classmethod
    def start(
        cls,
        run: Run,
        policy: Policy,
        teacher: str = TEACHER_SELF,
        with_naive: bool = False,
        show_ghost: bool = True,
        new_targets: int = 0,
        rng: np.random.Generator | None = None,
        show_curves: bool = False,
    ) -> ReplaySession:
        if new_targets > 0 and rng is None:
            raise ValueError("new targets are drawn with a random generator; pass rng")
        begins, targets = run.transport_begins()
        session: ReplaySession = cls(
            run,
            policy,
            begins,
            targets,
            teacher=teacher,
            with_naive=with_naive,
            show_ghost=show_ghost,
            new_targets=new_targets,
            rng=rng,
            show_curves=show_curves,
        )
        session._begin(0)
        return session

    @property
    def on_new_target(self) -> bool:
        """Whether the agent drives a target of its own, without a ghost."""
        return self.index >= self.run.transports

    def _begin(self, index: int) -> None:
        self.index = index
        self.segment += 1
        self.ghost_played = 0
        if self.on_new_target:
            # From wherever the agent stands after the pause, as the live
            # crane draws its next target (§8); no teacher drove this one.
            assert self.state is not None and self.rng is not None
            start: CraneState = self.state
            target: FloatArray = sample_targets(
                self.rng, start.x, self.run.crane, self.run.task
            )
            self.rows = None
        else:
            b: CraneState = self.begins
            k: slice = slice(index, index + 1)
            start = CraneState(self.frame, b.x[k], b.v[k], b.theta[k], b.omega[k])
            target = self.targets[k]
            self.rows = self.run.transport_rows(index)
        self.state = start
        self.transport = Transport.begin(target, self.frame)
        if self.with_naive:
            self.naive = start
            self.naive_transport = Transport.begin(target, self.frame)

    def set_input(self, deflection: float) -> None:
        """Nobody drives along in a replay; input is ignored."""

    def set_buttons(self, a: bool, b: bool) -> None:
        """The stand's steps live in stand.py; a replay ignores buttons."""

    @property
    def _ghost_done(self) -> bool:
        """True also when there is no ghost: a new target has no teacher."""
        return self.rows is None or self.ghost_played >= len(self.rows)

    def _ghost_state(self) -> CraneState:
        if self.ghost_played == 0:
            b: CraneState = self.begins
            k: slice = slice(self.index, self.index + 1)
            return CraneState(self.frame, b.x[k], b.v[k], b.theta[k], b.omega[k])
        assert self.rows is not None
        return self.run.states(
            self.rows[self.ghost_played - 1 : self.ghost_played], self.frame
        )

    def _ghost_command(self) -> float:
        """The command the teacher applied in the row just played; 0 before
        the first row and once the teacher's transport is over."""
        if self.ghost_played == 0 or self._ghost_done:
            return 0.0
        assert self.rows is not None
        return self.run.columns["command"][int(self.rows[self.ghost_played - 1])]

    def tick(self) -> None:
        """Advance one frame. The agent keeps driving after its transport
        has ended, as the live crane keeps responding; the ghost stops at
        the teacher's last row. When both are done, a pause, then the next
        transport."""
        assert self.state is not None and self.transport is not None
        task = self.run.task
        crane = self.run.crane
        running: bool = not self.transport.done(self.state.frame, task)[0]
        u: FloatArray = self.policy.act(features(self.state, self.transport.target, crane))
        self.command = float(u[0])
        self.state = step(self.state, u, crane)
        self.frame = self.state.frame
        if running:
            self.transport = advance(self.transport, self.state, crane, task)
        if self.naive is not None:
            self._tick_naive()
        if not self._ghost_done:
            self.ghost_played += 1
        if self.pause_left > 0:
            self.pause_left -= 1
            if self.pause_left == 0:
                self._begin((self.index + 1) % (self.run.transports + self.new_targets))
        elif self.transport.done(self.frame, task)[0] and self._ghost_done:
            self.pause_left = PAUSE_FRAMES
            self.transports_done += 1

    def _tick_naive(self) -> None:
        assert self.naive is not None and self.naive_transport is not None
        task = self.run.task
        crane = self.run.crane
        running: bool = not self.naive_transport.done(self.naive.frame, task)[0]
        u: FloatArray = naive_command(self.naive, self.naive_transport.target, crane)
        self.naive = step(self.naive, u, crane)
        if running:
            self.naive_transport = advance(self.naive_transport, self.naive, crane, task)

    def _others(self) -> list[dict[str, Any]]:
        """Further cranes beside the agent, each with its kind, so that the
        frontend can colour it; for now only the naive driver."""
        if self.naive is None:
            return []
        assert self.naive_transport is not None
        return [
            {
                "kind": "naive",
                "x": self.naive.x.tolist(),
                "theta": self.naive.theta.tolist(),
                "result": "success" if self.naive_transport.succeeded[0] else "running",
            }
        ]

    def _agent_result(self) -> str:
        assert self.transport is not None
        if self.transport.succeeded[0]:
            return "success"
        return (
            "timeout" if self.transport.timed_out(self.frame, self.run.task)[0] else "running"
        )

    def _agent_frames(self) -> int:
        assert self.transport is not None
        if self.transport.succeeded[0]:
            return int(self.transport.duration_frames()[0])
        return min(self.frame - self.transport.start_frame, self.run.task.limit_frames)

    def _ghost_result(self) -> str:
        if self.rows is None:
            return "none"
        if not self._ghost_done:
            return "running"
        # A recorded transport ends at success or at the time limit (§13).
        return "timeout" if len(self.rows) >= self.run.task.limit_frames else "success"

    def _banner(self) -> str | None:
        """Shown once both are done: the two times side by side."""
        if self._agent_result() == "running" or self._ghost_result() == "running":
            return None

        def time_of(result: str, frames: int) -> str:
            return _seconds(frames) if result == "success" else "Zeit abgelaufen"

        agent: str = time_of(self._agent_result(), self._agent_frames())
        if self.rows is None:
            return f"Agent {agent}"
        if not self.show_ghost and self.naive_transport is not None:
            # Without the ghost the comparison is learnt against naive; the
            # naive driver rarely comes to rest, and that is the point.
            naive: str = (
                _seconds(int(self.naive_transport.duration_frames()[0]))
                if self.naive_transport.succeeded[0]
                else "pendelt noch"
            )
            return f"Gelernt {agent} · Ohne Lernen {naive}"
        ghost: str = time_of(self._ghost_result(), len(self.rows))
        return f"Agent {agent} · {self.teacher} {ghost}"

    def _curves(self) -> dict[str, Any]:
        """One point of the curves over time, the agent's and, while the
        ghost plays, the teacher's, with the scales and labels to draw them
        by (§20). The frontend collects the points of one segment."""
        assert self.state is not None and self.transport is not None
        frames: int = self.frame - self.transport.start_frame
        longest: int = max(frames, len(self.rows) if self.rows is not None else 0)
        seconds: float = CURVE_STEP_SECONDS * max(
            1, math.ceil(longest / SIM_HZ / CURVE_STEP_SECONDS)
        )
        teacher: dict[str, float] | None = None
        if self.rows is not None and not self._ghost_done:
            ghost: CraneState = self._ghost_state()
            teacher = {
                "v": float(ghost.v[0]),
                "theta": float(ghost.theta[0]),
                "command": self._ghost_command(),
            }
        # The pause after a transport is not part of the drive.
        agent: dict[str, float] | None = None
        if self.pause_left == 0:
            agent = {
                "v": float(self.state.v[0]),
                "theta": float(self.state.theta[0]),
                "command": self.command,
            }
        return {
            "segment": self.segment,
            "t": frames / SIM_HZ,
            "seconds": seconds,
            "agent": agent,
            "teacher": teacher,
            "scales": {
                "v": self.run.crane.max_speed,
                "theta": math.radians(CURVE_SWING_DEG),
                "command": 1.0,
            },
            "labels": CURVE_LABELS,
        }

    def texts(self) -> dict[str, str]:
        """The labels of the readouts; German, from the server (§17)."""
        return {
            "elapsed_label": "Agent",
            "ghost_label": self.teacher,
            "command_label": "Stellgröße Agent",
            "ghost_command_label": f"Stellgröße {self.teacher}",
        }

    def config_message(self) -> dict[str, Any]:
        return {
            "type": "config",
            "hz": SIM_HZ,
            "crane": self.run.crane.to_dict(),
            "task": self.run.task.to_dict(),
            "mode": "replay",
            "headline": HEADLINE,
            "hint": "Wiedergabe – das Gamepad hat Pause",
            **self.texts(),
        }

    def state_message(self) -> dict[str, Any]:
        assert self.state is not None and self.transport is not None
        message: dict[str, Any] = {
            "type": "state",
            "frame": self.frame,
            "x": self.state.x.tolist(),
            "v": self.state.v.tolist(),
            "theta": self.state.theta.tolist(),
            "target": self.transport.target.tolist(),
            "command": self.command,
            "calm": min(1.0, int(self.transport.calm_frames[0]) / self.run.task.calm_frames),
            "result": self._agent_result(),
            "elapsed": self._agent_frames() / SIM_HZ,
            "transports": self.transports_done,
            "driver": "agent",
            "others": self._others(),
            "banner": self._banner(),
        }
        if self.show_ghost and self.rows is not None:
            ghost: CraneState = self._ghost_state()
            message["ghost"] = {
                "x": ghost.x.tolist(),
                "v": ghost.v.tolist(),
                "theta": ghost.theta.tolist(),
                "result": self._ghost_result(),
                "elapsed": self.ghost_played / SIM_HZ,
                "command": self._ghost_command(),
            }
        if self.show_curves:
            message["curves"] = self._curves()
        return message
