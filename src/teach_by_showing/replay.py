"""Ghost replay: the clone drives its teacher's transports while the
teacher's recorded drive plays alongside, translucent (DECISIONS.md §17).

The agent is simulated with the live crane's step() and advance(); the ghost
is not simulated at all but plays the recorded rows, so it shows exactly
what the teacher did. Both begin each transport in the state the teacher
began it in, with the same target. The module knows nothing about HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from teach_by_showing.cloning import Policy
from teach_by_showing.crane import SIM_HZ, CraneState, FloatArray, step
from teach_by_showing.features import features
from teach_by_showing.live import PAUSE_FRAMES
from teach_by_showing.recording import Run
from teach_by_showing.task import Transport, advance

HEADLINE: Final[str] = "Ihr Agent fährt – Ihre eigene Fahrt läuft als Schatten mit"


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

    @classmethod
    def start(cls, run: Run, policy: Policy) -> ReplaySession:
        begins, targets = run.transport_begins()
        session: ReplaySession = cls(run, policy, begins, targets)
        session._begin(0)
        return session

    def _begin(self, index: int) -> None:
        b: CraneState = self.begins
        k: slice = slice(index, index + 1)
        self.index = index
        self.state = CraneState(self.frame, b.x[k], b.v[k], b.theta[k], b.omega[k])
        self.transport = Transport.begin(self.targets[k], self.frame)
        self.rows = self.run.transport_rows(index)
        self.ghost_played = 0

    def set_input(self, deflection: float) -> None:
        """Nobody drives along in a replay; input is ignored."""

    @property
    def _ghost_done(self) -> bool:
        assert self.rows is not None
        return self.ghost_played >= len(self.rows)

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
        if not self._ghost_done:
            self.ghost_played += 1
        if self.pause_left > 0:
            self.pause_left -= 1
            if self.pause_left == 0:
                self._begin((self.index + 1) % self.run.transports)
        elif self.transport.done(self.frame, task)[0] and self._ghost_done:
            self.pause_left = PAUSE_FRAMES
            self.transports_done += 1

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
        assert self.rows is not None
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

        assert self.rows is not None
        agent: str = time_of(self._agent_result(), self._agent_frames())
        ghost: str = time_of(self._ghost_result(), len(self.rows))
        return f"Agent {agent} · Ihre Fahrt {ghost}"

    def config_message(self) -> dict[str, Any]:
        return {
            "type": "config",
            "hz": SIM_HZ,
            "crane": self.run.crane.to_dict(),
            "task": self.run.task.to_dict(),
            "mode": "replay",
            "headline": HEADLINE,
            "elapsed_label": "Agent",
            "hint": "Wiedergabe – das Gamepad hat Pause",
            "command_label": "Stellgröße Agent",
            "ghost_command_label": "Stellgröße Ihre Fahrt",
        }

    def state_message(self) -> dict[str, Any]:
        assert self.state is not None and self.transport is not None
        ghost: CraneState = self._ghost_state()
        return {
            "type": "state",
            "frame": self.frame,
            "x": self.state.x.tolist(),
            "theta": self.state.theta.tolist(),
            "target": self.transport.target.tolist(),
            "command": self.command,
            "calm": min(1.0, int(self.transport.calm_frames[0]) / self.run.task.calm_frames),
            "result": self._agent_result(),
            "elapsed": self._agent_frames() / SIM_HZ,
            "transports": self.transports_done,
            "ghost": {
                "x": ghost.x.tolist(),
                "theta": ghost.theta.tolist(),
                "result": self._ghost_result(),
                "elapsed": self.ghost_played / SIM_HZ,
                "command": self._ghost_command(),
            },
            "banner": self._banner(),
        }
