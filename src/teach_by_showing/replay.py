"""Ghost replay: the clone drives its teacher's transports while the
teacher's recorded drive plays alongside, translucent (DECISIONS.md §17).

The agent is simulated with the live crane's step() and advance(); the ghost
is not simulated at all but plays the recorded rows, so it shows exactly
what the teacher did. Both begin each transport in the state the teacher
began it in, with the same target. After the teacher's transports the agent
may drive new targets of its own, without a ghost (§20). On the teacher's
transports both are counted for the style profile (§22). In the gallery
another agent drives beside it, from the same start to the same target
(§23). The module knows nothing about HTTP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from teach_by_showing.cloning import Policy
from teach_by_showing.crane import SIM_HZ, CraneState, FloatArray, step
from teach_by_showing.features import features
from teach_by_showing.live import PAUSE_FRAMES
from teach_by_showing.naive import naive_command
from teach_by_showing.profile import (
    ProfileSummary,
    Tally,
    TransportProfile,
    profiles,
    run_profiles,
    summarize,
)
from teach_by_showing.recording import Run
from teach_by_showing.task import Transport, advance, sample_targets, sample_transports
from teach_by_showing.texts import DEFAULT_LANGUAGE, TEXTS, Texts, number

# The swing curve's full height. A quick teacher swings about 24°, the naive
# driver up to about 20° (§16, §19); beyond this the curve is clipped.
CURVE_SWING_DEG: Final[float] = 30.0
# The time axis is never shorter than this and grows in steps of it, so that
# it does not rescale with every frame while a drive runs long.
CURVE_STEP_SECONDS: Final[float] = 5.0
NOT_YET: Final[str] = "–"


def seconds_text(frames: int, words: Texts) -> str:
    return number(frames / SIM_HZ, 1, words, " s")


def _figure(value: float | None, words: Texts, unit: str = "") -> str:
    return NOT_YET if value is None else number(value, 1, words, unit)


def _time_cell(summary: ProfileSummary, words: Texts) -> str:
    """Mean time over the reached transports, and how many were not."""
    if summary.transports == 0:
        return NOT_YET
    if summary.reached == 0:
        return words.not_reached
    missed: int = summary.transports - summary.reached
    time: str = _figure(summary.seconds, words, " s")
    return time if missed == 0 else words.partly_reached.format(time=time, missed=missed)


def profile_cells(summary: ProfileSummary, words: Texts) -> list[str]:
    return [
        _time_cell(summary, words),
        _figure(summary.max_swing_deg, words, "°"),
        _figure(summary.rest_deg, words, "°"),
        _figure(summary.stick_travel, words),
    ]


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
    # The words it speaks; the stand replaces them when the staff switch
    # the language, so that the switch shows within the frame (§24).
    words: Texts = TEXTS[DEFAULT_LANGUAGE]
    # Whose drive the ghost plays, as the readouts and the banner name it:
    # the visitor's own, or an earlier teacher's in the stand's idle step.
    # A role, not a name, so that it is worded in the current language.
    earlier_teacher: bool = False
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
    # Whether states carry the style profile, teacher against agent on the
    # teacher's transports (§22). Each side's figures of a transport appear
    # once that side has finished it; the first round fixes them, since a
    # later one drives the same again.
    show_profile: bool = False
    tally: Tally | None = None
    teacher_profiles: list[TransportProfile] = field(default_factory=list)
    teacher_shown: list[bool] = field(default_factory=list)
    agent_profiles: list[TransportProfile | None] = field(default_factory=list)
    # Whether the teacher's transports are driven at all; without them only
    # new targets, as in the gallery (§23).
    teacher_transports: bool = True
    # Another agent beside this one, from the same start to the same target;
    # the next transport waits for it, as for the ghost (§23).
    rival: Policy | None = None
    rival_name: str = ""
    rival_state: CraneState | None = None
    rival_transport: Transport | None = None
    # Set by the stand when it stops the drive before both are done, so
    # that the banner shows who is still under way (§23).
    stopped: bool = False

    @classmethod
    def start(
        cls,
        run: Run,
        policy: Policy,
        earlier_teacher: bool = False,
        words: Texts = TEXTS[DEFAULT_LANGUAGE],
        with_naive: bool = False,
        show_ghost: bool = True,
        new_targets: int = 0,
        rng: np.random.Generator | None = None,
        show_curves: bool = False,
        show_profile: bool = False,
        teacher_transports: bool = True,
        rival: Policy | None = None,
        rival_name: str = "",
    ) -> ReplaySession:
        if new_targets > 0 and rng is None:
            raise ValueError("new targets are drawn with a random generator; pass rng")
        if not teacher_transports and new_targets == 0:
            raise ValueError("without the teacher's transports there must be new targets")
        begins, targets = run.transport_begins()
        session: ReplaySession = cls(
            run,
            policy,
            begins,
            targets,
            words=words,
            earlier_teacher=earlier_teacher,
            with_naive=with_naive,
            show_ghost=show_ghost,
            new_targets=new_targets,
            rng=rng,
            show_curves=show_curves,
            show_profile=show_profile,
            teacher_profiles=run_profiles(run) if show_profile else [],
            teacher_shown=[False] * run.transports,
            agent_profiles=[None] * run.transports,
            teacher_transports=teacher_transports,
            rival=rival,
            rival_name=rival_name,
        )
        session._begin(0 if teacher_transports else run.transports)
        return session

    @property
    def teacher(self) -> str:
        w: Texts = self.words
        return w.teacher_earlier if self.earlier_teacher else w.teacher_self

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
            # Without the teacher's transports the first starts at rest.
            assert self.rng is not None
            crane, task = self.run.crane, self.run.task
            if self.state is None:
                x, target = sample_transports(self.rng, crane, task, 1)
                start = CraneState.at_rest(x)
                start = CraneState(self.frame, start.x, start.v, start.theta, start.omega)
            else:
                start = self.state
                target = sample_targets(self.rng, start.x, crane, task)
            self.rows = None
        else:
            b: CraneState = self.begins
            k: slice = slice(index, index + 1)
            start = CraneState(self.frame, b.x[k], b.v[k], b.theta[k], b.omega[k])
            target = self.targets[k]
            self.rows = self.run.transport_rows(index)
        self.state = start
        self.transport = Transport.begin(target, self.frame)
        self.tally = Tally.begin(target)
        if self.rival is not None:
            self.rival_state = start
            self.rival_transport = Transport.begin(target, self.frame)
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
        assert self.tally is not None
        task = self.run.task
        crane = self.run.crane
        running: bool = not self.transport.done(self.state.frame, task)[0]
        u: FloatArray = self.policy.act(features(self.state, self.transport.target, crane))
        self.command = float(u[0])
        self.state = step(self.state, u, crane)
        self.frame = self.state.frame
        if running:
            self.transport = advance(self.transport, self.state, crane, task)
            self.tally = self.tally.update(
                u, self.state, np.ones(1, dtype=np.bool_), crane, task
            )
        if self.naive is not None:
            self._tick_naive()
        if self.rival is not None:
            self._tick_rival()
        if not self._ghost_done:
            self.ghost_played += 1
        if self.show_profile and not self.on_new_target:
            self._note_profile()
        if self.pause_left > 0:
            self.pause_left -= 1
            if self.pause_left == 0:
                self._begin(self._next_index())
        elif (
            self.transport.done(self.frame, task)[0] and self._ghost_done and self._rival_done
        ):
            self.pause_left = PAUSE_FRAMES
            self.transports_done += 1

    def _note_profile(self) -> None:
        """Record which side has finished the teacher's transport under way."""
        assert self.transport is not None and self.tally is not None
        i: int = self.index
        if self._ghost_done:
            self.teacher_shown[i] = True
        if (
            self.agent_profiles[i] is None
            and self.transport.done(self.frame, self.run.task)[0]
        ):
            self.agent_profiles[i] = profiles(self.transport, self.tally)[0]

    def _profile(self) -> dict[str, Any]:
        """The style profile as the screen shows it, worded here (§17, §22):
        means over the teacher's transports each side has finished."""
        teacher: ProfileSummary = summarize(
            [
                p
                for p, shown in zip(self.teacher_profiles, self.teacher_shown, strict=True)
                if shown
            ]
        )
        agent: ProfileSummary = summarize([p for p in self.agent_profiles if p is not None])
        both: int = sum(
            shown and p is not None
            for shown, p in zip(self.teacher_shown, self.agent_profiles, strict=True)
        )
        w: Texts = self.words
        title: str = (
            w.profile_title_mean.format(both=both, total=self.run.transports)
            if both > 0
            else w.profile_title
        )
        # Laid across, one row per driver, to fit beside the readouts.
        return {
            "title": title,
            "columns": ["", *w.profile_figures],
            "rows": [
                [self.teacher, *profile_cells(teacher, w)],
                [w.agent_self, *profile_cells(agent, w)],
            ],
            "note": w.profile_note,
        }

    def _next_index(self) -> int:
        if self.teacher_transports:
            return (self.index + 1) % (self.run.transports + self.new_targets)
        first: int = self.run.transports
        return first + (self.index - first + 1) % self.new_targets

    @property
    def _rival_done(self) -> bool:
        """True also when there is no rival."""
        if self.rival_transport is None:
            return True
        assert self.rival_state is not None
        return bool(self.rival_transport.done(self.rival_state.frame, self.run.task)[0])

    def _tick_rival(self) -> None:
        """As the agent: it keeps driving after its transport has ended."""
        assert self.rival is not None and self.rival_state is not None
        assert self.rival_transport is not None
        task = self.run.task
        crane = self.run.crane
        running: bool = not self.rival_transport.done(self.rival_state.frame, task)[0]
        u: FloatArray = self.rival.act(
            features(self.rival_state, self.rival_transport.target, crane)
        )
        self.rival_state = step(self.rival_state, u, crane)
        if running:
            self.rival_transport = advance(
                self.rival_transport, self.rival_state, crane, task
            )

    def _rival_result(self) -> str:
        assert self.rival_transport is not None and self.rival_state is not None
        if self.rival_transport.succeeded[0]:
            return "success"
        return "timeout" if self._rival_done else "running"

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
        others: list[dict[str, Any]] = []
        if self.naive is not None:
            assert self.naive_transport is not None
            others.append(
                {
                    "kind": "naive",
                    "x": self.naive.x.tolist(),
                    "theta": self.naive.theta.tolist(),
                    "result": "success" if self.naive_transport.succeeded[0] else "running",
                }
            )
        if self.rival_state is not None:
            others.append(
                {
                    "kind": "rival",
                    "x": self.rival_state.x.tolist(),
                    "theta": self.rival_state.theta.tolist(),
                    "result": self._rival_result(),
                }
            )
        return others

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
        """Shown once both are done, or once the drive is stopped: the two
        times side by side."""
        running: bool = self._agent_result() == "running" or self._ghost_result() == "running"
        if running and not self.stopped:
            return None

        w: Texts = self.words

        def time_of(result: str, frames: int) -> str:
            if result == "running":
                # Stopped before calm, as the gallery's time limit does.
                return w.still_driving
            return seconds_text(frames, w) if result == "success" else w.timed_out

        agent: str = time_of(self._agent_result(), self._agent_frames())
        if self.rival_transport is not None:
            if self._rival_result() == "running" and not self.stopped:
                return None
            rival: str = time_of(
                self._rival_result(), int(self.rival_transport.duration_frames()[0])
            )
            return w.banner_rival.format(agent=agent, rival=self.rival_name, rival_time=rival)
        if self.rows is None:
            return w.banner_agent.format(agent=agent)
        if not self.show_ghost and self.naive_transport is not None:
            # Without the ghost the comparison is learnt against naive; the
            # naive driver rarely comes to rest, and that is the point.
            naive: str = (
                seconds_text(int(self.naive_transport.duration_frames()[0]), w)
                if self.naive_transport.succeeded[0]
                else w.still_swinging
            )
            return w.banner_naive.format(agent=agent, naive=naive)
        ghost: str = time_of(self._ghost_result(), len(self.rows))
        return w.banner_ghost.format(agent=agent, teacher=self.teacher, ghost=ghost)

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
            "labels": {
                "v": self.words.curve_speed,
                "theta": self.words.curve_swing,
                "command": self.words.command_label,
            },
        }

    def texts(self) -> dict[str, str]:
        """The labels of the readouts, from the server (§17, §24)."""
        w: Texts = self.words
        return {
            "elapsed_label": w.agent_label,
            "ghost_label": self.teacher,
            "command_label": w.agent_command_label,
            "ghost_command_label": w.teacher_command_label.format(teacher=self.teacher),
        }

    def config_message(self) -> dict[str, Any]:
        return {
            "type": "config",
            "hz": SIM_HZ,
            "crane": self.run.crane.to_dict(),
            "task": self.run.task.to_dict(),
            "mode": "replay",
            "headline": self.words.agent_headline,
            "hint": self.words.replay_hint,
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
        if self.show_profile:
            message["profile"] = self._profile()
        return message
