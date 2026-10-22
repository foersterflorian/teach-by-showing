"""Style profile (DECISIONS.md §22): its figures recomputed by hand, one
code path for teacher, clone and replay, and how the stand shows it."""

from __future__ import annotations

import json
import math
from dataclasses import astuple
from functools import cache
from typing import Any

import numpy as np
import pytest

from teach_by_showing.cloning import Policy, Rollout, clone, rollout
from teach_by_showing.crane import SIM_HZ, CraneParams, CraneState, swing_amplitude
from teach_by_showing.profile import (
    Tally,
    TransportProfile,
    profiles,
    run_profiles,
    summarize,
)
from teach_by_showing.recording import Run
from teach_by_showing.replay import ReplaySession
from teach_by_showing.task import TaskParams, Transport
from teach_by_showing.texts import ENGLISH
from tests.teachers import quick, teach

CRANE: CraneParams = CraneParams()
TASK: TaskParams = TaskParams()
TARGET: float = 5.0


def close(a: TransportProfile | None, b: TransportProfile) -> bool:
    """Equal up to rounding: a batch may sum in another order than one crane."""
    assert a is not None
    return all(
        (x is None and y is None) or (x is not None and y is not None and math.isclose(x, y))
        for x, y in zip(astuple(a), astuple(b), strict=True)
    )


def at(x: float, theta: float = 0.0) -> CraneState:
    """A state whose swing amplitude is |theta|: the load is at its turning point."""
    return CraneState(0, np.asarray([x]), np.zeros(1), np.asarray([theta]), np.zeros(1))


def counted(frames: list[tuple[float, float, float]], running: int | None = None) -> Tally:
    """A tally over frames of (command, x, theta); only the first `running`
    frames belong to the transport."""
    tally: Tally = Tally.begin(np.asarray([TARGET]))
    for k, (u, x, theta) in enumerate(frames):
        on: bool = running is None or k < running
        tally = tally.update(
            np.asarray([u]), at(x, theta), np.asarray([on], dtype=np.bool_), CRANE, TASK
        )
    return tally


def test_stick_travel_sums_every_change_from_the_centre() -> None:
    tally: Tally = counted([(0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)])
    assert tally.stick_travel[0] == pytest.approx(0.5 + 0.0 + 1.0)


def test_the_residual_swing_counts_from_the_first_frame_near_the_target() -> None:
    # 4.9 m is within 0.15 m of the target; 5.2 m is not, but by then the
    # target has been reached and an overshoot is part of settling.
    tally: Tally = counted(
        [(0.0, 3.0, 0.3), (0.0, 4.9, 0.1), (0.0, 5.2, 0.2), (0.0, 5.0, 0.0)]
    )
    transport: Transport = Transport.begin(np.asarray([TARGET]), 0)
    p: TransportProfile = profiles(transport, tally)[0]
    assert p.rest_deg == pytest.approx(math.degrees((0.1 + 0.2 + 0.0) / 3))
    assert p.max_swing_deg == pytest.approx(math.degrees(0.3))
    assert p.seconds is None and not p.reached


def test_without_reaching_the_target_there_is_no_residual_swing() -> None:
    tally: Tally = counted([(1.0, 2.0, 0.1), (1.0, 3.0, 0.1)])
    p: TransportProfile = profiles(Transport.begin(np.asarray([TARGET]), 0), tally)[0]
    assert p.rest_deg is None


def test_frames_after_the_transport_ended_do_not_count() -> None:
    tally: Tally = counted([(0.5, 5.0, 0.1), (-1.0, 5.0, 0.4), (1.0, 2.0, 0.6)], running=1)
    assert tally.stick_travel[0] == pytest.approx(0.5)
    assert tally.max_swing[0] == pytest.approx(0.1)
    assert int(tally.rest_frames[0]) == 1


def test_a_batch_counts_each_transport_as_if_alone() -> None:
    run: Run = taught()
    policy: Policy = agent()
    begins, targets = run.transport_begins()
    batch: Rollout = rollout(policy, begins, targets, run.crane, run.task)
    together: list[TransportProfile] = profiles(batch.transport, batch.tally)
    for i in range(run.transports):
        k: slice = slice(i, i + 1)
        one_begin: CraneState = CraneState(
            0, begins.x[k], begins.v[k], begins.theta[k], begins.omega[k]
        )
        alone: Rollout = rollout(policy, one_begin, targets[k], run.crane, run.task)
        assert close(profiles(alone.transport, alone.tally)[0], together[i])


def test_means_are_over_the_reached_transports_and_the_missed_are_counted() -> None:
    items: list[TransportProfile] = [
        TransportProfile(10.0, 20.0, 4.0, 2.0),
        TransportProfile(None, 40.0, None, 9.0),
        TransportProfile(20.0, 10.0, 2.0, 4.0),
    ]
    s = summarize(items)
    assert (s.transports, s.reached) == (3, 2)
    assert (s.seconds, s.max_swing_deg, s.rest_deg, s.stick_travel) == (15.0, 15.0, 3.0, 3.0)
    none = summarize([TransportProfile(None, 40.0, None, 9.0)])
    assert none.reached == 0 and none.seconds is None and none.stick_travel is None


@cache
def taught() -> Run:
    return teach(quick, transports=3)


@cache
def agent() -> Policy:
    policy, _ = clone([taught()], np.random.default_rng(0))
    return policy


def test_the_teacher_is_counted_from_the_recording() -> None:
    """Time as the live session counted it while recording; the largest
    swing over the transport's rows; the stick travel of its commands."""
    run: Run = taught()
    items: list[TransportProfile] = run_profiles(run)
    amp: np.ndarray = swing_amplitude(run.states(np.arange(run.frames)), run.crane)
    command: np.ndarray = run.column("command")
    for i, p in enumerate(items):
        rows: np.ndarray = run.transport_rows(i)
        assert p.seconds == len(rows) / SIM_HZ
        assert p.max_swing_deg == pytest.approx(math.degrees(float(amp[rows].max())))
        u: np.ndarray = np.concatenate([[0.0], command[rows]])
        assert p.stick_travel == pytest.approx(float(np.abs(np.diff(u)).sum()))
        assert p.rest_deg is not None and 0.0 < p.rest_deg < p.max_swing_deg


def finish_round(s: ReplaySession) -> list[dict[str, Any]]:
    """Ticks through the teacher's transports, returning the profile at the
    end of each."""
    seen: list[dict[str, Any]] = []
    for i in range(s.run.transports):
        while s.index == i:
            s.tick()
        seen.append(s.state_message()["profile"])
    return seen


def replay(new_targets: int = 0) -> ReplaySession:
    return ReplaySession.start(
        taught(),
        agent(),
        new_targets=new_targets,
        rng=np.random.default_rng(1) if new_targets else None,
        show_profile=True,
    )


def test_the_replay_counts_the_agent_as_a_rollout_does() -> None:
    run: Run = taught()
    s: ReplaySession = replay()
    finish_round(s)
    begins, targets = run.transport_begins()
    r: Rollout = rollout(agent(), begins, targets, run.crane, run.task)
    expected: list[TransportProfile] = profiles(r.transport, r.tally)
    assert all(close(a, b) for a, b in zip(s.agent_profiles, expected, strict=True))


def test_the_profile_fills_transport_by_transport_and_new_targets_do_not_count() -> None:
    s: ReplaySession = replay(new_targets=2)
    first: dict[str, Any] = s.state_message()["profile"]
    assert first["columns"] == ["", *ENGLISH.profile_figures]
    assert [row[0] for row in first["rows"]] == [ENGLISH.teacher_self, ENGLISH.agent_self]
    assert all(cell == "–" for row in first["rows"] for cell in row[1:])

    seen: list[dict[str, Any]] = finish_round(s)
    for k, profile in enumerate(seen, start=1):
        assert profile["title"] == ENGLISH.profile_title_mean.format(both=k, total=3)
        assert all(cell != "–" for row in profile["rows"] for cell in row[1:])
    assert seen[-1]["note"] == ENGLISH.profile_note

    agent_means = summarize([p for p in s.agent_profiles if p is not None])
    assert seen[-1]["rows"][1][4] == f"{agent_means.stick_travel:.1f}"
    while s.on_new_target or s.index != 0:
        s.tick()
    assert s.state_message()["profile"] == seen[-1]
    json.dumps(s.state_message())


def test_a_replay_without_the_profile_sends_none() -> None:
    s: ReplaySession = ReplaySession.start(taught(), agent())
    assert "profile" not in s.state_message()
