"""The naive driver beside the agents (DECISIONS.md §19)."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

from teach_by_showing.cloning import clone
from teach_by_showing.crane import CraneParams, CraneState, swing_amplitude
from teach_by_showing.live import LiveSession
from teach_by_showing.naive import naive_command
from teach_by_showing.recording import Run
from teach_by_showing.replay import ReplaySession
from teach_by_showing.stand import Phase
from tests.teachers import quick, teach
from tests.test_stand import demonstrate, press, stand

CRANE: CraneParams = CraneParams()


@cache
def taught() -> Run:
    return teach(quick, transports=2)


def replay(with_naive: bool) -> ReplaySession:
    run: Run = taught()
    policy, _ = clone([run], np.random.default_rng(0))
    return ReplaySession.start(run, policy, with_naive=with_naive)


def until_pause(s: ReplaySession) -> None:
    for _ in range(2 * taught().task.limit_frames):
        s.tick()
        if s.pause_left > 0:
            return
    raise AssertionError("the transport did not end")


def test_full_speed_towards_the_target_then_braking_in_time() -> None:
    at_rest: CraneState = CraneState.at_rest(np.array([2.0, 8.0]))
    assert naive_command(at_rest, np.array([7.0, 3.0]), CRANE).tolist() == [1.0, -1.0]
    # At full speed 0.48 m from the target: the braking distance is 0.48 m.
    moving: CraneState = CraneState(
        0, np.array([6.5]), np.array([1.2]), np.array([0.0]), np.array([0.0])
    )
    assert naive_command(moving, np.array([6.98]), CRANE).tolist() == [0.0]
    assert naive_command(moving, np.array([7.1]), CRANE).tolist() == [1.0]


def test_the_naive_driver_leaves_the_load_swinging_where_the_agent_settles_it() -> None:
    """The contrast it is there for (§6): at the end of the transport the
    agent's load hangs calm, the naive driver's still swings."""
    s: ReplaySession = replay(with_naive=True)
    until_pause(s)
    assert s.state is not None and s.naive is not None
    agent: float = float(np.degrees(swing_amplitude(s.state, taught().crane))[0])
    naive: float = float(np.degrees(swing_amplitude(s.naive, taught().crane))[0])
    assert agent < 4.0 < 2 * 4.0 < naive


def test_the_naive_driver_holds_up_nothing() -> None:
    """It rarely comes to rest within the time limit; the transports follow
    each other on the agent and the ghost alone."""
    with_naive: ReplaySession = replay(with_naive=True)
    without: ReplaySession = replay(with_naive=False)
    for _ in range(3 * taught().task.limit_frames):
        with_naive.tick()
        without.tick()
        assert with_naive.transports_done == without.transports_done
        assert with_naive.state_message()["x"] == without.state_message()["x"]
    assert with_naive.transports_done >= 2


def test_it_begins_each_transport_where_the_teacher_began() -> None:
    s: ReplaySession = replay(with_naive=True)
    begins, targets = taught().transport_begins()
    for index in range(2):
        other: dict[str, Any] = s.state_message()["others"][0]
        assert other["kind"] == "naive" and other["x"] == [begins.x[index]]
        until_pause(s)
        while s.pause_left > 0:
            s.tick()


def test_only_the_waiting_stand_shows_it_for_now() -> None:
    """Decided 2026-10-16: the idle step only; the visitor's agent drives
    without it."""
    s = stand()
    assert [o["kind"] for o in s.state_message()["others"]] == ["naive"]
    press(s, a=True)
    assert "others" not in s.state_message()
    demonstrate(s)
    for _ in range(5 * 60):
        s.tick()
    assert s.phase is Phase.AGENT and s.state_message()["others"] == []


def test_every_state_names_who_drives_so_the_load_can_be_coloured() -> None:
    assert LiveSession.start(np.random.default_rng(0)).state_message()["driver"] == "human"
    assert replay(with_naive=False).state_message()["driver"] == "agent"
