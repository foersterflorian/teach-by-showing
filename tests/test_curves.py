"""The full ghost replay: curves over time and new targets without a ghost
(DECISIONS.md §20)."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest

from teach_by_showing.cloning import clone
from teach_by_showing.live import PAUSE_FRAMES
from teach_by_showing.recording import Run
from teach_by_showing.replay import HEADLINE, HEADLINE_NEW, ReplaySession
from teach_by_showing.stand import AGENT_NEW_TARGETS, LEARN_FRAMES, Phase
from tests.support import ROOT
from tests.teachers import quick, teach
from tests.test_stand import demonstrate, press, stand

STATIC = ROOT / "src" / "teach_by_showing" / "static"


@cache
def taught() -> Run:
    return teach(quick, transports=2)


def session(new_targets: int = 0) -> ReplaySession:
    run: Run = taught()
    policy, _ = clone([run], np.random.default_rng(0))
    return ReplaySession.start(
        run,
        policy,
        new_targets=new_targets,
        rng=np.random.default_rng(1) if new_targets else None,
        show_curves=True,
    )


def next_transport(s: ReplaySession) -> None:
    """Ticks through the rest of this transport and its pause."""
    segment: int = s.segment
    for _ in range(3 * taught().task.limit_frames):
        s.tick()
        if s.segment != segment:
            return
    raise AssertionError("the next transport did not begin")


def test_states_carry_the_speed_of_agent_and_ghost() -> None:
    s: ReplaySession = session()
    run: Run = taught()
    rows = run.transport_rows(0)
    for row in rows[:20]:
        s.tick()
        m: dict[str, Any] = s.state_message()
        assert s.state is not None and m["v"] == s.state.v.tolist()
        assert m["ghost"]["v"] == [run.columns["v"][row]]


def test_the_teachers_curve_follows_the_recording_and_ends_with_it() -> None:
    s: ReplaySession = session()
    run: Run = taught()
    rows = run.transport_rows(0)
    for row in rows[:-1]:
        s.tick()
        c: dict[str, Any] = s.state_message()["curves"]
        assert c["teacher"] == {
            "v": run.columns["v"][row],
            "theta": run.columns["theta"][row],
            "command": run.columns["command"][row],
        }
    s.tick()
    assert s.state_message()["curves"]["teacher"] is None


def test_the_agents_curve_pauses_with_the_transport() -> None:
    s: ReplaySession = session()
    for _ in range(3 * taught().task.limit_frames):
        s.tick()
        c: dict[str, Any] = s.state_message()["curves"]
        assert (c["agent"] is None) == (s.pause_left > 0)
        if s.pause_left > 0:
            break
    assert s.pause_left > 0


def test_the_time_axis_covers_the_longer_drive_in_whole_steps_of_five_seconds() -> None:
    s: ReplaySession = session()
    teacher_seconds: float = len(taught().transport_rows(0)) / 60
    for _ in range(600):
        s.tick()
        c: dict[str, Any] = s.state_message()["curves"]
        assert c["seconds"] % 5 == 0 and c["seconds"] >= max(c["t"], teacher_seconds)
    assert set(c["scales"]) == set(c["labels"]) == {"v", "theta", "command"}


def test_every_transport_is_a_new_segment_even_back_at_the_first() -> None:
    """The frontend starts its curves afresh on a new segment; after the
    last transport the loop returns to index 0, which must not continue the
    old curve."""
    s: ReplaySession = session()
    seen: list[tuple[int, int]] = [(s.index, s.segment)]
    for _ in range(2):
        next_transport(s)
        seen.append((s.index, s.segment))
    assert seen == [(0, 1), (1, 2), (0, 3)]


def test_after_the_teachers_transports_new_targets_without_a_ghost() -> None:
    s: ReplaySession = session(new_targets=2)
    run: Run = taught()
    _, recorded = run.transport_begins()
    next_transport(s)
    next_transport(s)
    assert s.on_new_target and s.index == run.transports
    m: dict[str, Any] = s.state_message()
    assert "ghost" not in m and m["curves"]["teacher"] is None
    assert m["target"][0] not in recorded.tolist()
    # Drawn from where the agent stands, at least the task's minimum away.
    assert s.state is not None
    assert abs(m["target"][0] - s.state.x[0]) >= run.task.min_distance - 1e-9
    for _ in range(run.task.limit_frames + PAUSE_FRAMES):
        s.tick()
        if s.pause_left > 0:
            break
    assert s.state_message()["result"] == "success"
    assert s.state_message()["banner"].startswith("Agent ")
    next_transport(s)
    next_transport(s)
    assert s.index == 0 and not s.on_new_target


def test_new_targets_need_a_random_generator() -> None:
    run: Run = taught()
    policy, _ = clone([run], np.random.default_rng(0))
    with pytest.raises(ValueError, match="rng"):
        ReplaySession.start(run, policy, new_targets=1)


def test_the_visitors_agent_step_shows_curves_and_new_targets_but_the_idle_step_not() -> None:
    s = stand()
    assert "curves" not in s.state_message()
    press(s, a=True)
    demonstrate(s)
    for _ in range(LEARN_FRAMES):
        s.tick()
    assert s.phase is Phase.AGENT
    replay: ReplaySession | None = s.replay
    assert replay is not None
    assert replay.new_targets == AGENT_NEW_TARGETS
    m: dict[str, Any] = s.state_message()
    assert "curves" in m and m["texts"]["headline"] == HEADLINE
    while not replay.on_new_target:
        next_transport(replay)
    assert s.state_message()["texts"]["headline"] == HEADLINE_NEW


def test_the_curves_are_drawn_from_the_message_with_no_wording_of_their_own() -> None:
    curves: str = (STATIC / "curves.js").read_text(encoding="utf-8")
    main: str = (STATIC / "main.js").read_text(encoding="utf-8")
    assert "recordCurves(message)" in main and "drawCurves(" in main
    assert "curves.labels[key]" in curves and "curves.scales[key]" in curves
    for wording in ("Geschwindigkeit", "Pendel", "Stellgröße"):
        assert wording not in curves, wording
