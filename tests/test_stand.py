"""The stand's steps for one visitor after another (DECISIONS.md §18)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teach_by_showing import api
from teach_by_showing.cloning import clone
from teach_by_showing.crane import CraneParams
from teach_by_showing.live import InputParams, Settings
from teach_by_showing.recording import Run, load_run
from teach_by_showing.stand import (
    LEARN_FRAMES,
    TEACH_TIMEOUT_FRAMES,
    TEACH_TRANSPORTS,
    Phase,
    Showcase,
    StandSession,
)
from tests.support import ROOT
from tests.teachers import quick, teach

# Linear, so that the scripted command is the command applied, as in teachers.py.
SETTINGS: Settings = Settings.from_dict({"input": {"expo": 0.0}})


@cache
def showcase() -> Showcase:
    run: Run = teach(quick, transports=2, seed=1)
    policy, _ = clone([run], np.random.default_rng(0))
    return Showcase(run, policy)


def stand(recorded: list[Run] | None = None) -> StandSession:
    sink: list[Run] = [] if recorded is None else recorded
    return StandSession.start(
        np.random.default_rng(0), SETTINGS, showcase(), lambda: "test", sink.append, seed=0
    )


def press(s: StandSession, a: bool = False, b: bool = False) -> None:
    """A press as the browser reports it: held for one frame, then released."""
    s.set_buttons(a, b)
    s.tick()
    s.set_buttons(False, False)


def demonstrate(s: StandSession) -> None:
    """Drive the visitor's crane with a scripted teacher until it has learnt."""
    for _ in range(TEACH_TRANSPORTS * 2 * SETTINGS.task.limit_frames):
        if s.phase is not Phase.TEACH:
            return
        assert s.live is not None
        s.set_input(max(-1.0, min(1.0, quick(s.live))))
        s.tick()
    raise AssertionError("the demonstration did not end")


def texts(s: StandSession) -> dict[str, Any]:
    texts: dict[str, Any] = s.state_message()["texts"]
    return texts


def test_the_stand_waits_with_an_earlier_agent_and_a_call_to_action() -> None:
    s: StandSession = stand()
    for _ in range(10 * 60):
        s.tick()
    m: dict[str, Any] = s.state_message()
    assert s.phase is Phase.IDLE and m["phase"] == "idle"
    assert "ghost" in m and m["texts"]["ghost_label"] == "Vorbild"
    assert "Drücken Sie A" in m["texts"]["prompt"]


def test_input_drives_nothing_while_the_stand_waits() -> None:
    a: StandSession = stand()
    b: StandSession = stand()
    for _ in range(120):
        a.set_input(1.0)
        a.tick()
        b.tick()
    assert a.state_message()["x"] == b.state_message()["x"]


def test_a_begins_the_demonstration_and_the_prompt_waits_for_the_first_deflection() -> None:
    s: StandSession = stand()
    press(s, a=True)
    assert s.phase is Phase.TEACH
    m: dict[str, Any] = s.state_message()
    assert "ghost" not in m and m["texts"]["ghost_label"] is None
    assert "Transport 1 von 3" in m["texts"]["headline"]
    assert m["texts"]["prompt"] is not None
    s.set_input(0.5)
    s.tick()
    assert texts(s)["prompt"] is None


def test_a_held_down_counts_once() -> None:
    """The browser reports A as held in every frame it is down; only the
    press changes the step, or the stand would skip through on one push."""
    s: StandSession = stand()
    s.set_buttons(True, False)
    s.tick()
    assert s.phase is Phase.TEACH
    s.set_buttons(True, True)  # B aborts; A is still held from before
    s.tick()
    s.set_buttons(True, False)
    for _ in range(30):
        s.tick()
    assert s.phase is Phase.IDLE


def test_the_whole_way_from_demonstration_to_the_visitors_agent() -> None:
    recorded: list[Run] = []
    s: StandSession = stand(recorded)
    press(s, a=True)
    demonstrate(s)
    assert s.phase is Phase.LEARN
    assert len(recorded) == 1 and recorded[0].transports == TEACH_TRANSPORTS
    assert recorded[0].label == "besuch"
    m: dict[str, Any] = s.state_message()
    assert m["banner"] is None and m["texts"]["prompt"] is not None
    for _ in range(LEARN_FRAMES):
        s.tick()
    assert s.phase is Phase.AGENT
    m = s.state_message()
    assert m["texts"]["ghost_label"] == "Ihre Fahrt" and m["texts"]["prompt"] is None
    assert s.showcase.run is recorded[0]


def test_the_visitors_agent_drives_until_a_button_decides() -> None:
    s: StandSession = stand()
    press(s, a=True)
    demonstrate(s)
    for _ in range(LEARN_FRAMES + 2 * TEACH_TIMEOUT_FRAMES):
        s.tick()
    assert s.phase is Phase.AGENT
    press(s, b=True)
    assert s.phase is Phase.IDLE
    # The stand now waits with this visitor's agent.
    assert s.replay is not None and s.replay.run is s.showcase.run
    assert texts(s)["ghost_label"] == "Vorbild"


def test_a_in_the_agents_step_begins_a_new_demonstration() -> None:
    s: StandSession = stand()
    press(s, a=True)
    demonstrate(s)
    for _ in range(LEARN_FRAMES):
        s.tick()
    press(s, a=True)
    assert s.phase is Phase.TEACH


def test_b_aborts_the_demonstration_and_nothing_is_recorded() -> None:
    recorded: list[Run] = []
    s: StandSession = stand(recorded)
    press(s, a=True)
    for _ in range(200):
        s.set_input(1.0)
        s.tick()
    press(s, b=True)
    assert s.phase is Phase.IDLE and recorded == []
    assert s.showcase is showcase()


def test_a_visitor_who_walks_away_leaves_the_stand_waiting() -> None:
    s: StandSession = stand()
    press(s, a=True)
    for _ in range(TEACH_TIMEOUT_FRAMES):
        s.set_input(0.0)
        s.tick()
    assert s.phase is Phase.TEACH
    s.tick()
    assert s.phase is Phase.IDLE


def test_the_frame_count_runs_on_across_the_steps() -> None:
    """The browser interpolates by frame number (§9)."""
    s: StandSession = stand()
    frames: list[int] = []
    for _ in range(30):
        s.tick()
        frames.append(s.state_message()["frame"])
    press(s, a=True)
    frames.append(s.state_message()["frame"])
    for _ in range(30):
        s.set_input(1.0)
        s.tick()
        frames.append(s.state_message()["frame"])
    assert frames == list(range(1, 62))


def test_a_showcase_under_other_physics_is_refused() -> None:
    other: Settings = Settings(
        CraneParams(rope_length=2.0), SETTINGS.task, InputParams(expo=0.0)
    )
    with pytest.raises(ValueError, match="other crane or task"):
        StandSession.start(np.random.default_rng(0), other, showcase(), str, print)


def test_the_shipped_showcase_is_the_quick_teaching_run_under_todays_physics() -> None:
    shipped: Path = ROOT / "src" / "teach_by_showing" / "showcase" / api.SHOWCASE_RUN
    run: Run = load_run(shipped)
    teaching: Path = ROOT / "tests" / "data" / "20261009-091246-zügig.json"
    assert run.to_dict() == load_run(teaching).to_dict()
    defaults: Settings = Settings.from_dict({})
    assert run.crane == defaults.crane and run.task == defaults.task


def test_buttons_arrive_with_the_input_and_anything_but_true_is_not_held() -> None:
    s: StandSession = stand()
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "a": 1, "b": "yes"}))
    s.tick()
    assert s.phase is Phase.IDLE
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "a": True}))
    s.tick()
    assert s.phase is Phase.TEACH
