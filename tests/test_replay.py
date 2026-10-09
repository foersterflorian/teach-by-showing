"""Ghost replay (DECISIONS.md §17)."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from teach_by_showing.api import create_app
from teach_by_showing.cloning import clone
from teach_by_showing.crane import CraneState
from teach_by_showing.live import PAUSE_FRAMES
from teach_by_showing.recording import Run
from teach_by_showing.replay import ReplaySession
from tests.teachers import quick, teach


@cache
def taught() -> Run:
    return teach(quick, transports=2)


def session() -> ReplaySession:
    run: Run = taught()
    policy, _ = clone([run], np.random.default_rng(0))
    return ReplaySession.start(run, policy)


def test_agent_and_ghost_begin_where_the_teacher_began() -> None:
    s: ReplaySession = session()
    begins, targets = taught().transport_begins()
    m: dict[str, Any] = s.state_message()
    assert m["x"] == [begins.x[0]] and m["ghost"]["x"] == [begins.x[0]]
    assert m["target"] == [targets[0]]


def test_the_ghost_plays_the_teachers_states_row_by_row() -> None:
    s: ReplaySession = session()
    run: Run = taught()
    for row in run.transport_rows(0):
        s.tick()
        ghost: dict[str, Any] = s.state_message()["ghost"]
        assert ghost["x"] == [run.columns["x"][row]]
        assert ghost["theta"] == [run.columns["theta"][row]]
    assert s.state_message()["ghost"]["result"] == "success"


def finish_transport(s: ReplaySession) -> int:
    """Ticks until both are done and the pause has begun."""
    for k in range(2 * taught().task.limit_frames):
        s.tick()
        if s.pause_left > 0:
            return k
    raise AssertionError("the transport did not end")


def test_after_both_are_done_a_pause_then_the_next_transport_then_the_first() -> None:
    s: ReplaySession = session()
    run: Run = taught()
    _, targets = run.transport_begins()
    finish_transport(s)
    m: dict[str, Any] = s.state_message()
    assert m["result"] == "success" and m["ghost"]["result"] == "success"
    assert m["banner"].startswith("Agent ") and "Ihre Fahrt" in m["banner"]
    for _ in range(PAUSE_FRAMES):
        s.tick()
    assert s.state_message()["target"] == [targets[1]]
    assert s.state_message()["banner"] is None
    finish_transport(s)
    for _ in range(PAUSE_FRAMES):
        s.tick()
    assert s.state_message()["target"] == [targets[0]]
    assert s.transports_done == 2


def test_frames_run_on_across_transports() -> None:
    """The browser interpolates by frame (§9); time must never go back."""
    s: ReplaySession = session()
    frames: list[int] = []
    finish_transport(s)
    for _ in range(PAUSE_FRAMES + 5):
        s.tick()
        frames.append(s.state_message()["frame"])
    assert all(b == a + 1 for a, b in zip(frames, frames[1:], strict=False))


def test_input_changes_nothing() -> None:
    a: ReplaySession = session()
    b: ReplaySession = session()
    for k in range(300):
        b.set_input(1.0 if k % 2 else -1.0)
        a.tick()
        b.tick()
    assert a.state_message() == b.state_message()


def test_a_replay_is_reproducible_and_its_messages_are_json() -> None:
    def messages() -> list[str]:
        s: ReplaySession = session()
        out: list[str] = [json.dumps(s.config_message())]
        for _ in range(600):
            s.tick()
            out.append(json.dumps(s.state_message()))
        return out

    first: list[str] = messages()
    assert first == messages()
    config: dict[str, Any] = json.loads(first[0])
    assert config["mode"] == "replay" and config["headline"]


def test_the_agent_starts_from_the_recorded_state_not_from_rest() -> None:
    """A later transport begins where the pause left the teacher's crane."""
    s: ReplaySession = session()
    finish_transport(s)
    for _ in range(PAUSE_FRAMES):
        s.tick()
    begins: CraneState = taught().transport_begins()[0]
    assert s.state is not None
    assert s.state.v[0] == begins.v[1] and s.state.theta[0] == begins.theta[1]


def test_the_web_layer_serves_a_replay() -> None:
    with TestClient(create_app(session)) as client, client.websocket_connect("/ws") as ws:
        config: dict[str, Any] = json.loads(ws.receive_text())
        state: dict[str, Any] = json.loads(ws.receive_text())
    assert config["mode"] == "replay"
    assert state["ghost"]["x"]


def test_the_ghost_carries_the_teachers_command() -> None:
    """Both commands side by side show how differently the two steer."""
    s: ReplaySession = session()
    run: Run = taught()
    rows = run.transport_rows(0)
    assert s.state_message()["ghost"]["command"] == 0.0
    for row in rows[:-1]:
        s.tick()
        assert s.state_message()["ghost"]["command"] == run.columns["command"][row]
    s.tick()
    assert s.state_message()["ghost"]["command"] == 0.0
    labels: dict[str, Any] = s.config_message()
    assert labels["command_label"] and labels["ghost_command_label"]
