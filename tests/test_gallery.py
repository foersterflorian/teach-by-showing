"""The gallery (DECISIONS.md §23): stored agents and the filter that hides
without deleting, the choice of the opponent and the table, the stand's
fourth step, the staff's key and the staff's command."""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teach_by_showing import api, gallery
from teach_by_showing.cloning import LinearPolicy, Policy, clone
from teach_by_showing.crane import CraneParams
from teach_by_showing.gallery import (
    TABLE_ROWS,
    Agent,
    GalleryStore,
    make_agent,
    opponent,
    reference_profile,
    table_agents,
)
from teach_by_showing.live import Settings
from teach_by_showing.profile import ProfileSummary
from teach_by_showing.recording import Run
from teach_by_showing.stand import Phase, Showcase, StandSession
from teach_by_showing.task import TaskParams
from tests.teachers import cautious, quick, teach

SETTINGS: Settings = Settings.from_dict({"input": {"expo": 0.0}})
CRANE: CraneParams = SETTINGS.crane
TASK: TaskParams = SETTINGS.task


def fake(
    name: str, seconds: float | None, number: int | None = 1, created: str = "x"
) -> Agent:
    """An agent with made-up reference figures, for the choice and the table."""
    reached: int = 0 if seconds is None else 10
    summary = ProfileSummary(10, reached, seconds, 10.0, 2.0, 1.5)
    policy: LinearPolicy = LinearPolicy(np.zeros(3), np.zeros(()))
    return Agent(number, name, created, None, policy, summary, CRANE, TASK)


@cache
def run_of(name: str) -> Run:
    return teach({"cautious": cautious, "quick": quick}[name], transports=2, seed=1)


@cache
def policy_of(name: str) -> Policy:
    policy, _ = clone([run_of(name)], np.random.default_rng(0))
    return policy


MODEL_NAMES: dict[str, str] = {"cautious": "Vorbild vorsichtig", "quick": "Vorbild zügig"}


@cache
def model(name: str) -> Agent:
    return make_agent(None, MODEL_NAMES[name], "20261009", None, policy_of(name), CRANE, TASK)


def visitor(number: int, created: str) -> Agent:
    return make_agent(
        number, f"Kranführer·in #{number}", created, None, policy_of("quick"), CRANE, TASK
    )


def test_an_agent_survives_json(tmp_path: Path) -> None:
    store: GalleryStore = GalleryStore(tmp_path)
    agent: Agent = visitor(1, "20261021-100000")
    store.add(agent)
    (loaded,) = store.agents()
    assert json.dumps(loaded.to_dict()) == json.dumps(agent.to_dict())


def test_the_reference_is_the_same_ten_targets_for_everyone() -> None:
    a: ProfileSummary = reference_profile(policy_of("quick"), CRANE, TASK)
    b: ProfileSummary = reference_profile(policy_of("quick"), CRANE, TASK)
    slow: ProfileSummary = reference_profile(policy_of("cautious"), CRANE, TASK)
    assert a == b and a.transports == 10 and a.reached == 10
    assert slow.seconds is not None and a.seconds is not None and slow.seconds > a.seconds


def test_numbers_run_on_across_a_restart_and_are_never_reused(tmp_path: Path) -> None:
    store: GalleryStore = GalleryStore(tmp_path)
    store.add(fake("Kranführer·in #1", 10.0, number=1))
    store.add(fake("Kranführer·in #2", 10.0, number=2))
    store.set_since("z")  # hides both
    assert GalleryStore(tmp_path).next_number() == 3


def test_other_parameters_drive_the_reference_again() -> None:
    agent: Agent = visitor(1, "x")
    assert agent.measured_under(CRANE, TASK) is agent
    slower: CraneParams = CraneParams(max_speed=0.6)
    again: Agent = agent.measured_under(slower, TASK)
    assert again.crane == slower and again.reference != agent.reference


def test_the_filter_hides_without_deleting_and_never_hides_the_models(tmp_path: Path) -> None:
    store: GalleryStore = GalleryStore(tmp_path, (fake("Vorbild", 20.0, number=None),))
    for n, created in enumerate(["20261021-090000", "20261021-110000", "20261022-090000"], 1):
        store.add(fake(f"#{n}", 10.0, number=n, created=created))
    names = lambda: [a.name for a in store.visible(CRANE, TASK)]  # noqa: E731
    assert names() == ["Vorbild", "#1", "#2", "#3"]
    store.set_since("20261021-120000")
    assert names() == ["Vorbild", "#3"]
    assert len(store.agents()) == 3
    store.set_since(None)
    assert names() == ["Vorbild", "#1", "#2", "#3"]
    assert gallery.since_argument("#2", store, "now") == "20261021-110000"


def test_the_opponent_differs_most_in_time_as_a_ratio() -> None:
    own: Agent = fake("own", 10.0)
    others: list[Agent] = [own, fake("a", 12.0), fake("b", 30.0), fake("c", 4.0)]
    # 30 s is three times as long, 4 s two and a half times as short.
    rival: Agent | None = opponent(own, others)
    assert rival is not None and rival.name == "b"
    # Between equals the most recent, which comes last.
    tie: Agent | None = opponent(own, [fake("old", 20.0), fake("new", 5.0)])
    assert tie is not None and tie.name == "new"
    assert opponent(own, [own]) is None


def test_an_agent_that_reached_nothing_counts_with_the_time_limit() -> None:
    own: Agent = fake("own", None)
    assert own.seconds == TASK.time_limit
    rival: Agent | None = opponent(own, [fake("slow", 50.0), fake("fast", 8.0)])
    assert rival is not None and rival.name == "fast"


def test_the_table_is_own_opponent_and_the_most_recent_sorted_by_time() -> None:
    own: Agent = fake("own", 15.0, created="5")
    rival: Agent = fake("rival", 40.0, created="1")
    others: list[Agent] = [fake(f"o{k}", 10.0 + k, created=str(k)) for k in range(2, 9)]
    rows: list[Agent] = table_agents(own, rival, [own, rival, *others])
    assert len(rows) == TABLE_ROWS
    assert {"own", "rival", "o8", "o7", "o6", "o5"} == {a.name for a in rows}
    assert [a.seconds for a in rows] == sorted(a.seconds for a in rows)


def stand(store: GalleryStore) -> StandSession:
    run: Run = run_of("quick")
    return StandSession.start(
        np.random.default_rng(0),
        SETTINGS,
        Showcase(run, policy_of("quick")),
        lambda: "20261021-120000",
        lambda run: None,
        seed=0,
        gallery=store,
    )


def press(s: StandSession, a: bool = False, b: bool = False) -> None:
    s.set_buttons(a, b)
    s.tick()
    s.set_buttons(False, False)


def until(s: StandSession, phase: Phase, limit: int = 20_000) -> None:
    for _ in range(limit):
        if s.phase is phase:
            return
        if s.phase is Phase.TEACH:
            assert s.live is not None
            s.set_input(max(-1.0, min(1.0, cautious(s.live))))
        s.tick()
    raise AssertionError(f"never reached {phase}")


@pytest.fixture
def visited(tmp_path: Path) -> StandSession:
    """A stand whose visitor has taught a cautious agent and watched it
    drive its first round, now in the gallery."""
    s: StandSession = stand(GalleryStore(tmp_path, (model("cautious"), model("quick"))))
    press(s, a=True)
    until(s, Phase.LEARN)
    until(s, Phase.AGENT)
    until(s, Phase.GALLERY)
    return s


def test_the_visitors_agent_joins_and_after_one_round_meets_its_opposite(
    visited: StandSession,
) -> None:
    s: StandSession = visited
    assert s.gallery is not None and [a.number for a in s.gallery.agents()] == [1]
    assert s.own is not None and s.own.name == "Kranführer·in #1"
    # A cautious visitor meets the quick model.
    assert s.rival is not None and s.rival.name == "Vorbild zügig"
    r = s.replay
    assert r is not None and r.rival_transport is not None and r.transport is not None
    assert r.rival_transport.target[0] == r.transport.target[0]
    m: dict[str, Any] = s.state_message()
    assert [o["kind"] for o in m["others"]] == ["rival"]
    assert "Vorbild zügig" in m["texts"]["headline"]
    table: dict[str, Any] = m["profile"]
    assert table["rows"][table["highlight"]][0] == "Ihr Agent (#1)"
    assert len(table["rows"]) == 3
    json.dumps(m)


def test_the_gallery_holds_its_result_then_the_stand_waits(visited: StandSession) -> None:
    s: StandSession = visited
    until(s, Phase.IDLE)
    assert s.hold_from is not None
    held: int = s.frame - s.hold_from
    assert held == SETTINGS.stand.frames(SETTINGS.stand.gallery_hold)


def test_a_and_b_work_in_the_gallery(visited: StandSession) -> None:
    press(visited, b=True)
    assert visited.phase is Phase.IDLE
    until(visited, Phase.IDLE)
    press(visited, a=True)
    assert visited.phase is Phase.TEACH


def test_the_staffs_key_counts_once_and_hides_the_earlier_agents(tmp_path: Path) -> None:
    store: GalleryStore = GalleryStore(tmp_path, (model("quick"),))
    store.add(fake("#1", 10.0, number=1, created="20261021-090000"))
    s: StandSession = stand(store)
    s.set_staff(True)
    s.tick()
    assert store.since() == "20261021-120000"
    store.set_since(None)
    s.set_staff(True)  # still held: no second press
    s.tick()
    assert store.since() is None
    s.set_staff(False)
    s.set_staff(True)
    s.tick()
    assert store.since() == "20261021-120000"
    assert s.state_message()["banner"] == "Galerie neu ab jetzt"
    assert [a.name for a in store.visible(CRANE, TASK)] == ["Vorbild zügig"]


def test_the_staffs_key_arrives_with_the_input_and_only_true_counts(tmp_path: Path) -> None:
    store: GalleryStore = GalleryStore(tmp_path, (model("quick"),))
    s: StandSession = stand(store)
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "reset": "yes"}))
    s.tick()
    assert store.since() is None
    api.handle_input(s, json.dumps({"type": "input", "u": 0.0, "reset": True}))
    s.tick()
    assert store.since() == "20261021-120000"


def test_the_staffs_command_lists_and_sets_the_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store: GalleryStore = GalleryStore(tmp_path)
    store.add(fake("Kranführer·in #1", 10.0, number=1, created="20261021-090000"))
    store.add(fake("Kranführer·in #2", 12.0, number=2, created="20261021-110000"))

    def command(*args: str) -> str:
        monkeypatch.setattr(
            sys, "argv", ["teach-by-showing-gallery", "--data-dir", str(tmp_path), *args]
        )
        gallery.main()
        return capsys.readouterr().out

    command("since", "#2")
    listing: str = command("list")
    assert "from 20261021-110000" in listing
    assert "hidden" in listing.splitlines()[1] and "shown" in listing.splitlines()[2]
    command("since", "all")
    assert store.since() is None
