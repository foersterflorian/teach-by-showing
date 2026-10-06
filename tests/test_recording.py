"""Recording of demonstrations (DECISIONS.md §13)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from teach_by_showing.crane import SIM_HZ, CraneState, step
from teach_by_showing.live import PAUSE_FRAMES, LiveSession, RecordingRequest
from teach_by_showing.recording import Run, load_run, run_filename, save_run


def drive(s: LiveSession) -> None:
    """One frame of a scripted anti-sway teacher, as in test_live.py."""
    u: float = 0.5 * float(s.transport.target[0] - s.state.x[0]) + 2.0 * float(
        s.state.theta[0]
    )
    s.set_input(max(-1.0, min(1.0, u)))
    s.tick()


def record(transports: int = 2, idle: int = 30, seed: int = 0) -> tuple[LiveSession, Run]:
    """A session that idles for `idle` frames, then records `transports`."""
    s: LiveSession = LiveSession.start(np.random.default_rng(seed))
    runs: list[Run] = []
    s.request_recording(
        RecordingRequest("test", seed, "20261006-120000", transports, runs.append)
    )
    for _ in range(idle):
        s.tick()
    for _ in range(transports * 60 * SIM_HZ):
        drive(s)
        if runs:
            return s, runs[0]
    raise AssertionError("recording did not finish")


def test_replaying_the_commands_reproduces_every_state() -> None:
    """A run is complete: its commands, sent through the same step(),
    give back its states bit for bit. Guards determinism and the one code
    path (CLAUDE.md)."""
    _, run = record()
    state: CraneState = run.initial
    commands: np.ndarray = run.column("command")
    for k in range(run.frames):
        state = step(state, commands[k : k + 1], run.crane)
        for name in ("x", "v", "theta", "omega"):
            assert getattr(state, name)[0] == run.columns[name][k]


def test_recording_begins_at_the_first_deflection() -> None:
    """Standing still before the teacher picks up the gamepad would teach
    "do nothing" and count as transport time."""
    s, run = record(idle=90)
    assert run.initial.frame == 90
    assert run.columns["command"][0] != 0.0
    assert run.columns["v"][0] != 0.0


def test_recording_ends_with_the_last_transport_and_marks_pauses() -> None:
    s, run = record(transports=2)
    assert run.transports == 2
    assert s.recording is None and s.request is None
    running: np.ndarray = run.running_mask()
    transport: np.ndarray = np.asarray(run.transport)
    # The last frame is the one that finished the last transport.
    assert running[-1]
    # Between the transports lies one pause, not recorded as driving.
    assert int(np.sum(~running)) == PAUSE_FRAMES + 1
    assert np.all(transport[~running] == 0)
    # The target in each row is the one the teacher was driving to.
    first: np.ndarray = np.asarray(run.columns["target"])[transport == 0]
    assert np.all(first == first[0])


def test_a_session_keeps_running_after_the_recording() -> None:
    s, _ = record(transports=1)
    for _ in range(PAUSE_FRAMES + 2):
        drive(s)
    assert s.state_message()["result"] == "running"


def test_a_run_survives_json(tmp_path: Path) -> None:
    _, run = record(transports=1)
    path: Path = save_run(run, tmp_path / "runs")
    assert path.name == "20261006-120000-test.json"
    assert not list(path.parent.glob("*.partial"))
    again: Run = load_run(path)
    assert again.to_dict() == run.to_dict()
    assert json.loads(path.read_text())["schema"] == 1


def test_a_run_of_another_schema_is_refused(tmp_path: Path) -> None:
    _, run = record(transports=1)
    data = run.to_dict() | {"schema": 0}
    with pytest.raises(ValueError, match="schema"):
        Run.from_dict(data)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("zügig", "t-zügig.json"),
        ("../../etc/passwd", "t-etcpasswd.json"),
        ("a b_c", "t-a-bc.json"),
        ("", "t-run.json"),
    ],
)
def test_labels_cannot_leave_the_folder(label: str, expected: str) -> None:
    assert run_filename("t", label) == expected
