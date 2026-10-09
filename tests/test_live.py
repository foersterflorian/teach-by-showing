"""Live session (DECISIONS.md §8)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from teach_by_showing.crane import SIM_HZ, CraneParams
from teach_by_showing.live import (
    INPUT_TIMEOUT_FRAMES,
    PAUSE_FRAMES,
    InputParams,
    LiveSession,
    Settings,
    parse_command,
)
from teach_by_showing.task import TaskParams
from tests.support import ROOT


def session(seed: int = 0) -> LiveSession:
    return LiveSession.start(np.random.default_rng(seed))


def test_without_input_the_crane_stands_still() -> None:
    s: LiveSession = session()
    x0: float = float(s.state.x[0])
    for _ in range(2 * SIM_HZ):
        s.tick()
    assert s.state.x[0] == x0


def test_input_drives_the_trolley() -> None:
    s: LiveSession = session()
    x0: float = float(s.state.x[0])
    direction: float = 1.0 if x0 < 5.0 else -1.0
    for _ in range(SIM_HZ):
        s.set_input(direction)
        s.tick()
    assert (s.state.x[0] - x0) * direction > 0.3


def test_stale_input_falls_back_to_zero() -> None:
    """A lost connection or unplugged gamepad must not leave the crane
    driving (§8)."""
    s: LiveSession = session()
    s.set_input(1.0)
    for _ in range(INPUT_TIMEOUT_FRAMES):
        s.tick()
    assert s.effective_command == 1.0
    s.tick()
    assert s.effective_command == 0.0
    for _ in range(2 * SIM_HZ):
        s.tick()
    assert s.state.v[0] == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.5, 0.5), (-3, -1.0), (7.0, 1.0), (0, 0.0)],
)
def test_commands_are_clipped(raw: object, expected: float) -> None:
    assert parse_command(raw) == expected


@pytest.mark.parametrize("raw", [None, "1", True, math.nan, math.inf, [1.0], {"u": 1}])
def test_malformed_commands_are_refused(raw: object) -> None:
    assert parse_command(raw) is None


def drive_to_target(s: LiveSession) -> int:
    """A scripted anti-sway driver standing in for the visitor. Returns the
    frames until the transport has succeeded."""
    for frame in range(60 * SIM_HZ):
        u: float = 0.5 * float(s.transport.target[0] - s.state.x[0]) + 2.0 * float(
            s.state.theta[0]
        )
        s.set_input(max(-1.0, min(1.0, u)))
        s.tick()
        if s.transport.succeeded[0]:
            return frame
    raise AssertionError("transport did not succeed")


def test_next_transport_starts_after_a_pause_from_where_the_trolley_stands() -> None:
    s: LiveSession = session()
    first_target: float = float(s.transport.target[0])
    drive_to_target(s)
    assert s.transports_done == 1
    assert s.state_message()["result"] == "success"
    for _ in range(PAUSE_FRAMES):
        s.tick()
    assert s.transport.target[0] == first_target
    s.tick()
    assert s.transport.target[0] != first_target
    assert abs(s.transport.target[0] - s.state.x[0]) >= s.task.min_distance - 1e-12
    assert s.state_message()["result"] == "running"


def test_session_is_reproducible_from_seed_and_inputs() -> None:
    def run(seed: int) -> list[str]:
        s: LiveSession = session(seed)
        out: list[str] = []
        for k in range(20 * SIM_HZ):
            if k % 7 == 0:
                s.set_input(math.sin(k / 50))
            s.tick()
            out.append(json.dumps(s.state_message()))
        return out

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_messages_are_json_and_carry_what_the_frontend_needs() -> None:
    s: LiveSession = session()
    s.tick()
    config: dict[str, Any] = json.loads(json.dumps(s.config_message()))
    state: dict[str, Any] = json.loads(json.dumps(s.state_message()))
    assert config["type"] == "config" and config["hz"] == SIM_HZ
    assert {"rope_length", "track_min", "track_max"} <= set(config["crane"])
    assert state["type"] == "state"
    assert {"frame", "x", "theta", "target", "command", "calm", "result", "elapsed"} <= set(
        state
    )


CORE: tuple[str, ...] = (
    "crane.py",
    "task.py",
    "live.py",
    "recording.py",
    "features.py",
    "cloning.py",
)
WEB: frozenset[str] = frozenset({"fastapi", "starlette", "uvicorn", "websockets", "httpx"})


@pytest.mark.parametrize("module", CORE)
def test_core_knows_nothing_about_http(module: str) -> None:
    """Layering (CLAUDE.md, §8): if web-framework types leak into the core,
    replacing the framework means rewriting the project."""
    tree: ast.Module = ast.parse((ROOT / "src" / "teach_by_showing" / module).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & WEB


def test_a_timed_out_transport_is_counted_and_paused_like_a_success() -> None:
    """Found in the first recorded run, 2026-10-09: a timeout drew the next
    target at once, without the pause and without counting the transport."""
    s: LiveSession = session()
    first_target: float = float(s.transport.target[0])
    for _ in range(s.task.limit_frames):
        s.tick()
    assert s.transports_done == 1
    assert s.state_message()["result"] == "timeout"
    for _ in range(PAUSE_FRAMES):
        s.tick()
    assert s.transport.target[0] == first_target
    s.tick()
    assert s.transport.target[0] != first_target
    assert s.state_message()["result"] == "running"


def test_settings_name_only_what_differs_and_refuse_typos() -> None:
    """Candidate sets for tuning on the gamepad (§14)."""
    s: Settings = Settings.from_dict({"crane": {"damping": 0.15}, "input": {"expo": 0.5}})
    assert s.crane.damping == 0.15 and s.crane.max_accel == CraneParams().max_accel
    assert s.task == TaskParams() and s.input.expo == 0.5
    with pytest.raises(TypeError):
        Settings.from_dict({"crane": {"dampnig": 0.15}})
    with pytest.raises(ValueError, match="unknown"):
        Settings.from_dict({"cran": {}})


@pytest.mark.parametrize("path", sorted((ROOT / "tools" / "tuning").glob("*.json")))
def test_candidate_files_load(path: Path) -> None:
    Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))


def test_the_input_curve_keeps_the_ends_and_refines_the_middle() -> None:
    curve: InputParams = InputParams(expo=0.7)
    u: np.ndarray = np.linspace(-1.0, 1.0, 201)
    shaped: np.ndarray = np.array([curve.shape(float(v)) for v in u])
    assert curve.shape(1.0) == 1.0 and curve.shape(-1.0) == -1.0 and curve.shape(0.0) == 0.0
    assert np.all(np.diff(shaped) > 0)
    assert abs(curve.shape(0.5) - 0.2375) < 1e-12
    assert InputParams(expo=0.0).shape(0.3) == 0.3


def test_the_session_shapes_the_humans_input() -> None:
    s: LiveSession = LiveSession.start(np.random.default_rng(0), input=InputParams(expo=0.7))
    s.set_input(0.5)
    assert abs(s.effective_command - 0.2375) < 1e-12
