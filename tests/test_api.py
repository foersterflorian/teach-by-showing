"""Web layer (DECISIONS.md §8). Runs the real app with its 60 Hz loop."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from teach_by_showing.api import STALL_SECONDS, create_app
from teach_by_showing.live import LiveSession


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(lambda: LiveSession.start(np.random.default_rng(0)))
    with TestClient(app) as c:
        yield c


def receive(ws: WebSocketTestSession) -> dict[str, Any]:
    message: dict[str, Any] = json.loads(ws.receive_text())
    return message


def test_health_reports_a_ticking_loop(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        receive(ws)
        receive(ws)  # the loop has ticked at least once
        r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["ticks"] > 0


def test_websocket_sends_config_then_states(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        assert receive(ws)["type"] == "config"
        frames: list[int] = [receive(ws)["frame"] for _ in range(5)]
    assert frames == sorted(frames) and len(set(frames)) == 5


def test_input_moves_the_crane_and_garbage_is_ignored(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        receive(ws)
        x0: float = receive(ws)["x"][0]
        direction: float = 1.0 if x0 < 5.0 else -1.0
        # Malformed messages first: none of them may close the connection.
        for junk in ("not json", "[1, 2]", '{"type": "input", "u": "fast"}', '{"type": "x"}'):
            ws.send_text(junk)
        state: dict[str, Any] = receive(ws)
        for _ in range(60):
            ws.send_text(json.dumps({"type": "input", "u": direction}))
            state = receive(ws)
    assert state["command"] == direction
    assert (state["x"][0] - x0) * direction > 0.1


def test_frontend_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text


class CrashingSession(LiveSession):
    def tick(self) -> None:
        if self.state.frame >= 3:
            raise RuntimeError("simulated crash")
        super().tick()


def test_health_fails_when_the_loop_has_crashed() -> None:
    """The container healthcheck must notice a dead loop, not only a dead
    server (§8)."""
    base: LiveSession = LiveSession.start(np.random.default_rng(0))
    crashing: LiveSession = CrashingSession(
        base.rng, base.crane, base.task, base.state, base.transport
    )
    with TestClient(create_app(lambda: crashing)) as c:
        time.sleep(STALL_SECONDS + 0.3)
        r = c.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "stalled"


@pytest.mark.parametrize("exit_on_crash", [True, False])
def test_crash_signals_shutdown_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, exit_on_crash: bool
) -> None:
    """Docker does not restart an unhealthy container, only an exited one;
    so the stand's server ends itself when its loop dies (§10)."""
    import asyncio
    import os
    import signal

    from teach_by_showing import api

    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))

    async def crash() -> None:
        raise RuntimeError("simulated crash")

    async def run() -> None:
        task: asyncio.Task[None] = asyncio.create_task(crash())
        await asyncio.wait([task])
        api._on_loop_done(task, exit_on_crash=exit_on_crash)  # noqa: SLF001

    asyncio.run(run())
    assert sent == ([(os.getpid(), signal.SIGTERM)] if exit_on_crash else [])
