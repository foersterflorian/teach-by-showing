"""Web layer: WebSocket for state and input, health check, static frontend
(DECISIONS.md §3, §8). Calls into live.py and nothing else; no simulation
logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from importlib import resources
from typing import Any, Final

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from teach_by_showing.live import LiveSession, LoopStats, parse_command, run_paced

log: logging.Logger = logging.getLogger("teach_by_showing")

DEFAULT_PORT: Final[int] = 8081  # feature-viz keeps 8080 (DECISIONS.md §3)
# The loop ticks every 17 ms; a second without a tick means it is stuck.
STALL_SECONDS: Final[float] = 1.0


class Outbox:
    """Latest message for one client. A client that falls behind skips
    frames instead of queueing them, and the loop never waits for it."""

    def __init__(self) -> None:
        self._latest: str = ""
        self._ready: asyncio.Event = asyncio.Event()

    def offer(self, text: str) -> None:
        self._latest = text
        self._ready.set()

    async def take(self) -> str:
        await self._ready.wait()
        self._ready.clear()
        return self._latest


def create_app(make_session: Callable[[], LiveSession]) -> FastAPI:
    """The app, with the session created at startup. Tests pass their own
    factory with a fixed seed."""
    outboxes: set[Outbox] = set()
    stats: LoopStats = LoopStats()
    holder: dict[str, LiveSession] = {}

    def publish(message: dict[str, Any]) -> None:
        text: str = json.dumps(message)
        for box in outboxes:
            box.offer(text)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        holder["session"] = make_session()
        loop_task: asyncio.Task[None] = asyncio.create_task(
            run_paced(holder["session"], publish, stats)
        )
        # A crash in the loop would otherwise vanish silently with its task.
        # It is logged, /healthz turns 503, and the container restarts.
        loop_task.add_done_callback(_log_crash)
        try:
            yield
        finally:
            # A loop that has crashed is already logged; only a live one is
            # cancelled and awaited.
            if not loop_task.done():
                loop_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await loop_task

    app: FastAPI = FastAPI(title="teach-by-showing", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """For the container healthcheck: fails when the simulation loop
        has stopped ticking, not merely when the server is down."""
        now: float = asyncio.get_running_loop().time()
        alive: bool = now - stats.last_tick < STALL_SECONDS
        body: dict[str, Any] = {
            "status": "ok" if alive else "stalled",
            "ticks": stats.ticks,
            "overruns": stats.overruns,
            "clients": len(outboxes),
        }
        return JSONResponse(body, status_code=200 if alive else 503)

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        session: LiveSession = holder["session"]
        await socket.send_text(json.dumps(session.config_message()))
        box: Outbox = Outbox()
        outboxes.add(box)

        async def send() -> None:
            while True:
                await socket.send_text(await box.take())

        async def receive() -> None:
            while True:
                handle_input(session, await socket.receive_text())

        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(send()),
            asyncio.create_task(receive()),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            outboxes.discard(box)
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await t

    static: str = str(resources.files("teach_by_showing") / "static")
    app.mount("/", StaticFiles(directory=static, html=True), name="static")
    return app


def _log_crash(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("simulation loop crashed", exc_info=task.exception())


def handle_input(session: LiveSession, text: str) -> None:
    """Apply one message from the browser. Anything malformed is dropped
    and logged; a bad message must not close the connection."""
    try:
        message: object = json.loads(text)
    except json.JSONDecodeError:
        log.warning("dropped non-JSON message")
        return
    if not isinstance(message, dict) or message.get("type") != "input":
        log.warning("dropped message of unknown type")
        return
    command: float | None = parse_command(message.get("u"))
    if command is None:
        log.warning("dropped input without a finite command")
        return
    session.set_input(command)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Serve the live crane and its frontend."
    )
    # localhost by default: the Gamepad API needs a secure context, and the
    # kiosk browser runs on the same machine (DECISIONS.md §3). The container
    # binds 0.0.0.0 and publishes the port to localhost only.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--seed", type=int, default=None, help="random seed; default: fresh")
    args: argparse.Namespace = parser.parse_args()

    seed: int = args.seed if args.seed is not None else secrets.randbits(32)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("seed %d", seed)
    app: FastAPI = create_app(lambda: LiveSession.start(np.random.default_rng(seed)))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
