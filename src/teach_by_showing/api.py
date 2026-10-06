"""Web layer: WebSocket for state and input, health check, static frontend
(DECISIONS.md §3, §8). Calls into live.py and nothing else; no simulation
logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import json
import logging
import os
import secrets
import signal
import time
from collections.abc import AsyncIterator, Callable
from importlib import resources
from pathlib import Path
from typing import Any, Final

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from teach_by_showing.funding import FUNDING_LOGOS, FUNDING_TEXT, FUNDING_URL, funding_asset
from teach_by_showing.live import (
    LiveSession,
    LoopStats,
    RecordingRequest,
    parse_command,
    run_paced,
)
from teach_by_showing.recording import Run, save_run

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


def create_app(
    make_session: Callable[[], LiveSession], exit_on_crash: bool = False
) -> FastAPI:
    """The app, with the session created at startup. Tests pass their own
    factory with a fixed seed.

    With `exit_on_crash`, a crash of the simulation loop shuts the server
    down, so that the container's restart policy brings it back: Docker only
    reports an unhealthy container, it does not restart one (DECISIONS.md
    §10). Tests leave it off, since it would signal the test process."""
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
        loop_task.add_done_callback(
            lambda task: _on_loop_done(task, exit_on_crash=exit_on_crash)
        )
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

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(index_page())

    @app.get("/funding/{name}")
    async def funding_logo(name: str) -> Response:
        try:
            data: bytes = funding_asset(name)
        except KeyError:
            return Response(status_code=404)
        media: str = "image/png" if name.endswith(".png") else "image/jpeg"
        return Response(data, media_type=media)

    static: str = str(resources.files("teach_by_showing") / "static")
    app.mount("/", StaticFiles(directory=static), name="static")
    return app


def index_page() -> str:
    """The start page with the funding notice filled in on the server, so
    that the wording exists once, in funding.py, and the frontend's script
    never touches it (DECISIONS.md §2)."""
    page: str = (resources.files("teach_by_showing") / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    logos: str = "".join(
        f'<img src="/funding/{name}" alt="{html.escape(alt)}">' for name, alt in FUNDING_LOGOS
    )
    notice: str = (
        f'{logos}<p>{html.escape(FUNDING_TEXT)} <a href="{FUNDING_URL}">{FUNDING_URL}</a></p>'
    )
    return page.replace("<!--FUNDING-->", notice)


def _on_loop_done(task: asyncio.Task[None], exit_on_crash: bool) -> None:
    if task.cancelled() or task.exception() is None:
        return
    log.error("simulation loop crashed", exc_info=task.exception())
    if exit_on_crash:
        # SIGTERM, not sys.exit: uvicorn shuts down cleanly on it, closing
        # connections, and the process ends for the restart policy to act.
        log.error("shutting down for a restart")
        os.kill(os.getpid(), signal.SIGTERM)


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
    parser.add_argument(
        "--record",
        metavar="LABEL",
        help="record the next transports as a run named LABEL, e.g. vorsichtig (§13)",
    )
    parser.add_argument(
        "--transports", type=int, default=3, help="transports per recording; default: 3"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        # The container sets DATA_DIR to its volume (§10).
        default=Path(os.environ.get("DATA_DIR", "data")),
        help="where runs are saved, under runs/; default: $DATA_DIR or ./data",
    )
    args: argparse.Namespace = parser.parse_args()

    seed: int = args.seed if args.seed is not None else secrets.randbits(32)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("seed %d", seed)
    runs: Path = args.data_dir / "runs"

    def saved(run: Run) -> None:
        log.info("recorded %d frames to %s", run.frames, save_run(run, runs))

    def make_session() -> LiveSession:
        session: LiveSession = LiveSession.start(np.random.default_rng(seed))
        if args.record:
            session.request_recording(
                RecordingRequest(
                    args.record,
                    seed,
                    time.strftime("%Y%m%d-%H%M%S"),
                    args.transports,
                    saved,
                )
            )
            log.info("recording %d transports from the first deflection", args.transports)
        return session

    app: FastAPI = create_app(make_session, exit_on_crash=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
