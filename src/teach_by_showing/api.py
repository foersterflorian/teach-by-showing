"""Web layer: WebSocket for state and input, health check, static frontend
(DECISIONS.md §3, §8). Builds a session from the engine (live.py, stand.py,
replay.py) and calls into it; no simulation logic lives here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import html
import json
import logging
import os
import re
import secrets
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib import resources
from pathlib import Path
from typing import Any, Final

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from teach_by_showing.cloning import clone
from teach_by_showing.funding import FUNDING_LOGOS, FUNDING_TEXT, FUNDING_URL, funding_asset
from teach_by_showing.gallery import Agent, GalleryStore, make_agent
from teach_by_showing.live import (
    LiveSession,
    LoopStats,
    RecordingRequest,
    Session,
    Settings,
    parse_command,
    run_paced,
)
from teach_by_showing.recording import Run, load_run, save_run
from teach_by_showing.replay import ReplaySession
from teach_by_showing.stand import Showcase, StandSession
from teach_by_showing.texts import (
    DEFAULT_LANGUAGE,
    MODEL_CAUTIOUS,
    MODEL_QUICK,
    TEXTS,
    Language,
    Texts,
    load_language,
    save_language,
)
from teach_by_showing.theme import DEFAULT_THEME, Theme, load_theme, save_theme

log: logging.Logger = logging.getLogger("teach_by_showing")

DEFAULT_PORT: Final[int] = 8081  # feature-viz keeps 8080 (DECISIONS.md §3)
# The loop ticks every 17 ms; a second without a tick means it is stuck.
STALL_SECONDS: Final[float] = 1.0
# What the stand shows until its first visitor has taught an agent: the
# maintainer's quick drive of 2026-10-09, a copy of the one in tests/data/
# (DECISIONS.md §18).
SHOWCASE_RUN: Final[str] = "zuegig.json"
# The gallery's models until colleagues have recorded more: the maintainer's
# two teaching drives, each cloned with a fixed seed (DECISIONS.md §23),
# stored under a key that texts.py words in the screen's language (§24).
MODELS: Final[tuple[tuple[str, str], ...]] = (
    ("vorsichtig.json", MODEL_CAUTIOUS),
    ("zuegig.json", MODEL_QUICK),
)
# A name in double braces in index.html: a field of texts.Texts, or the
# page's language or theme.
PAGE_WORD: Final[re.Pattern[str]] = re.compile(r"\{\{(\w+)\}\}")


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
    make_session: Callable[[], Session],
    exit_on_crash: bool = False,
    language: Language = DEFAULT_LANGUAGE,
    theme: Theme = DEFAULT_THEME,
) -> FastAPI:
    """The app, with the session created at startup. Tests pass their own
    factory with a fixed seed.

    The page speaks the stand's language and shows its theme, which the
    staff may switch; a session without the stand's steps uses `language`
    and `theme` (§24, §25).

    With `exit_on_crash`, a crash of the simulation loop shuts the server
    down, so that the container's restart policy brings it back: Docker only
    reports an unhealthy container, it does not restart one (DECISIONS.md
    §10). Tests leave it off, since it would signal the test process."""
    outboxes: set[Outbox] = set()
    stats: LoopStats = LoopStats()
    holder: dict[str, Session] = {}

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

    @app.middleware("http")
    async def revalidate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Every file is checked with the server before it is used. Without
        this a browser kept an old stylesheet after an update, and the new
        canvas of §20 grew without bound under it (2026-10-16); with ETags
        an unchanged file costs a short 304."""
        response: Response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

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
        session: Session = holder["session"]
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
        session: Session | None = holder.get("session")
        if isinstance(session, StandSession):
            return HTMLResponse(index_page(session.language, session.theme))
        return HTMLResponse(index_page(language, theme))

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


def index_page(language: Language, theme: Theme = DEFAULT_THEME) -> str:
    """The start page with its words and the funding notice filled in on the
    server, in `language`, so that every wording exists once, in texts.py and
    funding.py, and the frontend's scripts never word anything (§2, §24)."""
    page: str = (resources.files("teach_by_showing") / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    words: Texts = TEXTS[language]

    def word(match: re.Match[str]) -> str:
        name: str = match[1]
        if name in ("language", "theme"):
            return str(language if name == "language" else theme)
        value: object = getattr(words, name)
        assert isinstance(value, str), name
        return html.escape(value)

    page = PAGE_WORD.sub(word, page)
    logos: str = "".join(
        f'<img src="/funding/{name}" alt="{html.escape(alt)}">' for name, alt in FUNDING_LOGOS
    )
    notice: str = (
        f"{logos}<p>{html.escape(FUNDING_TEXT[language])}"
        f' <a href="{FUNDING_URL}">{FUNDING_URL}</a></p>'
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


def handle_input(session: Session, text: str) -> None:
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
    # Anything but true is a button not held; older pages send none.
    session.set_buttons(message.get("a") is True, message.get("b") is True)
    # The staff's keys exist only on the stand (§23, §24).
    if isinstance(session, StandSession):
        session.set_staff(message.get("reset") is True)
        session.set_language_key(message.get("language") is True)
        session.set_theme_key(message.get("theme") is True)


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
        "--params",
        type=Path,
        metavar="FILE",
        help="JSON file of crane, task and input parameters that differ from the defaults",
    )
    parser.add_argument(
        "--record",
        metavar="LABEL",
        help="record the next transports as a run named LABEL, e.g. vorsichtig (§13)",
    )
    parser.add_argument(
        "--ghost",
        type=Path,
        metavar="RUN",
        help="replay: an agent cloned from RUN drives its transports beside the recording",
    )
    parser.add_argument(
        "--transports", type=int, default=3, help="transports per recording; default: 3"
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="drive freely, one transport after another, without the stand's steps",
    )
    parser.add_argument(
        "--showcase",
        type=Path,
        metavar="RUN",
        help="the run whose agent the stand shows until a visitor has taught one;"
        " default: a quick drive shipped with the package (§18)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        # The container sets DATA_DIR to its volume (§10).
        default=Path(os.environ.get("DATA_DIR", "data")),
        help="where runs and the gallery are saved, under runs/ and agents/;"
        " default: $DATA_DIR or ./data",
    )
    args: argparse.Namespace = parser.parse_args()

    seed: int = args.seed if args.seed is not None else secrets.randbits(32)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("seed %d", seed)
    runs: Path = args.data_dir / "runs"
    language: Language = load_language(args.data_dir)
    theme: Theme = load_theme(args.data_dir)
    log.info("screen language %s, theme %s", language, theme)
    settings: Settings = Settings.from_dict(
        json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    )
    if args.params:
        log.info("parameters from %s: %s", args.params, settings)

    def saved(run: Run) -> None:
        log.info("recorded %d frames to %s", run.frames, save_run(run, runs))

    def make_replay() -> ReplaySession:
        # The teacher's own crane and task, not --params: the ghost must
        # move under the physics it was recorded with.
        run: Run = load_run(args.ghost)
        start: float = time.perf_counter()
        policy, losses = clone([run], np.random.default_rng(seed))
        log.info(
            "cloned %s in %.2f s, loss %.4f -> %.4f",
            args.ghost,
            time.perf_counter() - start,
            losses[0],
            losses[-1],
        )
        return ReplaySession.start(run, policy, words=TEXTS[language], show_curves=True)

    def make_session() -> LiveSession:
        session: LiveSession = LiveSession.start(
            np.random.default_rng(seed), settings.crane, settings.task, settings.input
        )
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

    def make_stand() -> StandSession:
        # Unpacked on disk, as the static files are.
        package: str = str(resources.files("teach_by_showing"))
        shipped: Path = Path(package) / "showcase" / SHOWCASE_RUN
        run: Run = load_run(args.showcase or shipped)
        rng: np.random.Generator = np.random.default_rng(seed)
        policy, _ = clone([run], rng)
        models: list[Agent] = []
        for name, title in MODELS:
            model: Run = load_run(Path(package) / "showcase" / name)
            model_policy, _ = clone([model], np.random.default_rng(0))
            models.append(
                make_agent(
                    None,
                    title,
                    model.created,
                    None,
                    model_policy,
                    settings.crane,
                    settings.task,
                )
            )
        return StandSession.start(
            rng,
            settings,
            Showcase(run, policy),
            lambda: time.strftime("%Y%m%d-%H%M%S"),
            saved,
            seed,
            gallery=GalleryStore(args.data_dir, tuple(models)),
            language=language,
            on_language=lambda chosen: save_language(args.data_dir, chosen),
            theme=theme,
            on_theme=lambda chosen: save_theme(args.data_dir, chosen),
        )

    if args.ghost and args.record:
        parser.error("--ghost replays a recording; it cannot record at the same time")
    make: Callable[[], Session] = make_stand
    if args.ghost:
        make = make_replay
    elif args.free or args.record:
        make = make_session
    app: FastAPI = create_app(make, exit_on_crash=True, language=language, theme=theme)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
