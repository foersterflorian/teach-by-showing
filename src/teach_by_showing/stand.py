"""The stand's steps for one visitor after another (concept §6.1,
DECISIONS.md §18): an earlier agent drives while the stand waits, the
visitor demonstrates, the agent learns, the visitor's agent drives.

Each step hands its frames to a session that already exists: the live crane
for the demonstration, a ghost replay for the agents. This module only
decides which one runs and what the screen says. Like live.py it knows
nothing about HTTP and counts in frames.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

from teach_by_showing.cloning import Policy, clone
from teach_by_showing.crane import SIM_HZ
from teach_by_showing.live import LiveSession, RecordingRequest, Settings
from teach_by_showing.recording import Run
from teach_by_showing.replay import HEADLINE, ReplaySession

log: logging.Logger = logging.getLogger("teach_by_showing")

# Concept §6.1: three transports are enough to learn from (DECISIONS.md §16)
# and fit the minute and a half the step may take.
TEACH_TRANSPORTS: Final[int] = 3
# A visitor who walks away mid-demonstration leaves the stand to the next.
# Concept §6.3 asks for 30–60 s; 45 s is the middle, to be judged on the stand.
TEACH_TIMEOUT_FRAMES: Final[int] = 45 * SIM_HZ
# Training takes 0.1 s at most (§15); the step is held long enough to read.
LEARN_FRAMES: Final[int] = 2 * SIM_HZ
# What a visitor's recording is called. Never a name (concept §6.3).
RUN_LABEL: Final[str] = "besuch"
# Whose drive the ghost plays while the stand waits.
TEACHER_EARLIER: Final[str] = "Vorbild"


class Phase(StrEnum):
    IDLE = "idle"
    TEACH = "teach"
    LEARN = "learn"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class Showcase:
    """An agent with the drive it was cloned from: what the stand shows
    while it waits, until the next visitor has taught one."""

    run: Run
    policy: Policy


@dataclass(slots=True)
class StandSession:
    """One visitor after another. Mutable on purpose, like LiveSession: the
    single owner of the stand's state, ticked only by the pacing loop."""

    rng: np.random.Generator
    settings: Settings
    showcase: Showcase
    # The wall clock's stamp for a recording's file name, and what to do with
    # a finished recording; both from the caller, as in live.py.
    clock: Callable[[], str]
    on_recorded: Callable[[Run], None]
    seed: int | None = None
    phase: Phase = Phase.IDLE
    replay: ReplaySession | None = None
    live: LiveSession | None = None
    recorded: Run | None = None
    # Frames since the stand started. The sessions it switches between each
    # count their own; the browser interpolates by frame number (§9) and must
    # never see time go backwards.
    frame: int = 0
    phase_frame: int = 0
    last_activity: int = 0
    # Buttons as last reported, and presses not yet acted on. The browser
    # reports what is held; a press is decided here, so that the frontend
    # holds no logic (§3).
    held_a: bool = False
    held_b: bool = False
    pressed_a: bool = False
    pressed_b: bool = False

    @classmethod
    def start(
        cls,
        rng: np.random.Generator,
        settings: Settings,
        showcase: Showcase,
        clock: Callable[[], str],
        on_recorded: Callable[[Run], None],
        seed: int | None = None,
    ) -> StandSession:
        # The frontend draws every step to the one scale the config message
        # gives it; a showcase recorded under other physics would be drawn
        # wrongly and drive differently from the visitor's crane.
        if showcase.run.crane != settings.crane or showcase.run.task != settings.task:
            raise ValueError("the showcase was recorded with other crane or task parameters")
        session: StandSession = cls(rng, settings, showcase, clock, on_recorded, seed)
        session._idle()
        return session

    def set_input(self, deflection: float) -> None:
        if deflection != 0.0:
            self.last_activity = self.frame
        if self.phase is Phase.TEACH:
            assert self.live is not None
            self.live.set_input(deflection)

    def set_buttons(self, a: bool, b: bool) -> None:
        if a and not self.held_a:
            self.pressed_a = True
        if b and not self.held_b:
            self.pressed_b = True
        if a or b:
            self.last_activity = self.frame
        self.held_a, self.held_b = a, b

    def tick(self) -> None:
        self.frame += 1
        a, b = self.pressed_a, self.pressed_b
        self.pressed_a = self.pressed_b = False
        match self.phase:
            case Phase.IDLE:
                if a:
                    self._teach()
                else:
                    self._replay().tick()
            case Phase.TEACH:
                if b or self.frame - self.last_activity > TEACH_TIMEOUT_FRAMES:
                    self._idle()
                    return
                assert self.live is not None
                self.live.tick()
                if self.recorded is not None:
                    self._learn(self.recorded)
            case Phase.LEARN:
                if self.frame - self.phase_frame >= LEARN_FRAMES:
                    self._agent()
            case Phase.AGENT:
                if a:
                    self._teach()
                elif b:
                    self._idle()
                else:
                    self._replay().tick()

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_frame = self.frame

    def _idle(self) -> None:
        self.live = None
        # The naive driver only here, for now (decided 2026-10-16, §19).
        self.replay = ReplaySession.start(
            self.showcase.run, self.showcase.policy, teacher=TEACHER_EARLIER, with_naive=True
        )
        self._enter(Phase.IDLE)

    def _teach(self) -> None:
        s: Settings = self.settings
        self.recorded = None
        self.replay = None
        self.live = LiveSession.start(self.rng, s.crane, s.task, s.input)
        self.live.request_recording(
            RecordingRequest(RUN_LABEL, self.seed, self.clock(), TEACH_TRANSPORTS, self._done)
        )
        self.last_activity = self.frame
        self._enter(Phase.TEACH)

    def _done(self, run: Run) -> None:
        self.recorded = run
        self.on_recorded(run)

    def _learn(self, run: Run) -> None:
        # Inside the loop, not in a worker: the linear policy trains in
        # 0.1 s at most, while the crane stands still (DECISIONS.md §18).
        begin: float = time.perf_counter()
        policy, losses = clone([run], self.rng)
        log.info(
            "cloned a visitor's run in %.3f s, loss %.4f -> %.4f",
            time.perf_counter() - begin,
            losses[0],
            losses[-1],
        )
        self.showcase = Showcase(run, policy)
        self._enter(Phase.LEARN)

    def _agent(self) -> None:
        self.live = None
        self.replay = ReplaySession.start(self.showcase.run, self.showcase.policy)
        self._enter(Phase.AGENT)

    def _replay(self) -> ReplaySession:
        assert self.replay is not None
        return self.replay

    def _transports_taught(self) -> int:
        """Finished transports of the demonstration under way."""
        live: LiveSession | None = self.live
        assert live is not None
        if live.recording is not None:
            return live.transports_done - live.recording_base
        # Before the first deflection nothing is recorded yet; after the
        # last transport the request is gone.
        return 0 if live.request is not None else TEACH_TRANSPORTS

    def _texts(self) -> dict[str, str | None]:
        """Everything the screen says, in German, from the server (§17)."""
        live_labels: dict[str, str | None] = {
            "elapsed_label": "Fahrzeit",
            "ghost_label": None,
            "command_label": "Stellgröße",
            "ghost_command_label": None,
        }
        match self.phase:
            case Phase.IDLE:
                return {
                    "headline": "Bringen Sie einer KI das Kranfahren bei",
                    "hint": "Grün: ein Agent, der von einem Menschen gelernt hat"
                    " · Schatten: dessen Fahrt · Rot: ohne Lernen – Vollgas, dann bremsen",
                    "prompt": "Übernehmen Sie den Kran! Drücken Sie A",
                    **self._replay().texts(),
                }
            case Phase.TEACH:
                n: int = min(self._transports_taught() + 1, TEACH_TRANSPORTS)
                assert self.live is not None
                waiting: bool = self.live.recording is None and self.live.request is not None
                return {
                    "headline": "Zeigen Sie Ihrem Agenten, wie Sie fahren"
                    f" – Transport {n} von {TEACH_TRANSPORTS}",
                    "hint": "Linker Stick: Katze fahren · B: abbrechen",
                    "prompt": "Fahren Sie die Last ins Ziel – vorsichtig oder zügig,"
                    " ganz nach Ihrem Stil"
                    if waiting
                    else None,
                    **live_labels,
                }
            case Phase.LEARN:
                return {
                    "headline": "Ihr Agent lernt aus Ihrer Fahrt",
                    "hint": "",
                    "prompt": "Ihr Agent lernt …",
                    **live_labels,
                }
            case Phase.AGENT:
                return {
                    "headline": HEADLINE,
                    "hint": "A: noch einmal vormachen · B: beenden",
                    "prompt": None,
                    **self._replay().texts(),
                }

    def config_message(self) -> dict[str, Any]:
        """The scale and the parameters, the same in every step (checked in
        start()), so that it can be sent once per connection."""
        s: Settings = self.settings
        return {
            "type": "config",
            "hz": SIM_HZ,
            "crane": s.crane.to_dict(),
            "task": s.task.to_dict(),
            "input": s.input.to_dict(),
            "mode": "stand",
        }

    def state_message(self) -> dict[str, Any]:
        """The running session's state, with the stand's frame count, its
        step and its texts. Texts travel with every state, not once with the
        config: the steps change while a browser stays connected, and the
        outbox keeps only the latest message (§8)."""
        message: dict[str, Any] = (
            self.live.state_message() if self.live else self._replay().state_message()
        )
        message["frame"] = self.frame
        message["phase"] = str(self.phase)
        message["texts"] = self._texts()
        if self.phase is Phase.LEARN:
            # The last transport's "Ziel erreicht" would cover the prompt.
            message["banner"] = None
        return message
