"""The stand's steps for one visitor after another (concept §6.1,
DECISIONS.md §18): an earlier agent drives while the stand waits, the
visitor demonstrates, the agent learns, the visitor's agent drives, then
joins the gallery and drives against another agent there (§23).

Each step hands its frames to a session that already exists: the live crane
for the demonstration, a ghost replay for the agents. This module only
decides which one runs and what the screen says. Like live.py it knows
nothing about HTTP and counts in frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np

from teach_by_showing.cloning import LinearPolicy, Policy
from teach_by_showing.crane import SIM_HZ
from teach_by_showing.features import features
from teach_by_showing.gallery import (
    REFERENCE_TARGETS,
    VISITOR_NAME,
    Agent,
    GalleryStore,
    Joining,
    opponent,
    reference_driving,
    table_agents,
)
from teach_by_showing.live import LiveSession, RecordingRequest, Settings, StandParams
from teach_by_showing.model_view import LearningTrace, learn, model_message
from teach_by_showing.recording import Run, run_filename
from teach_by_showing.replay import (
    AGENT_SELF,
    HEADLINE,
    HEADLINE_NEW,
    PROFILE_FIGURES,
    PROFILE_NOTE,
    ReplaySession,
    profile_cells,
)

log: logging.Logger = logging.getLogger("teach_by_showing")

# Concept §6.1: three transports are enough to learn from (DECISIONS.md §16)
# and fit the minute and a half the step may take.
TEACH_TRANSPORTS: Final[int] = 3
# After the visitor's own transports, targets the agent has never seen, to
# show that it learnt a way of driving, not three drives (§20).
AGENT_NEW_TARGETS: Final[int] = 2
# How long the steps last is in settings.stand (StandParams, live.py), so
# that it is tuned on the stand without editing code (§21).
# What a visitor's recording is called. Never a name (concept §6.3).
RUN_LABEL: Final[str] = "besuch"
# Whose drive the ghost plays while the stand waits.
TEACHER_EARLIER: Final[str] = "Vorbild"
# Frames of the reference drive per frame of the stand: about 3 ms, done
# within two seconds; in one go it stalled the loop for 0.38 s (§23).
REFERENCE_STEPS_PER_TICK: Final[int] = 30
# How long the staff's key confirms that the gallery starts afresh.
NOTICE_SECONDS: Final[float] = 3.0
GALLERY_NOTICE: Final[str] = "Galerie neu ab jetzt"


class Phase(StrEnum):
    IDLE = "idle"
    TEACH = "teach"
    LEARN = "learn"
    AGENT = "agent"
    GALLERY = "gallery"


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
    # The visitor's learning as recorded for the slow motion (§21); None
    # until a visitor has taught an agent, or for a network.
    trace: LearningTrace | None = None
    # Where agents join (§23); without one the visitor's agent drives on
    # until a button decides, as before the gallery.
    gallery: GalleryStore | None = None
    joining: Joining | None = None
    own: Agent | None = None
    rival: Agent | None = None
    table: list[Agent] = field(default_factory=list)
    # The frame from which the gallery's result is held, once driven.
    hold_from: int | None = None
    # The staff's key, as A and B: held as reported, a press decided here.
    held_reset: bool = False
    pressed_reset: bool = False
    notice_until: int = -1

    @classmethod
    def start(
        cls,
        rng: np.random.Generator,
        settings: Settings,
        showcase: Showcase,
        clock: Callable[[], str],
        on_recorded: Callable[[Run], None],
        seed: int | None = None,
        gallery: GalleryStore | None = None,
    ) -> StandSession:
        # The frontend draws every step to the one scale the config message
        # gives it; a showcase recorded under other physics would be drawn
        # wrongly and drive differently from the visitor's crane.
        if showcase.run.crane != settings.crane or showcase.run.task != settings.task:
            raise ValueError("the showcase was recorded with other crane or task parameters")
        session: StandSession = cls(
            rng, settings, showcase, clock, on_recorded, seed, gallery=gallery
        )
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

    def set_staff(self, reset: bool) -> None:
        """The staff's key: from now on the gallery shows only the agents
        taught from here (§23). Hidden, not deleted."""
        if reset and not self.held_reset:
            self.pressed_reset = True
        self.held_reset = reset

    def tick(self) -> None:
        self.frame += 1
        a, b = self.pressed_a, self.pressed_b
        self.pressed_a = self.pressed_b = False
        if self.joining is not None:
            self._join()
        if self.pressed_reset:
            self.pressed_reset = False
            if self.gallery is not None:
                self.gallery.set_since(self.clock())
                self.notice_until = self.frame + self._params.frames(NOTICE_SECONDS)
                log.info("gallery shown from now on")
        match self.phase:
            case Phase.IDLE:
                if a:
                    self._teach()
                else:
                    self._replay().tick()
            case Phase.TEACH:
                timeout: int = self._params.frames(self._params.teach_timeout)
                if b or self.frame - self.last_activity > timeout:
                    self._idle()
                    return
                assert self.live is not None
                self.live.tick()
                if self.recorded is not None:
                    self._learn(self.recorded)
            case Phase.LEARN:
                # The slow motion, then the model stays until A; B ends.
                p: StandParams = self._params
                if b:
                    self._idle()
                elif a or self._in_phase() >= p.frames(p.learn_replay + p.learn_wait):
                    self._agent()
            case Phase.AGENT:
                if a:
                    self._teach()
                elif b:
                    self._idle()
                else:
                    r: ReplaySession = self._replay()
                    r.tick()
                    # After the first round, once its last pause is over.
                    first_round: int = TEACH_TRANSPORTS + AGENT_NEW_TARGETS
                    if self.own is not None and r.transports_done >= first_round:
                        if r.pause_left == 0:
                            self._gallery()
            case Phase.GALLERY:
                if a:
                    self._teach()
                elif b:
                    self._idle()
                elif self.hold_from is None:
                    g: ReplaySession = self._replay()
                    g.tick()
                    p = self._params
                    if (
                        g.transports_done >= p.gallery_transports
                        or self._in_phase() >= p.frames(p.gallery_limit)
                    ):
                        # The picture stops on the result, banner and all.
                        g.stopped = True
                        self.hold_from = self.frame
                elif self.frame - self.hold_from >= self._params.frames(
                    self._params.gallery_hold
                ):
                    self._idle()

    @property
    def _params(self) -> StandParams:
        return self.settings.stand

    def _in_phase(self) -> int:
        """Frames since the current step began."""
        return self.frame - self.phase_frame

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self.phase_frame = self.frame

    def _idle(self) -> None:
        self.live = None
        # Learnt against naive, without the teacher's ghost: three cranes at
        # once looked busy on the stand (decided 2026-10-16, §19).
        self.replay = ReplaySession.start(
            self.showcase.run,
            self.showcase.policy,
            teacher=TEACHER_EARLIER,
            with_naive=True,
            show_ghost=False,
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
        policy, trace = learn([run], self.rng)
        self.trace = trace
        if trace is not None:
            log.info(
                "cloned a visitor's run in %.3f s, loss %.4f -> %.4f",
                trace.seconds,
                trace.losses[0],
                trace.losses[-1],
            )
        self.showcase = Showcase(run, policy)
        if self.gallery is not None:
            # Driven beside the next frames (_join), not in this one.
            number: int = self.gallery.next_number()
            s: Settings = self.settings
            self.own = None
            self.joining = Joining(
                number,
                VISITOR_NAME.format(number),
                run.created,
                run_filename(run.created, run.label),
                reference_driving(policy, s.crane, s.task),
            )
        self._enter(Phase.LEARN)

    def _join(self) -> None:
        """A few more steps of the joining agent's reference drive; once it
        is done, the agent is stored and can meet the gallery."""
        assert self.joining is not None and self.gallery is not None
        if self.joining.driving.drive(REFERENCE_STEPS_PER_TICK):
            self.own = self.joining.agent()
            self.gallery.add(self.own)
            self.joining = None
            log.info("agent #%s joined the gallery", self.own.number)

    def _agent(self) -> None:
        self.live = None
        self.replay = ReplaySession.start(
            self.showcase.run,
            self.showcase.policy,
            new_targets=AGENT_NEW_TARGETS,
            rng=self.rng,
            show_curves=True,
            show_profile=True,
        )
        self._enter(Phase.AGENT)

    def _gallery(self) -> None:
        """Against the gallery agent whose reference time differs most
        (§23); with nobody to drive against, back to waiting."""
        assert self.gallery is not None and self.own is not None
        s: Settings = self.settings
        shown: list[Agent] = self.gallery.visible(s.crane, s.task)
        rival: Agent | None = opponent(self.own, shown)
        if rival is None:
            self._idle()
            return
        self.rival = rival
        self.table = table_agents(self.own, rival, shown)
        self.hold_from = None
        self.replay = ReplaySession.start(
            self.showcase.run,
            self.own.policy,
            new_targets=max(1, self._params.gallery_transports),
            rng=self.rng,
            show_curves=True,
            teacher_transports=False,
            rival=rival.policy,
            rival_name=rival.name,
        )
        self._enter(Phase.GALLERY)

    def _gallery_table(self) -> dict[str, Any]:
        """The gallery laid across as the style profile (§22): one row per
        agent, its figures on the reference targets, the own row marked."""
        assert self.own is not None

        def label(agent: Agent) -> str:
            return f"{AGENT_SELF} (#{agent.number})" if agent is self.own else agent.name

        return {
            "title": f"Galerie – Mittel aus {REFERENCE_TARGETS} Testzielen",
            "columns": ["", *PROFILE_FIGURES],
            "rows": [[label(a), *profile_cells(a.reference)] for a in self.table],
            "highlight": self.table.index(self.own),
            "note": PROFILE_NOTE,
        }

    def _replay(self) -> ReplaySession:
        assert self.replay is not None
        return self.replay

    def _snapshot(self) -> int:
        """Which snapshot of the learning the slow motion has reached."""
        assert self.trace is not None
        replay: int = max(1, self._params.frames(self._params.learn_replay))
        progress: float = min(1.0, self._in_phase() / replay)
        return round(progress * (self.trace.points - 1))

    def _model(self) -> dict[str, Any] | None:
        """The model display (§21): learning in slow motion while the agent
        learns, at work while it drives; only for the visitor's own agent,
        and only for a linear policy."""
        policy: Policy = self.showcase.policy
        if not isinstance(policy, LinearPolicy) or self.trace is None:
            return None
        if self.phase is Phase.LEARN:
            k: int = self._snapshot()
            return model_message(self.trace.policies[k], trace=self.trace, shown=k)
        if self.phase in (Phase.AGENT, Phase.GALLERY):
            r: ReplaySession = self._replay()
            assert r.state is not None and r.transport is not None
            inputs = features(r.state, r.transport.target, r.run.crane)[0]
            return model_message(policy, inputs=inputs, output=r.command)
        return None

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
                    " · Rot: ohne Lernen – Vollgas, dann bremsen",
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
                seconds: str = (
                    f"{self.trace.seconds:.2f}".replace(".", ",") if self.trace else "0,1"
                )
                replaying: bool = self._in_phase() < self._params.frames(
                    self._params.learn_replay
                )
                return {
                    "headline": "Ihr Agent lernt aus Ihrer Fahrt",
                    "hint": f"Gelernt in {seconds} s – hier in Zeitlupe · B: beenden",
                    "prompt": None if replaying else "A: Ihr Agent fährt los",
                    **live_labels,
                }
            case Phase.AGENT:
                return {
                    "headline": HEADLINE_NEW if self._replay().on_new_target else HEADLINE,
                    "hint": "A: noch einmal vormachen · B: beenden",
                    "prompt": None,
                    **self._replay().texts(),
                }
            case Phase.GALLERY:
                assert self.rival is not None
                return {
                    "headline": f"Ihr Agent in der Galerie – gegen {self.rival.name}",
                    "hint": f"Alle fahren dieselben {REFERENCE_TARGETS} Testziele"
                    " · A: noch einmal vormachen · B: beenden",
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
        model: dict[str, Any] | None = self._model()
        if model is not None:
            message["model"] = model
        if self.phase is Phase.LEARN:
            # The last transport's "Ziel erreicht" would cover the prompt.
            message["banner"] = None
        if self.phase is Phase.GALLERY:
            message["profile"] = self._gallery_table()
        if self.frame < self.notice_until:
            message["banner"] = GALLERY_NOTICE
        return message
