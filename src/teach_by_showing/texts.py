"""Everything the screen says, in German and in English (DECISIONS.md §24),
and which of the two the stand speaks.

One frozen dataclass holds every text; the two languages are two instances
of it, so that a text missing in one of them is a type error, not a gap on
the screen. Templates use str.format placeholders. The module knows nothing
about HTTP or the frontend; the stand's language is kept in the data
directory (choices.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from teach_by_showing.choices import load_choice, save_choice


class Language(StrEnum):
    DE = "de"
    EN = "en"


# Decided with the maintainer on 2026-10-22: English until the staff switch.
DEFAULT_LANGUAGE: Final[Language] = Language.EN
LANGUAGE_FILE: Final[str] = "language.json"
# The two shipped gallery models (api.MODELS), by the key they are stored
# under; their names on the screen are worded below.
MODEL_CAUTIOUS: Final[str] = "cautious"
MODEL_QUICK: Final[str] = "quick"


@dataclass(frozen=True, slots=True)
class Texts:
    """One language's words. Grouped by where they appear."""

    # Number style: the decimal separator.
    decimal: str
    # The page before the first state, and the words the scripts draw.
    page_headline: str
    page_hint: str
    offline: str
    transports_label: str
    target_label: str
    reached: str
    timed_out: str
    # Readouts.
    elapsed_label: str
    command_label: str
    agent_label: str
    agent_command_label: str
    teacher_command_label: str
    # The stand's steps (§18).
    idle_headline: str
    idle_hint: str
    idle_prompt: str
    teach_headline: str
    teach_hint: str
    teach_prompt: str
    learn_headline: str
    learn_hint: str
    learn_prompt: str
    agent_hint: str
    gallery_headline: str
    gallery_hint: str
    gallery_title: str
    gallery_notice: str
    language_notice: str
    # The theme the staff switched to (§25).
    dark_notice: str
    light_notice: str
    # Who drives.
    teacher_self: str
    teacher_earlier: str
    agent_self: str
    own_in_gallery: str
    visitor_name: str
    model_cautious: str
    model_quick: str
    # The ghost replay and its banners (§17, §19, §20, §23).
    agent_headline: str
    agent_headline_new: str
    replay_hint: str
    banner_agent: str
    banner_ghost: str
    banner_naive: str
    banner_rival: str
    still_driving: str
    still_swinging: str
    curve_speed: str
    curve_swing: str
    # The style profile (§22).
    profile_title: str
    profile_title_mean: str
    profile_note: str
    profile_figures: tuple[str, str, str, str]
    not_reached: str
    partly_reached: str
    # The model display (§21), in the order of features.NAMES.
    model_labels: tuple[str, str, str]
    model_short: tuple[str, str, str]
    model_explain: str
    loss_label: str
    loss_axis: str


GERMAN: Final[Texts] = Texts(
    decimal=",",
    page_headline="Bringen Sie die Last ins Ziel – ohne dass sie pendelt",
    page_hint="Linker Stick: Katze fahren",
    offline="Verbindung zum Kran wird hergestellt …",
    transports_label="Transporte",
    target_label="Ziel",
    reached="Ziel erreicht",
    timed_out="Zeit abgelaufen",
    elapsed_label="Fahrzeit",
    command_label="Stellgröße",
    agent_label="Agent",
    agent_command_label="Stellgröße Agent",
    teacher_command_label="Stellgröße {teacher}",
    idle_headline="Bringen Sie einer KI das Kranfahren bei",
    idle_hint="Grün: ein Agent, der von einem Menschen gelernt hat"
    " · Rot: ohne Lernen – Vollgas, dann bremsen",
    idle_prompt="Übernehmen Sie den Kran! Drücken Sie A",
    teach_headline="Zeigen Sie Ihrem Agenten, wie Sie fahren – Transport {n} von {total}",
    teach_hint="Linker Stick: Katze fahren · B: abbrechen",
    teach_prompt="Fahren Sie die Last ins Ziel – vorsichtig oder zügig, ganz nach Ihrem Stil",
    learn_headline="Ihr Agent lernt aus Ihrer Fahrt",
    learn_hint="Gelernt in {seconds} s – hier in Zeitlupe · B: beenden",
    learn_prompt="A: Ihr Agent fährt los",
    agent_hint="A: noch einmal vormachen · B: beenden",
    gallery_headline="Ihr Agent in der Galerie – gegen {rival}",
    gallery_hint="Alle fahren dieselben {targets} Testziele"
    " · A: noch einmal vormachen · B: beenden",
    gallery_title="Galerie – Mittel aus {targets} Testzielen",
    gallery_notice="Galerie neu ab jetzt",
    language_notice="Deutsch",
    dark_notice="Dunkel",
    light_notice="Hell",
    teacher_self="Ihre Fahrt",
    teacher_earlier="Vorbild",
    agent_self="Ihr Agent",
    own_in_gallery="Ihr Agent (#{number})",
    visitor_name="Kranführer·in #{number}",
    model_cautious="Vorbild vorsichtig",
    model_quick="Vorbild zügig",
    agent_headline="Ihr Agent fährt – Ihre eigene Fahrt läuft als Schatten mit",
    agent_headline_new="Ihr Agent fährt ein neues Ziel – das hat er nie gesehen",
    replay_hint="Wiedergabe – das Gamepad hat Pause",
    banner_agent="Agent {agent}",
    banner_ghost="Agent {agent} · {teacher} {ghost}",
    banner_naive="Gelernt {agent} · Ohne Lernen {naive}",
    banner_rival="Ihr Agent {agent} · {rival} {rival_time}",
    still_driving="fährt noch",
    still_swinging="pendelt noch",
    curve_speed="Geschwindigkeit",
    curve_swing="Pendelwinkel",
    profile_title="Stil-Steckbrief",
    profile_title_mean="Stil-Steckbrief – Mittel aus {both} von {total} Transporten",
    profile_note="Kleiner ist ruhiger bzw. schonender.",
    profile_figures=(
        "Fahrzeit",
        "Größtes Pendeln",
        "Ruhe am Ziel (Restpendeln)",
        "Schonung (Stickweg)",
    ),
    not_reached="nicht erreicht",
    partly_reached="{time} · {missed}× nicht erreicht",
    model_labels=("Abstand zum Ziel", "Pendelwinkel", "Pendelgeschwindigkeit"),
    model_short=("Abstand", "Winkel", "Winkelgeschw."),
    model_explain="Je größer das Gewicht beim Abstand,"
    " desto zügiger fährt der Agent aufs Ziel zu.",
    loss_label="Abweichung von Ihrer Fahrt",
    loss_axis="Lernschritte",
)

# British English (decided 2026-10-22).
ENGLISH: Final[Texts] = Texts(
    decimal=".",
    page_headline="Bring the load to the target – without letting it swing",
    page_hint="Left stick: move the trolley",
    offline="Connecting to the crane …",
    transports_label="Transports",
    target_label="Target",
    reached="Target reached",
    timed_out="Time is up",
    elapsed_label="Time",
    command_label="Command",
    agent_label="Agent",
    agent_command_label="Command: agent",
    teacher_command_label="Command: {teacher}",
    idle_headline="Teach an AI to drive a crane",
    idle_hint="Green: an agent that learnt from a person"
    " · Red: no learning – full speed, then brake",
    idle_prompt="Take over the crane! Press A",
    teach_headline="Show your agent how you drive – transport {n} of {total}",
    teach_hint="Left stick: move the trolley · B: cancel",
    teach_prompt="Bring the load to the target – carefully or briskly, in your own style",
    learn_headline="Your agent learns from your drive",
    learn_hint="Learnt in {seconds} s – shown here in slow motion · B: finish",
    learn_prompt="A: your agent sets off",
    agent_hint="A: demonstrate again · B: finish",
    gallery_headline="Your agent in the gallery – against {rival}",
    gallery_hint="All drive the same {targets} test targets"
    " · A: demonstrate again · B: finish",
    gallery_title="Gallery – mean over {targets} test targets",
    gallery_notice="Gallery starts afresh now",
    language_notice="English",
    dark_notice="Dark",
    light_notice="Light",
    teacher_self="Your drive",
    teacher_earlier="Role model",
    agent_self="Your agent",
    own_in_gallery="Your agent (#{number})",
    visitor_name="Crane operator #{number}",
    model_cautious="Role model, cautious",
    model_quick="Role model, quick",
    agent_headline="Your agent drives – your own drive runs alongside as a shadow",
    agent_headline_new="Your agent drives to a new target – one it has never seen",
    replay_hint="Replay – the gamepad is resting",
    banner_agent="Agent {agent}",
    banner_ghost="Agent {agent} · {teacher} {ghost}",
    banner_naive="Learnt {agent} · Without learning {naive}",
    banner_rival="Your agent {agent} · {rival} {rival_time}",
    still_driving="still driving",
    still_swinging="still swinging",
    curve_speed="Speed",
    curve_swing="Swing angle",
    profile_title="Style profile",
    profile_title_mean="Style profile – mean over {both} of {total} transports",
    profile_note="Smaller means calmer or gentler.",
    profile_figures=(
        "Time",
        "Largest swing",
        # As short as the German, so that the headers wrap into the same two
        # lines and the scene keeps its size in both languages: beside the
        # readouts a column has about 150 px, and "Calm (residual swing)"
        # took three (measured 2026-10-22, §24).
        "Calm (final swing)",
        "Gentleness (stick travel)",
    ),
    not_reached="not reached",
    partly_reached="{time} · {missed}× not reached",
    model_labels=("Distance to target", "Swing angle", "Swing speed"),
    model_short=("Distance", "Angle", "Ang. speed"),
    model_explain="The larger the weight on distance,"
    " the more briskly the agent heads for the target.",
    loss_label="Deviation from your drive",
    loss_axis="Training steps",
)

TEXTS: Final[dict[Language, Texts]] = {Language.DE: GERMAN, Language.EN: ENGLISH}


def number(value: float, digits: int, words: Texts, unit: str = "") -> str:
    """The one place a decimal separator is chosen."""
    return f"{value:.{digits}f}".replace(".", words.decimal) + unit


def other(language: Language) -> Language:
    """The language the staff's key switches to."""
    return Language.EN if language is Language.DE else Language.DE


def load_language(directory: Path) -> Language:
    """The stand's language as last chosen, or the default (choices.py)."""
    return load_choice(directory / LANGUAGE_FILE, "language", Language, DEFAULT_LANGUAGE)


def save_language(directory: Path, language: Language) -> None:
    save_choice(directory / LANGUAGE_FILE, "language", language)
