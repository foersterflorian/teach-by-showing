"""Funding notice: wording, link and logos (DECISIONS.md §2)."""

from __future__ import annotations

from importlib import resources
from typing import Final

from teach_by_showing.texts import Language

# The German text verbatim from the funder's material, German quotes
# included; the English one is our own translation of it, in British
# English (decided 2026-10-22, §24), and character for character the same as
# in README.md. The screen shows the one of its language.
FUNDING_TEXT: Final[dict[Language, str]] = {
    Language.DE: (
        "Das Forschungs- und Entwicklungsprojekt KMI wird im Rahmen der Fördermaßnahme "
        "„Zukunft der Arbeit: Regionale Kompetenzzentren der Arbeitsforschung – "
        "Künstliche Intelligenz“ im Programm „Innovationen für die Produktion, "
        "Dienstleistung und Arbeit von morgen“ des Bundesministeriums für Forschung, "
        "Technologie und Raumfahrt (BMFTR) gefördert und vom Projektträger Karlsruhe "
        "(PTKA) betreut."
    ),
    Language.EN: (
        "The KMI research and development project is funded as part of the "
        "“Future of Work: Regional Competence Centres for Labour Research – "
        "Artificial Intelligence” funding initiative within the “Innovations for "
        "Tomorrow's Production, Services, and Work” programme of the German Federal "
        "Ministry of Research, Technology and Space (BMFTR) and is supervised by "
        "the Project Management Agency Karlsruhe (PTKA)."
    ),
}
FUNDING_URL: Final[str] = "https://kmi-netzwerk.org/kmi-projekt/"

# (file in teach_by_showing/funding/, alt text). The files are the funder's
# originals, unmodified: logo guidelines forbid cropping the protected margin
# or recolouring, so they are only ever scaled.
FUNDING_LOGOS: Final[tuple[tuple[str, str], ...]] = (
    (
        "BMFTR_de_Web_RGB_gef_durch.jpg",
        "Funded by the German Federal Ministry of Research, Technology and Space",
    ),
    (
        "Logo_Kompetenzzentren_Arbeitsforschung.png",
        "Regional Competence Centres of Work Research (ReKodA)",
    ),
)


def funding_asset(name: str) -> bytes:
    """Raw bytes of a logo, read from the installed package rather than the
    repository, so that a wheel or container install finds them too.

    Only the names in FUNDING_LOGOS are served; anything else is refused, so
    that a name taken from a request can never reach another file."""
    if name not in {n for n, _ in FUNDING_LOGOS}:
        raise KeyError(name)
    return (resources.files("teach_by_showing") / "funding" / name).read_bytes()
