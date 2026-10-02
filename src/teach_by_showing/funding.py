"""Funding notice: wording, link and logos (DECISIONS.md §2)."""

from __future__ import annotations

from importlib import resources
from typing import Final

# Verbatim from the project's funding notice, typographic quotes included, and
# character for character the same as in feature-viz and in README.md.
FUNDING_TEXT: Final[str] = (
    "The K-M-I research and development project is funded as part of the "
    "“Future of Work: Regional Competence Centers for Labor Research – "
    "Artificial Intelligence” funding initiative within the “Innovations for "
    "Tomorrow's Production, Services, and Work” program of the German Federal "
    "Ministry of Research, Technology and Space (BMFTR) and is supervised by "
    "the Project Management Agency Karlsruhe (PTKA)."
)
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
