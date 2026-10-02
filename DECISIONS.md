# Design Decisions

**Project:** teach-by-showing — crane demonstrator for imitation learning (TUC-KMI)
**Status:** repository set up; first spikes pending
**Last updated:** 2026-10-02

This file records *why* the code looks the way it does. The code says what it
does; this says what was considered and rejected, so that changing something
here is a deliberate act rather than an accident.

Read this before modifying the code. If you change one of these decisions,
update the corresponding section rather than deleting it — the rejected
alternatives are the useful part.

Scope, audience and priorities come from the concept (German, internal, not
part of this repository), kept in `../Konzept/` as a separate local Git
repository; the highest document version is authoritative. References of the
form "concept §n" point to its version V02. This file must remain
understandable without it.

---

## 0. What this demonstrator shows

A side view of an overhead crane: a trolley runs along a girder, the load hangs
on a rope and swings when the trolley accelerates or brakes. The task is to
bring the load to a target and have it hang still there.

A visitor drives a few transports with a gamepad. From that demonstration a
small multilayer perceptron learns, within seconds, to map the crane's state
to the visitor's commands (behavioural cloning). The learned agent then drives
new transports in the visitor's style, shown against a ghost replay of the
visitor's own run. An evolution strategy can refine the agent towards goals —
fast, calm, gentle, and faithful to the teacher — that the human weights.

The message is the mechanism, not the crane: people teach by showing rather
than by writing rules, and people decide what counts as a good result. The
crane is deliberately not modelled on any partner's process (concept §3, §4).

It is shown at a staffed exhibition stand on 3–4 November 2026, next to
`feature-viz`, which shows what a detection network sees. One is watched, this
one is played.

---

## 1. Licence: MIT

**Decision.** MIT, the author's default.

**Why it holds here, unlike in feature-viz.** feature-viz had to become
AGPL-3.0-or-later because it is built around ultralytics and serves its output
over HTTP (feature-viz `DECISIONS.md` §16). Nothing comparable applies here:
the planned dependencies — NumPy, FastAPI/Starlette, uvicorn — are all
permissive (BSD-3-Clause or MIT). Any new dependency must keep it that way; a
copyleft one reopens this section.

**A consequence worth stating.** MIT permits closed reuse, including by an
industry partner. For a transfer-oriented research demonstrator that is
intended.

**Excluded.** The two funding logos (§2) are third-party marks and are excluded
from the licence in `README.md`.

**Open, not decided here** (as in feature-viz §16). Whether copyright sits with
the author or with TU Chemnitz as the employer, and whether the funder imposes
licence conditions beyond the acknowledgement, was not assessed. Both questions
belong to the institution.

`tests/test_licence.py` guards that `pyproject.toml`, `LICENSE` and the logo
exclusion agree.

---

## 2. Funding notice

**Context.** The funding terms ask for an acknowledgement with the funder's
logo wherever project results are shown. feature-viz settled how (its §17);
this project follows it so that the two demonstrators on the stand read as a
pair.

**Decision.** The same wording and the same two logos, byte-identical copies of
feature-viz's files, kept as package data in `src/teach_by_showing/funding/`
and read through `importlib.resources`. `funding.py` holds the text, the link
and the logo list. The notice appears in `README.md` and, once the frontend
exists, as a white strip along the bottom of the screen.

**Details that are deliberate.**

- *Verbatim text, one wording everywhere.* `FUNDING_TEXT` matches the README
  character for character, typographic quotes and dash included.
- *White ground.* The BMFTR logo must sit on white or a very light colour;
  the screen is otherwise dark.
- *Files unmodified, only scaled.* The protected margin is part of the mark.
- *Allow-list.* `funding_asset()` serves only the names in `FUNDING_LOGOS`,
  so a name taken from an HTTP request can never reach another file.

`tests/test_funding.py` guards the verbatim match, the README references and
the allow-list.

---

## 3. Architecture: Python backend in a container, browser frontend

**Context.** The stand machine runs both demonstrators as Docker containers,
one monitor each. The visitor uses a USB gamepad. The concept first planned a
pygame window.

**Decision.** All logic — simulation, learning, evolution, recording — runs in
Python inside the container. The display is a browser on the host in kiosk
mode, connected over a WebSocket: the backend sends the state at the
simulation rate, the browser renders it and sends back the gamepad commands it
reads through the Gamepad API.

```
container (Python)                         host browser (kiosk, one monitor)
------------------                         ---------------------------------
simulation, 60 Hz      -- state ------->   canvas renderer, UI, plots
training, evolution    <-- commands ----   Gamepad API (USB gamepad on the host)
runs, agents, gallery (JSON on a volume)
```

**Why.**

- *No display or device access from the container.* A window from a container
  needs the X11 or Wayland socket, and under Wayland goes through XWayland;
  feature-viz rejected exactly that as "the single most fragile part of the
  setup" (its §7). A gamepad would additionally need `/dev/input` passed
  through. Here the container needs neither display, nor devices, nor a GPU.
- *One code path.* The simulation stays authoritative in Python, so the live
  stand, training and headless experiments run the same code (CLAUDE.md).
- *State, not pictures.* A few numbers per step instead of encoded 4K frames:
  no encoding cost, negligible latency on localhost.
- *Text and controls.* HTML/CSS renders sharp text, sliders and layout at 4K
  with far less code than hand-drawn widgets.

**Constraints that follow.**

- *Thin frontend.* No framework, no npm, no build step; plain HTML, CSS and
  JavaScript served by the container. The frontend holds no logic.
- *Offline at runtime.* No CDN; every file comes from the container.
- *localhost.* Current browsers restrict the Gamepad API to secure contexts
  (not checked for every browser, §5); `http://localhost` is one, so the
  kiosk points there, not at the machine's address.
- *Names and ports.* The container is `teach-by-showing` and serves on port
  8081; feature-viz keeps 8080.
- *The core knows nothing about HTTP* (feature-viz §12). If web-framework
  types leak into the simulation, replacing the framework means rewriting the
  project.
- *Backend:* FastAPI/Starlette with uvicorn — the switch feature-viz §12
  names for the point where more than one endpoint is needed. Training and
  evolution run in a worker process so that the simulation loop never stalls.
  Not yet added as dependencies; that happens with the first code that needs
  them.

**Rejected alternatives.**

- *pygame-ce in the container.* Display and input passthrough, see above.
- *pygame-ce natively on the host.* Contradicts the container requirement.
- *Rendered frames as MJPEG plus an input back channel* (the feature-viz
  pattern taken literally). Extra latency and costly 4K encoding, for no gain
  when the state is a handful of numbers.
- *A frontend framework with a build chain.* Harder to maintain and to hand
  over; a dependency risk over years.
- *Unity, Godot or another engine.* The predecessor demonstrator's Unity/Python
  split is what made it unmaintainable (concept §2).

---

## 4. Tooling and organisation: as in feature-viz

**Decision.** The same organisation and tools as feature-viz, so that one
person can maintain both: PDM with a lockfile, `src/` layout, Python 3.12 for
development (`requires-python = ">=3.12,<3.14"`), the same `pyproject.toml`
settings for ruff, pyright, pytest, coverage and bump-my-version, and the same
dependency groups. `DECISIONS.md` and `CLAUDE.md` play the same roles.

**Deliberate differences.**

- *Licence* (§1).
- *Funding module.* feature-viz keeps its funding constants in its single
  module; here they live in `funding.py`, because this project will have
  several modules and the frontend also needs them.
- *No pytest markers yet.* feature-viz's markers (`weights`, `gpu`, `camera`)
  have no counterpart; tiers will be added when there is something to tier.

**Planned, following feature-viz once there is something to run:** a
`Dockerfile` on a digest-pinned `python:3.12-slim` with hash-checked installs
from `pdm export` (feature-viz §9), `compose.yaml`, a `start-container.sh`
that checks preconditions before starting, and an image archive via
`docker save`.

---

## 5. Unverified

Reasoned about, never measured. Each item is an open task, not a fact.

- **Behavioural cloning from a short demonstration.** Whether three transports
  of 20–30 s each suffice for an agent that reaches the target reliably and
  visibly keeps its teacher's style. This is the project's central risk
  (concept §5.2); the first spike tests it with real people.
- **Gamepad mapping per browser on Linux.** Chrome is documented to report
  XInput gamepads with the `standard` mapping, Firefox on Linux not (triggers
  and D-pad as axes, stick axes at other indices). The sources are several
  years old. To be checked with the GameSir G7 SE and the Logitech F310 in
  both browsers.
- **Kiosk placement under COSMIC/Wayland.** Whether a browser can be put
  full-screen on a given monitor reliably and from a script.
- **Gamepad input with two browser windows.** Whether the crane window keeps
  receiving gamepad input while the feature-viz window runs next to it.
- **Secure context for the Gamepad API.** Assumed for current browsers,
  not checked in Chrome and Firefox on the deployment machine. Harmless
  either way, since the kiosk uses `http://localhost`.
- **Training time.** "One to three seconds" for the MLP in NumPy is an
  estimate.
- **Evolution speed.** Whether a vectorised population shows a visible
  improvement within 20–60 s on the deployment machine.
