# Design Decisions

**Project:** teach-by-showing — crane demonstrator for imitation learning (TUC-KMI)
**Status:** spike day 1 in progress: simulation core and task (§6, §7)
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
  full-screen on a given monitor reliably and from a script. Connector names
  cannot identify the monitors: in the office they hang on DP-4 and HDMI-A-2
  and on a docking station in parallel, at the venue both on DisplayPort.
  Open whether placement by monitor identity (EDID: make, model, serial)
  works, or whether the assignment is set at setup time.
- **Gamepad input with two browser windows.** Whether the crane window keeps
  receiving gamepad input while the feature-viz window runs next to it.
- **Secure context for the Gamepad API.** Assumed for current browsers,
  not checked in Chrome and Firefox on the deployment machine. Harmless
  either way, since the kiosk uses `http://localhost`.
- **Training time.** "One to three seconds" for the MLP in NumPy is an
  estimate.
- **Evolution speed.** Whether a vectorised population shows a visible
  improvement within 20–60 s on the deployment machine.
- **Crane and task parameters** (§6, §7). A first set chosen on paper and
  checked only with a scripted controller. Whether the swing is dramatic
  enough for the naive agent, whether visitors can reach the calm criterion,
  and whether transports take the intended 20–30 s, is tuned on the gamepad
  (concept §15 items 7 and 8). The drive limits were raised once already
  for a visible naive swing (§6); fine-tuning on the gamepad stays open.

---

## 6. Simulation core: batched NumPy, semi-implicit Euler, a drive with limits

**Context.** The same crane is driven by a visitor at 60 Hz, by a learned
agent, by the naive baseline, and — during evolution — by a swarm of 50–100
agents at once (concept §5.3). CLAUDE.md demands one code path for all of
them.

**Decision.** `crane.py` simulates *n* cranes at once: every state quantity is
a NumPy array with one entry per crane, and a single crane is the case
*n* = 1. `step()` is a pure function from a state and one command per crane to
the next state; nothing is mutated, so recorded states stay valid.

**Physics** (concept §4). The load is a point mass on a massless rope of fixed
length *l*, hung from the trolley:

```
θ̈ = −(g·sin θ + ẍ·cos θ) / l − c·θ̇
```

- *Integration:* semi-implicit Euler at 240 Hz, four substeps per 60 Hz frame,
  fixed step. Time is counted in integer frames, so it cannot drift.
- *Command:* the target trolley velocity as a fraction of `max_speed`, clipped
  to −1..1, as on a real crane with a frequency converter. The drive follows
  it with at most `max_accel`.
- *End stops:* the drive limits the speed towards each end so that braking
  at `max_accel` always stops in time, like a crane's pre-limit switch. The
  continuous limit `sqrt(2·max_accel·distance)` is not enough: within a
  substep the trolley moves after its speed is set, so it reached the end
  slightly too fast and the hard clip stopped it with a jolt of up to
  3.4 × `max_accel` (found by `test_drive_respects_speed_and_acceleration_limits`
  once the drive became faster). The limit is therefore taken from the
  discrete braking distance, `v²/(2a) + v·h/2 ≤ d`, with the distance
  shortened by `a·h²` (0.03 mm) for the case that v is not a whole multiple
  of `a·h` and for rounding. Checked with 5 × 1000 cranes under random and
  sustained full commands for 120 s each: the acceleration never exceeded
  `max_accel` and the hard clip never acted. It stays as a last guard.
- The drive limits and end stops are the **safety layer** under every driver
  (concept §5.1). It is part of the crane, not of the agent, so a network
  cannot bypass it.

**Parameters** (`CraneParams`). `rope_length` 1.55 m gives a period of
2.5 s (measured in the simulation: 2.498 s). `damping` 0.05/s halves an
uncontrolled swing in about 28 s. Track 0–10 m, `max_speed` 1.2 m/s,
`max_accel` 1.5 m/s². Fine-tuning on the gamepad stays open (§5).

*Drive limits chosen for a visible difference* (decided 2026-10-02). The
demonstrator lives on the contrast between a naive driver and a trained
agent, so the naive swing must be obvious. The first set (0.8 m/s, 0.8 m/s²)
looked too tame in a playback of the simulation. Three sets were compared
over 500 random transports (seed 0) with a naive driver — full speed, then
the latest possible full braking, hitting the target exactly — and a scripted
anti-sway controller (deployment machine, 2026-10-02):

| Set | max_accel | max_speed | rope (period) | naive: max swing / residual at target, mean / median | ends calm by luck | controller: success, median time |
|---|---|---|---|---|---|---|
| A (first) | 0.8 m/s² | 0.8 m/s | 1.55 m (2.50 s) | 12° / 7° (20 cm) | 12 % | 100 %, 10.7 s |
| **B (chosen)** | 1.5 m/s² | 1.2 m/s | 1.55 m (2.50 s) | 22° / 15° (40 cm) | 4 % | 100 %, 7.3 s |
| C | 2.0 m/s² | 1.2 m/s | 2.0 m (2.84 s) | 21° / 13° (45 cm) | 3 % | 100 %, 8.1 s |

The residual swing depends mainly on top speed and period, much less on the
acceleration: a short acceleration phase gives the load an impulse
proportional to the velocity change. B doubles the naive swing and keeps the
concept's period of 2.5 s; C swings comparably with a slower period and was
not preferred. The price of B: it is faster and harder than a real crane
(about 0.3–0.5 m/s²), and a well-driven transport is short, around 7–11 s
for the controller against the concept's 20–30 s per demonstration;
visitors will drive more slowly. How dramatic the swing *looks* also depends
on the rendering scale, which is decided with the renderer.

**Why.**

- *Batched from the start*, because the swarm would otherwise need a second
  implementation later, and the rule against that would be broken exactly
  where it is hardest to test. The cost is that the core reads a little less
  directly than with plain floats. `test_batch_equals_individual_cranes`
  guards that a batch behaves bit for bit like separate runs.
- *Semi-implicit Euler* keeps the energy of a free swing bounded (within 1 %
  over 120 s, tested). Explicit Euler with the same step pumps energy in:
  measured, a free swing from 0.5 rad gains a factor of 12 in energy within
  120 s, so the load would swing up on its own.
- *Velocity, not acceleration or force, as the command*, because that is how
  a crane is driven and what a gamepad stick maps to naturally; it also gives
  the network a bounded, well-scaled output.

**Measured** (deployment machine, RTX 4090 / AMD 16-core, CPU only,
2026-10-02). One `step()`: 52 µs for 1 crane, 59 µs for 100, 127 µs for 1000.
Together with `task.advance()`, a 60 s episode costs 0.22 s simulated for one
crane and 0.24 s for 100 — so a frame of the live loop uses 0.3 % of its
budget, and the swarm costs almost nothing extra, as the concept assumed.

**Rejected alternatives.**

- *Scalar core plus a vectorised copy for the swarm.* Two implementations
  that must agree; see above.
- *RK4 or `scipy.integrate`.* More accurate per step, but the frame rate is
  fixed anyway, the error at 240 Hz is far below what is visible (period
  error 0.015 %, measured), and SciPy would be a dependency for nothing.
- *A physics engine (pymunk, Box2D).* A dependency and an opaque model for a
  single pendulum equation.
- *Hard end stops without early braking.* A trolley hitting the end at full
  speed would be a non-physical jolt and a cheap way to make the load swing.

`tests/test_crane.py` guards this section.

---

## 7. Task: start, target, and "calm" by swing amplitude

**Decision.** `task.py` draws start and target per transport from an injected
generator, uniform over the track minus a margin of 0.5 m at each end and at
least 4 m apart, in both directions. A transport succeeds when the load has
hung calm for one second; it fails after 60 s.

**Calm** means, all at once: the trolley is within 0.15 m of the target,
slower than 0.05 m/s, and the **swing amplitude** is below 2°. The amplitude
is `sqrt(θ² + (θ̇/ω₀)²)`, the amplitude the load would keep swinging with if
the trolley stood still from now on.

**Why the amplitude and not |θ|.** A swinging load passes the vertical twice
per period with θ = 0. A test on θ alone would count those moments as calm;
the one-second hold would mostly catch it, but not reliably for small
swings. The amplitude is constant over a free swing (tested) and is also the
natural measure for the style profile's "calm" and "residual swing" (concept
§6.2).

**Details that are deliberate.**

- *A finished transport is frozen.* Once it has succeeded or timed out,
  later frames do not change it, so a batch can keep stepping until the last
  crane is done.
- *Duration includes the calm second.* It is measured from the start to the
  frame at which the hold completes.
- *Sampling fails loudly* if the track is too short for the minimum distance,
  instead of silently producing short transports.

`tests/test_task.py` guards this section.
