# Design Decisions

**Project:** teach-by-showing — crane demonstrator for imitation learning (TUC-KMI)
**Status:** spike day 1 done: core, server, frontend, container, kiosk placement and gamepad (§6–§12); day 2 under way: recording (§13), tuning (§14), behavioural cloning (§15); evaluation and ghost replay next
**Last updated:** 2026-10-06

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
the dependencies — NumPy, FastAPI/Starlette, uvicorn and websockets, with
everything they pull in — are all permissive (MIT, BSD-3-Clause, PSF-2.0;
listed in §3). Any new dependency must keep it that way; a
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
and the logo list. The notice appears in `README.md` and as a white strip
along the bottom of the screen, filled in by the server (§9). It is in
English; a German option is decided but not built (§5).

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
  Added with the server (§8): `fastapi`, `uvicorn`, and `websockets`, which
  uvicorn needs to speak WebSocket at all; the alternative `wsproto` is less
  common. `uvicorn[standard]` was not taken: it adds uvloop, httptools,
  watchfiles, python-dotenv and PyYAML, none of which this server needs.
  The test group gains `httpx`, which Starlette's test client requires.
  Every package in the runtime tree is MIT, BSD-3-Clause or PSF-2.0 (§1).

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
- *`tools/` for host-side scripts.* Scripts that run on the stand machine
  outside the container, such as `kiosk.py` (§11), live in `tools/`, outside
  the package, and use the system Python, so they need neither the venv nor
  a build. mypy and ruff cover them like `src` and `tests`.
- *An entry point:* `pdm run teach-by-showing` starts the server from the
  venv (`api.main`); the container runs the same module.

**Container:** `Dockerfile` and `compose.yaml` as in feature-viz §9, see
§10. Still planned: a start script for the stand that checks preconditions
and opens the kiosk (concept §15 item 16), and an image archive via
`docker save` once there is a version worth archiving.

---

## 5. Unverified

Reasoned about, never measured. Each item is an open task, not a fact.

- **Calm threshold and input curve** — *chosen on 2026-10-09, see §14*:
  4° and a curve of 0.7, judged by one practised driver. Whether 3° would
  suit, and whether the curve is right for people who have not driven
  before, is open; to be decided with such tests (concept §15 items 6, 7).
  Recorded runs keep every state, so they can be judged again against
  another threshold.
- **Behavioural cloning from a short demonstration.** Whether three transports
  of 20–30 s each suffice for an agent that reaches the target reliably and
  visibly keeps its teacher's style. This is the project's central risk
  (concept §5.2); the first spike tests it with real people.
- **Kiosk placement under COSMIC/Wayland** — *settled on 2026-10-02, see
  §11*: Firefox with `--kiosk-monitor`, the monitor chosen by its EDID
  serial. Still open: the venue's cabling (both DisplayPort, no docking
  station), to be checked at the dress rehearsal; and whether a COSMIC or
  Firefox update changes the behaviour before November, so the placement is
  rechecked after updates on the stand machine. Firefox's placement rests on
  6 of 6 starts checked by eye, against series of 10–20 for Chrome; a
  longer series follows at the dress rehearsal at the latest.
- **Gamepad in Firefox** — *checked on 2026-10-06, see §12*: both pads
  report the left stick as axis 0 of the `standard` mapping. Still open:
  Chrome for comparison (not needed while the crane runs in Firefox), and
  the A button for the stand's steps once the state machine uses it.
- **Firefox's scale on a fractionally scaled monitor** — *judged on
  2026-10-02*. At COSMIC's 150 % Firefox reported `devicePixelRatio` 1.36
  and a 2816-pixel-wide screen, not 1.5 and 2560. By eye the crane in the
  Firefox kiosk looked sharp on the 4K monitor, so nothing is changed;
  Firefox's pref for fractional scaling on Wayland stays in reserve.
- **Gamepad input with two browser windows** — *checked on 2026-10-06, see
  §12*: the crane's Firefox receives the stick while a second Firefox
  instance has the focus. Checked with a stand-in page, not with feature-viz
  itself; to be repeated with both containers at the dress rehearsal.
- **Secure context for the Gamepad API** — *settled for Firefox on
  2026-10-06*: the gamepad works on `http://localhost` (§12). Not checked
  in Chrome.
- **Training time** — *measured on 2026-10-09, see §15*: 0.05–0.1 s for
  the linear policy, 0.3–2.4 s for the network (2000 epochs) on one or two
  recorded drives. Within the concept's one to three seconds, but the
  network's figure varies more than its data size explains; to be watched
  when several drives are pooled.
- **Loop timing under load** — *measured on 2026-10-02, see §8*: unchanged
  with feature-viz running next to it. Driving in the kiosk next to a
  running feature-viz (shown in Chrome) felt smooth and looked sharp, judged
  by eye on 2026-10-02.
- **feature-viz in Firefox** — *reproduced on 2026-10-02*. Both containers,
  crane in a Firefox kiosk on the left, feature-viz on the right, 3 minutes
  each, logged every 5 s. In Firefox, feature-viz showed its picture at
  first, then the image area turned black and stayed black, while its
  server stayed healthy and the stream connection stayed open (frames of
  3790 × 1372 JPEG, about 870 KB, about 24 MB/s). The Firefox content
  process ran at 150 % CPU, its memory rising from about 0.25 to 2.3 GB
  within every 20 s and dropping back. The same page in Chrome on the same
  machine: picture throughout, 33 % CPU, memory steady at 160 MB. The crane
  next to it, in Firefox, drove smoothly and looked sharp (judged by eye).
  So Firefox cannot show feature-viz's MJPEG stream, and Chrome cannot be
  placed on a monitor (§11). Either feature-viz is changed so that Firefox
  shows it, or the feature-viz kiosk runs in Chrome with a placement that
  is checked after start and retried (Chrome's debugging port tells which
  screen it is on, `screen.availLeft`).
  Decided 2026-10-02: the first way, as a task in feature-viz; the Chrome
  fallback stays open and is built only if that fails.
- **Smoothness against latency in the kiosk.** Interpolation (§9) costs
  about 21 ms of display delay. In the development browser pane it removed
  all visible skips, but that pane drew only 48 frames per second. In the
  Firefox kiosk on the 4K monitor, driving with the keyboard felt smooth
  with no noticeable delay (judged by eye, 2026-10-02), and so did the
  gamepad (2026-10-06, §12). Still open: whether `?delay=0` looks worse.
- **Funding notice language.** The notice is shown in English, verbatim as
  in feature-viz, on an otherwise German screen. Decided 2026-10-02, not yet
  built: the language becomes an option, German or English, with English as
  the default so that the repository reads internationally. The German
  wording must come from the funder's material, not from a translation of
  our own.
- **Several browsers driving at once.** Every connected client may send
  input, and the latest message wins. The stand has one kiosk, so this only
  matters when a second tab is open for debugging: the two then fight over
  the crane. Whether the server should accept input from one client only is
  decided with the stand's state machine.
- **Green on success against green for the agent.** The frontend colours
  the load and the target zone green when a transport succeeds (§9);
  concept §8 reserves green for the learned agent. Once agents drive, that
  reads as "the agent". Resolved with the agent display.
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
2026-10-02, after the end-stop fix). One `step()`: 70 µs for 1 crane, 78 µs
for 100, 160 µs for 1000. Together with `task.advance()`, a 60 s episode
costs 0.28 s simulated for one crane and 0.31 s for 100 — so a frame of the
live loop uses under 0.5 % of its budget, and the swarm costs almost nothing
extra, as the concept assumed. (Before the end-stop fix: 52, 59 and 127 µs.)

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

`tests/test_crane.py` guards this section; the choice of set B in
particular is guarded by
`test_naive_driver_leaves_a_visible_swing_a_controller_does_not`.

---

## 7. Task: start, target, and "calm" by swing amplitude

**Decision.** `task.py` draws start and target per transport from an injected
generator, uniform over the track minus a margin of 0.5 m at each end and at
least 4 m apart, in both directions. A transport succeeds when the load has
hung calm for one second; it fails after 60 s.

**Calm** means, all at once: the trolley is within 0.15 m of the target,
slower than 0.05 m/s, and the **swing amplitude** is below 4° (2° until
2026-10-09, too hard on the gamepad, §14). The amplitude
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
- *A target from anywhere.* `sample_targets()` draws a target for a given
  start, which may lie anywhere on the track, including the ends where a
  timed-out transport left the trolley; the live loop (§8) needs that.
  `sample_transports()` draws starts and uses it.

`tests/test_task.py` guards this section.

---

## 8. Live loop and protocol: one paced loop, latest-only outboxes

**Context.** The visitor drives the crane in a browser; the simulation is
authoritative in Python (§3). Something has to pace the simulation at 60 Hz,
carry the state out and the stick back in, and stay up for two days.

**Decision.** Three layers, as in feature-viz §12:

```
crane.py, task.py   simulation core           — knows nothing about HTTP
live.py             LiveSession, run_paced()  — the engine, counts in frames
api.py              FastAPI: /ws, /healthz, static files — calls into live.py only
```

- *Pacing:* `run_paced()` is an asyncio task that ticks the session against
  absolute deadlines (`deadline += 1/60`), so the inaccuracy of each sleep
  does not add up to drift. If it falls more than 0.1 s behind it drops the
  lost time and counts an overrun, rather than catching up with a burst the
  viewer would see as a jump.
- *An asyncio task, not a thread* — unlike feature-viz §12, whose loop is
  blocking inference. One tick here costs about 0.08 ms (§6), so it never
  holds the event loop noticeably, and with a single owner of the state no
  lock is needed. Training and evolution will take seconds and go to a
  worker process (§3), never into this loop.
- *Outboxes:* each connection has a slot holding only the latest state
  message. The loop only overwrites slots; a sender task per connection
  sends what is there. A slow or stalled browser skips frames and can never
  hold up the simulation.
- *Messages* are JSON. On connect, one `config` message carries the crane
  and task parameters, so the frontend draws to scale without physical
  constants of its own. Then one `state` message per frame: positions, angle,
  target, effective command, calm progress, result, elapsed time. Positions
  are lists, as the core is batched; ghosts and swarms will use the same
  shape. The browser sends `{"type": "input", "u": <-1..1>}`.
- *Untrusted input:* anything that is not a finite number is dropped and
  logged without closing the connection; numbers are clipped to −1..1.
- *Input timeout:* without fresh input for 0.25 s the command falls back to
  zero and the drive brakes. A closed tab, a lost connection or an
  unplugged gamepad must never leave the crane driving.
- *Transports follow each other:* after success or timeout, a pause of
  1.5 s, then a new target is drawn from wherever the trolley stands. The
  stand's real sequence of steps 0–4 replaces this with the state machine.
  Until 2026-10-09 a timeout skipped the pause and was not counted; the
  first recorded run (§13) showed it, and a test now covers the timeout.
- *Health:* `/healthz` returns 503 when the loop has not ticked for a
  second, not merely when the server is down. A crash in the loop is logged
  and turns the health check red. Run as a program (`main()`), the server
  then also shuts itself down, because Docker restarts an exited container
  but not an unhealthy one (§10; tested with `os.kill` replaced).
- *Bind address:* `127.0.0.1:8081` by default; the Gamepad API needs a
  secure context, which `localhost` is (§3). `--seed` makes a session
  reproducible; without it a fresh seed is drawn and logged.

**Measured** (deployment machine, RTX 4090 / AMD 16-core, otherwise idle,
outside the container, 2026-10-02; script: a WebSocket client on the same
machine, 30 s, written for the occasion and not kept; `tools/measure_ws.py`
replaces it, so the figures can be taken again the same way):

| Quantity | Figure |
|---|---|
| state messages | 1801 in 30.0 s, 60.00 Hz, none skipped |
| interval between messages | median 16.57 ms, 1st–99th percentile 15.51–17.69 ms, max 18.82 ms |
| simulated vs. wall-clock time | −0.3 ms over 30 s |
| input round trip (send → first state showing it) | median 16.5 ms, p95 17.6 ms, max 17.6 ms (see below) |
| overruns | 0 |

The round trip is at most one frame, as designed: an input takes effect at
the next tick. The median of 16.5 ms above was the worst case, not the
spread: the script of 2026-10-02 sent in step with the server's tick and so
always just missed it. Remeasured on 2026-10-06 (deployment machine, idle,
outside the container) with `tools/measure_ws.py roundtrip`, which waits a
random part of a frame before each send: median 6.6 ms, p95 15.4 ms, max
17.6 ms over 53 commands. A first run without that random wait gave a
median of 2.4 ms, the opposite extreme of the same artefact. Browser rendering and the gamepad's own polling come on top and are
part of the spike's acceptance ("latency not noticeable").

**Under load** (deployment machine, 2026-10-02, 60 s): crane and
feature-viz both in their containers, feature-viz with GPU and webcam, both
pages open full screen in two Firefox kiosks (§11). The crane: 3601 state
messages, 60.00 Hz, none skipped, interval 1st–99th percentile
15.70–17.56 ms, max 17.90 ms; 0 overruns in 112 682 ticks since the
container started (31 min). The machine: GPU 19–30 %, CPU 82–88 % idle of 32
threads; the busiest process was feature-viz's Firefox decoding its MJPEG
stream (about 150 % of one core), the crane's server took 4–21 %. The
input round trip could not be measured this way: the open kiosk sends its
own command every frame and overwrote the measuring client's (§5, several
browsers driving at once).

**Rejected alternatives.**

- *A thread for the loop* (feature-viz §12). Needed there for blocking
  inference; here it would add a lock around the session for nothing.
- *Awaiting each client's send inside the loop.* One slow browser would slow
  the crane for everyone.
- *A queue per client.* Old states are worthless; queueing them only adds
  latency.
- *Simulation in the browser, server only for learning.* Two implementations
  of the physics, or JavaScript as the authoritative one; both contradict
  "one code path" and "logic belongs in Python".
- *Server-Sent Events plus POST for input.* Two channels where one
  WebSocket suffices, and an HTTP request per input at 60 Hz.

`tests/test_live.py` and `tests/test_api.py` guard this section, including
that `crane.py`, `task.py` and `live.py` import nothing from the web stack.

---

## 9. Frontend: plain files, drawn to scale, interpolated

**Decision.** Five static files, no framework, no build step (§3):

```
index.html   layout, German texts, placeholder for the funding notice
style.css    sizes in viewport units for a 4K monitor read from standing distance
scene.js     drawScene(): girder, trolley, rope, load, target zone — drawing only
input.js     readCommand(): the gamepad's left stick, or the keyboard, ramped (§12)
main.js      WebSocket, state buffer, interpolation, readouts, render loop
```

Beside them, `gamepad.html`: a page for staff that shows the raw Gamepad
API values, self-contained and not linked from the crane page (§12).

About 280 lines of JavaScript, plus about 50 in the staff page
`gamepad.html` (§12), against a budget of about 900 for the whole
project (concept §7.4). A fault can be traced by file: wrong picture →
`scene.js`, crane does not react → `input.js`, nothing at all → `main.js`
and the browser console.

**Details that are deliberate.**

- *Drawn to scale from the config message.* Track, rope length and target
  tolerance come from the server; the scripts hold no physical parameter of
  their own (tested for the distinctive ones). One scale for both axes, so
  that the drawn angle is the simulated angle. Drawn part sizes (trolley,
  load) are in metres too.
- *Interpolation with a fixed delay.* The picture runs about 21 ms — one
  state interval plus margin for the ±1 ms jitter (§8) — behind the newest
  state and interpolates position and angle between the two received states
  around it. Sim time is mapped to browser time through the least-delayed
  arrival in a 2 s buffer. Without this, a display refreshing at 60 Hz
  beside a server ticking at 60 Hz shows some states twice and skips others,
  a visible stutter of a few pixels on a trolley moving at 1.2 m/s.
  Measured in the development browser pane (deployment machine,
  2026-10-02, 10 s): without interpolation 120 of 480 drawn frames skipped
  a state; with it, every frame advanced by 0.94–1.0 states except where the
  pane itself dropped a frame. `?delay=0` switches it off, to compare on the
  stand (§5).
- *The command is sent every displayed frame.* That doubles as a heartbeat:
  a hidden tab stops its render loop, the server's input timeout (§8)
  fires, and the crane brakes.
- *Keyboard ramped* from 0 to full in 0.4 s. A key is all or nothing; without
  the ramp every key press would be a full-throttle jolt that no gamepad
  produces, and keyboard-driven test runs would teach the network the wrong
  style. The keyboard is for development and staff, not for visitors
  (concept §6.3).
- *Reconnect* every second after the connection drops, with a notice on
  screen; the server may be restarted by its healthcheck.
- *Funding notice filled in by the server* (`index_page()`), html-escaped
  from `funding.py`, so the wording exists once and the script never touches
  it. Logos come from `/funding/<name>`, through the allow-list of §2;
  anything else is 404 (tested, including a traversal attempt). White strip
  along the bottom, as in feature-viz §17.
- *Offline:* no external script, stylesheet or font (tested); system fonts
  only.

**Rejected alternatives.**

- *Drawing the newest state as it arrives* (no interpolation): lowest
  latency, but the stutter described above. Kept as `?delay=0`.
- *Extrapolating past the newest state:* no added delay, but it predicts
  the swing, and every correction would show as a jolt.
- *SVG or DOM elements instead of a canvas:* comparable for one crane, but
  the swarm of the evolution step will draw a hundred.
- *ES modules:* would need a module-aware layout for four small files;
  plain script tags in a fixed order are easier to follow.
- *Keyboard without a ramp:* see above.

`tests/test_frontend.py` guards this section.

---

## 10. Container: as in feature-viz, without GPU or devices

**Decision.** `Dockerfile`, `.dockerignore` and `compose.yaml` in the
repository root, following feature-viz §9; `docker compose up -d` starts the
stand's backend, and the host's browser opens `http://localhost:8081`.

**Taken over from feature-viz §9.**

- *The image runs what the tests ran:* `python:3.12-slim` pinned by digest,
  on top exactly what `pdm.lock` pins, installed with `pip --require-hashes`
  from a `pdm export`. pdm, pinned, exists only in the build stage. The
  project is copied as source and found via `PYTHONPATH`.
- *The same base digest as feature-viz,* so that the two images on the stand
  machine share their base layers. The first build here took 14 s because
  that layer was already present.
- *Unprivileged user* (UID 10001), *build context as an allow-list*,
  *health check on `/healthz`* (§8), *image tag follows the package version*
  (`bump-my-version` updates `compose.yaml`).

**Different from feature-viz.**

- *No GPU, no devices, no display, no `ipc: host`.* NumPy on a few cranes
  needs none of them (§6), and the browser on the host reads the gamepad
  (§3). Nothing in `compose.yaml` depends on the machine, so there is no
  `.env` and no variant files.
- *Port published to the host's localhost only* (`127.0.0.1:8081`). The kiosk
  runs on the same machine, the Gamepad API wants `http://localhost`, and
  nobody else on the venue's network should drive the crane. Inside, the
  server binds `0.0.0.0`, since Docker forwards to the container's own
  interface.
- *`restart: unless-stopped`, and the server ends itself when its loop
  crashes* (§8). Docker only reports an unhealthy container; it restarts an
  exited one. A container stopped by hand stays stopped.
- *A named volume `data` at `/data`* (`DATA_DIR`) for runs, agents and the
  gallery, so they survive rebuilds. Recorded runs go to `/data/runs`
  (§13).

**Verified** (deployment machine, Docker 29.8.2, 2026-10-02):

| Check | Result |
|---|---|
| Build, base layer cached | 14 s; image 73.7 MB content, 314 MB on disk |
| Start to healthy | about 7 s |
| Container setup | no devices, not privileged, user `teach-by-showing`, port bound to `127.0.0.1:8081` only |
| Timing at a client on the host, 30 s (script of §8) | 60.00 Hz, none skipped; interval median 16.64 ms, 1st–99th percentile 15.57–17.79 ms; input round trip median 16.6 ms; no overrun — the same as outside the container |
| Self-shutdown (SIGTERM to PID 1, the path a loop crash takes) | restarted by Docker, healthy again within seconds; the open browser reconnected by itself |
| `docker stop` | 0.4 s, exit code 0 |
| `--network none` | healthy after about 7 s, page and logos served |

**Not yet done.** The `org.opencontainers.image.source` label names the
intended GitHub location; the remote does not exist yet (concept §15 item
15). No archive with `docker save` yet; that follows once a version is worth
keeping, as in feature-viz.

**Rejected alternatives.**

- *`uvicorn --reload` or a development server in the image:* the image is
  for the stand; development runs from the venv with
  `pdm run teach-by-showing`.
- *Host networking (`network_mode: host`):* would expose whatever the server
  binds to the venue's network and loses the port mapping that keeps it on
  localhost.
- *An autoheal sidecar that restarts unhealthy containers:* a second
  container and a mounted Docker socket, where a process that exits on its
  own fault does the same with nothing extra.

---

## 11. Kiosk placement: Firefox with `--kiosk-monitor`, chosen by EDID serial

**Context.** Both demonstrators run full screen on the same machine, one
monitor each, under COSMIC on Wayland. A start script must put each browser
on the right monitor without anyone touching the mouse. Connector names do
not identify a monitor: in the office the two hang on DP-4 and HDMI-A-2 (and
a docking station in parallel), at the venue both on DisplayPort. The two
office monitors are the same model (AOC U2790B) and differ only in their
EDID serial (114210 on DP-4, 114203 on HDMI-A-2).

**Decision.** `tools/kiosk.py open --serial <EDID serial> [--url ...]`:

```
EDID serial  --cosmic-randr list--> connector and layout position
layout position  --GTK 3 (Gdk)-->   Firefox's monitor number
firefox --no-remote --profile <fresh> --kiosk --kiosk-monitor <number> <url>
```

The script runs on the host with the system Python (standard library, plus
the GTK bindings that come with the desktop), so it needs neither the venv
nor a build.

**What was tried, and why it failed** (stand machine, COSMIC with
cosmic-comp of 2026-09, Chrome 154, Firefox 156, 2026-10-02). Placement
was read back with `xwininfo` for X11 windows, through Chrome's debugging
port (`screen.availLeft`) for Wayland windows, and by eye.

- *Chrome under XWayland with `--window-position`.* The first start, with a
  fresh profile, landed where asked; the second, reusing that profile, did
  not; the third, with a fresh profile again, did — which looked like
  success once the profile was understood. Repeated runs did not: COSMIC
  ignores the position an X11 window asks for and opens it centred on the
  output under the mouse pointer (a 1200 × 800 window asked for at 4200,200
  and at 300,200 landed at the same spot both times). Separately, a reused
  Chrome profile restores its last position and overrides the flag, and
  `--window-size` together with the position sent the window to the first
  monitor.
- *Moving the pointer first,* with a short-lived virtual mouse through
  `/dev/uinput` (writable for the session's user through an ACL), pushed far
  beyond the layout's edge so that acceleration does not matter. A device
  with only REL_X was ignored; with both axes it moved the pointer, but
  placement stayed unreliable: 16 of 20 under XWayland (all failures on the
  left monitor after a run on the right), 3 of 10 when the pointer first
  moved away and back, and with Chrome on native Wayland 10 of 20, now
  failing only on the right. The compositor's choice of output depends on
  more than the pointer position, in ways not documented and liable to
  change with any COSMIC update.
- *Firefox with `--kiosk-monitor`.* On Wayland the client may name the
  output in its fullscreen request, and the compositor follows it. Verified
  by eye with the pointer deliberately on the *other* monitor, in both
  directions: left with the pointer on the right, right with the pointer on
  the left, each full screen and showing only the page. In all, 6 of 6
  starts landed correctly (two by hand, two through the script with the
  pointer opposite, two in the parallel runs with feature-viz); a longer
  series is open (§5).

**Details that are deliberate.**

- *The monitor number comes from GTK, not from cosmic-randr.* Firefox counts
  monitors in GTK 3's order, which put the right monitor first, while
  `cosmic-randr` listed DP-4 first in one call and HDMI-A-2 first in the
  next. The script matches the two by layout position.
- *Layout positions are scaled* (the right monitor starts at 2560, not
  3840, at 150 %). The script never mixes them with device pixels; the
  XWayland coordinates of the failed attempts were device pixels.
- *A fresh profile each start,* removed when Firefox exits, with prefs that
  suppress the welcome page, the data-reporting notice, the default-browser
  question, session restore after a crash and the translation offer.
  `--no-remote` keeps the crane's and feature-viz's Firefox apart.
- *`MOZ_ENABLE_WAYLAND=1`*: only a native Wayland client can name the output.

**Consequences.**

- *Firefox for the crane,* not Chrome as concept §7.1 proposed: Chrome
  cannot be placed. The gamepad mapping in Firefox and input with two
  Firefox windows are checked on the spike's day 1 (§5).
- *Placement is verified by eye,* not by a test: a Wayland page does not
  learn which monitor it is on (Firefox reports `screen.left` 0 on both).
  `tests/test_kiosk.py` covers the parsing, the matching and the command.
- *feature-viz must run in Firefox too,* and so far it does not: its MJPEG
  picture turns black there while Chrome shows it (§5). It is being fixed in
  feature-viz; Chrome with a placement checked after start and retried stays
  the fallback for the feature-viz kiosk.
- *Tied to COSMIC* through `cosmic-randr`. Another desktop needs another
  source for serial and position; the rest of the chain is generic.

**Rejected alternatives.**

- *Chrome with `--window-position`, with or without moving the pointer:*
  see above.
- *Assigning the windows by hand at setup:* a manual step at every start,
  the risk the concept names.
- *A separate kiosk session with sway,* whose config can assign windows to
  outputs by make, model and serial: deterministic, but a second desktop to
  install and maintain on the stand machine. The fallback, should Firefox's
  placement stop working.
- *The Window Management API* (`requestFullscreen({screen})`): needs a
  permission granted through a browser policy and a user gesture, and puts
  placement logic into JavaScript.

---

## 12. Gamepad: left stick in the "standard" mapping, judged at the server

**Context.** Visitors drive with a USB gamepad (concept §7.7): the GameSir
G7 SE as the main device, the Logitech F310 as the spare. The browser reads
it through the Gamepad API (§3), and since §11 that browser is Firefox,
where older sources reported XInput pads without the `standard` mapping.
The spike's day 1 had to show that the stick reaches the crane, also while
feature-viz's Firefox runs beside it, and survives unplugging.

**Decision.** `input.js` reads axis 0 of the Gamepad API's `standard`
mapping (left stick, horizontal) from whichever connected pad is pushed
furthest, with a dead zone of 0.05 rescaled so the command starts at 0.
The stick wins whenever it is deflected; at rest the keyboard drives, for
development and staff. Fixed indices, no calibration yet: both pads report
the same layout in Firefox, and the calibration dialog of concept §7.1
comes only when a device needs it.

**Measured** (deployment machine, RTX 4090 / AMD 16-core, Firefox 156.0.1,
kernel 7.1.5, 2026-10-06). The browser's view was read from
`/gamepad.html`, a staff page that shows the raw Gamepad API values; what
reached the simulation from `tools/measure_ws.py trace`, which listens on
the WebSocket and prints the command the server applies, so a fault on the
screen and a fault in the input can be told apart.

| Check | GameSir G7 SE | Logitech F310 |
|---|---|---|
| kernel driver | `hid-generic`, not `xpad` | `xpad` |
| Firefox's `mapping` | `standard`, 8 axes, 19 buttons | — (drove the crane through axis 0 unchanged) |
| left stick, horizontal | axis 0, −1..+1 | axis 0, −1..+1 |
| A, B | buttons 0, 1 | — |
| stick resolution | 8 bit (0..255) | 16 bit |
| rest value, X (evdev) | +0.004 | +0.004 |
| rest value, Y (evdev, unused) | +0.004 | −0.043 |

- *Firefox reports the GameSir with the `standard` mapping* although the
  kernel binds it to `hid-generic`, not to `xpad`, which the concept's
  choice relied on. Linux picks a driver by the interface a device
  announces: `xpad` claims GameSir pads only on a vendor-specific interface
  (class 0xFF, subclass 0x47, protocol 0xD0: the Xbox One protocol, GIP).
  The G7 SE announces only HID interfaces (class 3), so `hid-generic` is
  the one driver that fits, and `xpad` cannot be forced on it — it would
  not understand the reports. With `xpad`, the kernel would translate every
  Xbox pad into one fixed layout that browsers know; with `hid-generic` the
  layout is the order in the pad's own HID descriptor, and the browser has
  to guess it. In this mode Firefox guesses right. The pad has two more
  modes, one of them with `xpad`; see below.
- *Its triggers read as pressed at rest* in Firefox: buttons 6 and 7 at
  0.5 with `pressed` true, axes 6 and 7 at −1. "Press any button" would fire
  on its own; buttons must be read by index (A is 0). Guarded by
  `tests/test_frontend.py`. After the calibration below and with the udev
  rule installed, the triggers rested at 0 and nothing read as pressed;
  which of the two changed it is not known, so the rule stands.
- *Focus does not matter.* With the crane in the Firefox kiosk on the left
  monitor and a second Firefox instance with its own profile (standing in
  for feature-viz) focused on the right, the full stick range reached the
  server, and the trolley moved, confirmed by eye.
- *Unplugging* the GameSir with the stick held full right dropped the
  command from +1 to 0 within one 0.25 s bin: the pad leaves
  `getGamepads()` at once, without waiting for the server's input timeout
  (§8). Plugged back in, the stick worked again at once, without a button
  press or a reload.
- *Latency:* "not noticeable" by eye, driving with the GameSir in the kiosk.
  The server side is one tick at most (§8).
- *Both pads at once:* plugging the F310 in beside the GameSir needed no
  selection; it drove as soon as it was pushed.
- *Calibration:* the GameSir was measured before the stick calibration its
  manual describes. Afterwards the rest values of all four stick axes were
  unchanged at 128 (the 8-bit centre); the full deflection after
  calibration was not rechecked.

**The GameSir's keyboard and mouse interface.** In its default mode the
pad announces two HID interfaces: 00, the gamepad, and 01, a keyboard and a
mouse (plus volume keys as a consumer control on 00). Read at the kernel
while every button, stick, trigger and combination was tried for 90 s,
only one thing came out: the *Share* button sends the Print key
(`KEY_SYSRQ`), and COSMIC opened its screenshot dialog over the kiosk. The
interface declares a full keyboard (Ctrl, Alt, Super, function keys) and a
mouse with buttons and wheel besides, presumably for the pad's macro
functions. A visitor could reach the desktop through the gamepad — the
risk concept §14 names for mouse and keyboard.

`tools/udev/70-teach-by-showing-gamepad.rules` deauthorises interface 01
of 3537:1082, so the kernel creates no keyboard or mouse for it, and leaves
the gamepad on 00 untouched. It runs on the host, where the browser reads
the gamepad; the container has no devices (§10) and is not involved. It is
installed once by hand (the file says how) and later by the stand's start
script (concept §15 item 16). The F310, through `xpad`, has no such
interface.

Verified with the rule installed (deployment machine, 2026-10-06):
interface 01 deauthorised and without a driver, no keyboard or mouse node
left; Share opened nothing; Firefox still reported the pad with the
`standard` mapping, A as button 0, B as 1, the left stick as axis 0. The
consumer control on interface 00 (volume keys) remains; it sent nothing in
the test.

**The GameSir's three modes.** The pad announces itself in one of three
ways, depending on what was pressed. Established from the kernel log of
every plug-in on 2026-10-06 and then reproduced step by step:

| Mode | How it is reached | USB | Driver | Layout in Firefox |
|---|---|---|---|---|
| default | plugged in, nothing pressed; LED off | 3537:1082, gamepad + keyboard/mouse (HID) | `hid-generic` | correct (A 0, B 1, stick axis 0); 8-bit sticks |
| View+Menu | View and Menu held while plugging in; LED on | 3537:1010, gamepad only (HID) | `hid-generic` | mixed up, see rejected alternatives |
| Xbox | while plugged in, Menu + View + M held for 3 s (the manual calls it a reset); LED on, the pad vibrates | 3537:1010 with serial number, Xbox protocol (GIP) | `xpad` | correct and complete |

- *The Xbox mode does not last.* The pad re-announces itself within a
  second of the combination. It survives the next unplug and replug, and
  falls back to the default mode on the one after (three times in the log,
  once in a deliberate test). Share + Menu, View + Menu, Xbox + Menu, and
  Xbox or M alone did nothing while plugged in.
- *In the Xbox mode Firefox reports exactly the standard layout:* 4 axes,
  17 buttons; A, B, X, Y as 0–3, LB and RB as 4 and 5, the triggers as
  analogue buttons 6 and 7, left stick on axes 0 and 1 (up is −1), right
  stick on 2 and 3; nothing pressed at rest. Sticks have 16 bits, the
  triggers 10. No keyboard or mouse interface, so the udev rule, which
  names 1082, has nothing to do.

So `input.js` works in the default and in the Xbox mode alike. The stand
relies on the default mode, protected by the udev rule: it is what the pad
falls back to after any second replug, whoever does it. The Xbox mode is
checked and harmless, but nothing may depend on it.

**Details that are deliberate.**

- *The dead zone is 0.05.* The GameSir rests one 8-bit step off the centre
  (0.004); the F310's X axis rests there too. Without a dead zone that step
  would enter the recordings as creep (concept §7.7). The F310's unused Y
  axis rests at −0.043, close to the dead zone; a worn potentiometer on the
  X axis could exceed it, which the calibration dialog would then measure.
- *The furthest-pushed pad wins,* rather than the first connected one, so a
  spare plugged in next to the main pad needs no setting.
- *No ramp on the stick.* The keyboard's ramp (§9) imitates a stick; the
  stick itself is analogue.

**Not done.** Chrome as a comparison (concept §16 asked for it only as
such; Firefox works, and Chrome stays out of the crane kiosk, §11). The
real feature-viz page beside the crane, rather than a stand-in Firefox:
the stand-in is a separate Firefox instance with its own profile, which is
what matters for focus, but feature-viz's load was not on the machine.

**Rejected alternatives.**

- *Reading the gamepad in the container* through `/dev/input`: device
  pass-through, the fragility §3 avoids.
- *A calibration dialog now:* both pads report the same layout; built when
  a device needs it.
- *Reading the state of Firefox through WebDriver BiDi* for these checks:
  tried, but a remote-controlled Firefox kept reporting focus after a
  click into another window, so the focus check would have proven
  nothing. The server's view, through `tools/measure_ws.py`, does not
  depend on the browser.
- *The GameSir's View+Menu mode.* It announces 3537:1010 — the ID `xpad`
  lists for the G7 SE — but only a HID interface, so the kernel binds
  `hid-generic`. The keyboard and mouse interface is gone and the triggers
  rest at 0, but Firefox, apparently assuming the `xpad` layout for this
  ID, mixes everything except the left stick up: A, B, X, Y arrive as
  buttons 1, 17, 0, 3; the right stick as the trigger buttons 6 and 7,
  pressed at rest; the triggers as the right stick's axes; the left
  stick's vertical axis inverted. It is also lost on every unplug.
- *Relying on the GameSir's Xbox mode:* the best layout and 16-bit sticks,
  but it has to be set again after every second replug, and a restart of
  the machine presumably counts as one. A step the staff can forget must
  not be what keeps the stand safe. The crane does not need the
  resolution: one 8-bit step is about 1 cm/s of the trolley.
- *The out-of-tree driver xone* for Xbox One pads: installed on the stand
  machine on 2026-10-06 while looking for the Xbox mode, but not involved —
  the Xbox mode appeared five minutes before its installation, it is
  `xpad` that binds the pad, and xone's modules had been built empty. A
  kernel module outside the kernel would have to be rebuilt on every kernel
  update.
- *Taping over Share:* does not cover macro buttons and comes off within
  two days of visitors.
- *Switching off the Print shortcut in COSMIC:* hits every keyboard,
  including the staff's, and covers only that one key.
- *Hiding the interface from libinput* (`LIBINPUT_IGNORE_DEVICE`): works
  only as long as the desktop uses libinput, and the devices still exist.

`tests/test_frontend.py` (served probe, no "any button"),
`tests/test_measure.py` and `tests/test_udev.py` guard this section.

---

## 13. Recording: a run is one JSON file of columns, begun by the first deflection

**Context.** Behavioural cloning learns from what a teacher did (concept
§5.1). Recording makes that an offline problem: a run can be trained on,
replayed and turned into a test without hardware or a person at the stand.

**Decision.** `recording.py` holds a `Run`: the crane and task parameters,
the seed, a label (`vorsichtig`, `zügig`, never a name, concept §6.3), the
state before the first recorded frame, and per frame the command applied,
the state that step produced (x, v, θ, ω), the target, the transport's index
and whether the transport was still running. `LiveSession` appends one row
per tick while a recording is active. For the spike it is started from the
command line, before the stand's state machine exists:

```bash
teach-by-showing --record vorsichtig --transports 3   # saved under $DATA_DIR/runs
```

**Why.**

- *Commands and states, both.* The states follow from the commands, and a
  test sends the commands through the same `step()` and gets every state
  back bit for bit (`tests/test_recording.py`). Storing them anyway makes a
  file usable without re-simulating it, and a change to the physics shows
  as a failing test instead of silently changing old demonstrations.
- *The command applied, not the one received:* after the input timeout
  (§8) the crane brakes on 0 whatever the browser last sent, and the
  network must learn what moved the crane.
- *It begins with the first deflection.* The first transport's clock
  restarts there. The seconds a teacher needs to pick up the gamepad are not
  driving; recorded, they would teach "stand still far from the target"
  (the averaging failure of concept §5.2) and count as transport time.
  Hesitation *during* a transport is the teacher's style and is kept.
- *It ends with the frame that finished the last transport;* its pause is
  not recorded. Pauses between transports are recorded but marked, since
  the target is gone and the teacher is not driving; training skips them.
  A pause row carries the index and target of the transport it follows.
- *Columns, not a list of frame objects:* a file of 5000 frames stays
  compact and loads straight into arrays.
- *A schema number.* A file of another schema is refused, not misread.

**Details that are deliberate.**

- *The file name is `<time requested>-<label>.json`*, with everything but
  letters, digits and hyphens dropped, so that a label can never leave the
  folder. The time is the wall clock of the server's start, passed in from
  `api.py`; `live.py` counts frames, never wall-clock time.
- *Written to a `.partial` name and renamed,* so a crash never leaves half a
  file that looks like a run.
- *Saved inside the simulation loop.* Measured on the deployment machine
  (RTX 4090 / AMD 16-core, 2026-10-06): a run of 6535 frames, 0.8 MB, takes
  a median 10.8 ms to serialise and write, at most 17.1 ms in 20 tries,
  about one frame, once per recording, after the last transport. Not worth a
  thread.
- *`/data/` is ignored by Git.* Runs chosen as test data go to `tests/data/`
  deliberately (concept §5.1).

**Rejected alternatives.**

- *Recording in the browser or through a WebSocket listener* such as
  `tools/measure_ws.py`: the state messages carry neither v nor ω, and the
  server alone knows the command it applied.
- *One row per frame as an object:* about three times the size, for no
  reader that needs it.
- *Recording from frame 0:* see the first deflection above.
- *A staff key in the frontend to start a recording:* the trigger belongs to
  the state machine; until then a command-line flag needs no frontend code.

`tests/test_recording.py` guards this section.


---

## 14. Tuning on the gamepad: calm below 4°, an input curve, settings files

**Context.** The first recorded run (deployment machine, 2026-10-09, the
maintainer, 5 transports with set B of §6 and the task of §7) showed the
crane too hard to bring to rest: two of five transports timed out, the
three others took 22, 38 and 56 s. The approach was quick (6–14 s to the
target zone); calming the load there took 12–51 s. After arrival the
trolley stood within the position tolerance most of the time, but the swing
amplitude was below 2° in only 0–9 % of the frames. The maintainer: "the
hard part is stopping the swing at the target; it takes a lot of finesse."
Judged afterwards against looser thresholds on the same frames, a 4°
threshold would have ended the successful transports after 15, 24 and 39 s
and saved one of the timeouts; a wider position tolerance changed nothing.

**Decision** (2026-10-09). Candidate C below is the default: calm below
4° of swing, and the input curve with 0.7. The damping stays at 0.05/s.

**How it was chosen.** `--params FILE` takes a JSON file that names only the
crane, task and input parameters that differ from the defaults
(`live.Settings`); a misspelt name is refused. Three candidates in
`tools/tuning/` are driven on the gamepad, three transports each, recorded:

| | Change | Aim |
|---|---|---|
| A | calm below 4° of swing instead of 2° | the threshold alone |
| B | A, plus damping 0.15/s instead of 0.05/s (a swing halves in about 9 s instead of 28 s) | the load helps calm itself |
| C | A, plus an input curve, command = 0.3·u + 0.7·u³ | finer commands for small deflections |

The input curve is applied in the server (`InputParams`), to the human's
input only, so the frontend holds no logic and an agent commands directly.
Recordings hold the shaped command, which is what moved the crane.

**The naive driver must still swing visibly** (§6). Full speed, then a hard
stop, 200 transports, seed 0, computed on the deployment machine:

| Set | Swing when stopped (median) | after 25 s (median) | counted calm after 25 s |
|---|---|---|---|
| current | 24.3° | 14.6° | 4 % (below 2°) |
| A, C | 24.3° | 14.6° | 8 % (below 4°) |
| B | 21.6° | 4.8° | 31 % (below 4°) |

B keeps the swing when the naive driver stops, but lets it fade within
seconds afterwards, so the contrast lasts less long.

**Driven on the gamepad** (deployment machine, the maintainer, 2026-10-09,
in the order A, B, C; the first three transports of each recorded):

| | Transports | Calming at the target | Command near the target, median | Maintainer |
|---|---|---|---|---|
| before | 22–56 s, 2 of 5 timed out | 12–51 s | — | "very hard" |
| A | 19–31 s | 12–23 s | 0.56 | "a little easier" |
| B | 11–18 s | 6–13 s | 0.78 | "easier still, almost too easy" |
| **C** | 10–22 s | 1–16 s | 0.06 | "finer at the target, more realistic" |

With C the driving changed, not only its speed: in two of three transports
the load arrived nearly calm (calm within 1.1 and 1.3 s of arrival), so the
swing was taken out on the way, and near the target the stick worked with
small commands instead of large ones back and forth. Industrial crane
controls have such a curve, which is why it felt more realistic.

**Why C.**

- *It makes skill visible rather than unnecessary.* B's damping did part of
  the work and shortens the naive driver's visible swing; C keeps the
  physics of §6 and lets the driver calm the load.
- *Style shows in the data.* Behavioural cloning learns from what the
  teacher did; with C cautious and quick driving differ in how the swing is
  handled, not in how long the teacher waits for it to fade.
- *The naive contrast is unchanged* (table above): the curve applies to the
  human only.

**Caveats.** One driver, three transports per candidate, and practice: C
came last, after more than a dozen transports, so part of its advantage may
be practice. The difference in *how* it was driven is hard to explain by
practice alone.

**Rejected alternatives.**

- *The old threshold of 2°:* the first run above.
- *B, more damping:* easiest, but the swing fades on its own and the naive
  contrast lasts less long.
- *A wider position tolerance:* changed nothing in the recorded runs.
- *The curve in the browser:* logic in the frontend (§3).

**Open** (§5): 3° or 4°, and the curve's strength, decided with people who
have not driven before. The files in `tools/tuning/` keep A and B
reproducible against the new defaults.

`tests/test_live.py` guards the settings file and the input curve.

---

## 15. Behavioural cloning: three features, a small network or a linear policy, NumPy

**Context.** The heart of the demonstrator (concept §5.1): from a recorded
run (§13), a model learns to map what the crane looks like to the command
the teacher gave, and then drives on its own.

**Decision.**

- *Features* (`features.py`, in the core, batched): distance to the target,
  `tanh((target − x) / 2 m)`; rope angle `θ / 0.2 rad`; angular speed
  `ω / (ω₀ · 0.2 rad)`. Fixed scales, so every agent shares one input space.
- *Models* (`cloning.py`): an MLP 3 → 8 → 1 with tanh in both layers
  (41 weights), and a linear policy `tanh(w · f + b)` with the same
  interface, the fallback of concept §5.2 item 7. Both are batched, so one
  call drives one agent or a swarm, and both are JSON.
- *Training:* full-batch Adam on the mean squared error, a fixed number of
  epochs (2000, learning rate 0.01), initial weights from an injected
  generator: the same runs and seed give the same agent bit for bit. Only
  frames of a running transport are used (§13). The loss curve is returned
  for the network display (concept §6.2).
- *Mirroring* (`dataset(..., mirror=True)`): every example once more
  reflected left to right, all features and the command negated. Offered,
  not the default; the evaluation decides.
- *`rollout()`* drives n transports at once with the live crane's `step()`
  and `advance()`; evaluation, ghost replay and swarm build on it.

**Why the trolley speed is not a feature.** The concept listed it, and the
first version had it. The command is a target speed which the drive reaches
within about 0.3 s, so the current speed nearly *is* the last command: in
the maintainer's cautious drive the two correlate at 0.993. Trained on it,
both models learnt "keep doing what you are doing" with a tiny loss
(0.0007) and failed at driving: from rest the linear policy never started,
and the network drifted, partly away from the target. Measured on the
deployment machine, 2026-10-09, clones of the two drives in `tests/data/`,
ten new transports (seed 42), first network seed:

| Drive | Model | with speed | without speed |
|---|---|---|---|
| cautious | linear | 0/10 | 10/10, 30 s, swing 5° |
| cautious | network | 0/10 | 0/10 (other seeds 3/10, 8/10) |
| quick | linear | 10/10 | 10/10, 8.8 s, swing 14° |
| quick | network | 0/10 | 6/10 (mirrored 10/10) |

The same trap is why the last command is not a feature either.

**First look at real drives** (same setting; median transport time of the
successful transports, median of each transport's largest swing). The maintainer drove 21–42 s with
4–14° cautiously and 12–26 s with 19–32° quickly.

| Drive | Model | Success | Time | Swing |
|---|---|---|---|---|
| cautious | linear | 10/10 | 30.2 s | 5.0° |
| cautious | linear, mirrored | 10/10 | 28.0 s | 5.0° |
| quick | linear | 10/10 | 8.8 s | 14.1° |
| quick | linear, mirrored | 10/10 | 8.8 s | 14.0° |
| cautious | network, mirrored | 9/10 | 45.6 s | 35.2° |
| quick | network, mirrored | 10/10 | 13.6 s | 30.4° |

The linear policy reaches every target and keeps the style; the network
depends strongly on its seed and swings far more than its teacher, the
covariate shift of concept §5.2. Which model the stand uses, and whether
mirroring becomes the default, is decided by the evaluation over more
seeds and targets, not by this table. Training took 0.05–0.10 s for the
linear policy and 0.3–2.4 s for the network.

**Details that are deliberate.**

- *Fixed scales, not fitted to the data:* a scale fitted to a cautious
  teacher would differ from a quick one's, and agents of the gallery and
  the evolution could no longer be compared or mixed.
- *Gradients by hand,* checked against finite differences (largest error
  1e-10), rather than an autodiff library: the network has 41 weights.
- *A fixed number of epochs,* not a stopping rule on the loss: the
  training time on the stand stays predictable, and a stopping rule would
  make the agent depend on the floating-point path to it.

**Rejected alternatives.**

- *The trolley speed and the last command as features:* see above.
- *Normalising by the data's mean and spread:* see fixed scales.
- *PyTorch or another framework:* a dependency of hundreds of megabytes for
  41 weights, against "standard library first" (CLAUDE.md).
- *Mini-batches or stochastic gradient descent:* a few thousand examples
  fit in one batch, and full batches keep training deterministic.

`tests/test_cloning.py` guards this section, with scripted teachers
(`tests/teachers.py`) driving and recorded through the live session like a
visitor.

