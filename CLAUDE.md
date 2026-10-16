# CLAUDE.md

Project instructions for Claude Code. Keep this file short — it is prepended to
every session.

## What this is

A demonstrator for a research project (K-M-I, TU Chemnitz): visitors steer a
simulated overhead crane with a swinging load; a small learned model (by
default a linear policy) picks up their driving style from the demonstration
within seconds (behavioural cloning), and an evolution strategy refines it towards goals the human sets.
It runs on an exhibition stand next to its sibling `feature-viz`, one monitor
each, both in Docker containers.

It is shown to visitors and handed to colleagues who did not write it. Clarity
and reproducibility outrank cleverness.

## Read first

`DECISIONS.md` records why the code is shaped the way it is, including the
alternatives that were rejected and why. Read it before changing anything
structural. If you change one of those decisions, update that section — do not
delete it.

`DECISIONS.md` §5 lists what has been reasoned about but never measured. Treat
those as open tasks, not as facts.

The concept (German, internal, not part of this repository) lives in
`../Konzept/`, a separate local Git repository; each document carries its
version in its title, and the highest version is authoritative. It is the
source for scope and priorities; this repository records the technical
decisions and must remain understandable without the concept.

## Conventions

- **English only.** Code, comments, docstrings, commit messages, documentation.
  On-screen texts for visitors are German; that is content, not convention.
- **Full type annotations** on every function signature, class attribute, and
  any local whose type is not obvious. `mypy src tests tools` must stay clean.
- **Tests protect decisions.** Each test module names the `DECISIONS.md`
  section it guards. Run `pytest` before every commit.
- **One code path.** The live stand, training and headless runs share the same
  simulation; they differ by parameters, never by a second implementation.
- **Standard library first.** Every dependency is a future breakage; state why
  it is needed in `DECISIONS.md` when adding one.
- **Determinism.** Fixed time step; no global `random` in the core — the
  generator is injected. All state is JSON-serialisable.
- **Layering.** The simulation core knows nothing about HTTP, the frontend or
  the AI layer. The frontend holds no logic, only display and input.
- Comments explain *why*, not *what*. The annotations already say what.
- **Commit messages** as in feature-viz: English, lower case, imperative,
  optionally with a scope prefix such as `docs:` or `tests:`.
- **Branches.** `dev` is the development branch. Every feature or section gets
  its own branch from `dev`; after the maintainer has reviewed it, it is
  fast-forwarded into `dev`. `dev` is fast-forwarded into `main` only on the
  maintainer's command. Never merge or fast-forward without that review.
- **Remote.** `origin` is the public GitHub repository
  `foersterflorian/teach-by-showing`. Push only on the maintainer's command.

## Commands

```bash
pdm install                              # venv with all groups
mypy src tests tools
ruff check src tests tools
pytest
```

Outside an activated venv, prefix with `pdm run`.

## Machines

Two machines, both Pop!_OS with CUDA; `nvidia-smi` tells which one you are on.
This is the current pair, the same as for feature-viz; others may join.

- **Deployment:** RTX 4090, AMD 16-core, two 4K monitors. Runs the stand; the
  figures that matter are the ones measured here.
- **Development:** private box, RTX 4070 Ti (12 GB), Ryzen 7 5700G. Figures
  from here are a lower bound for the deployment machine.

Every recorded measurement names its machine.

## Rules

- **Verify, do not assert.** If a claim about timing, latency or learning
  quality matters, measure it and record the figure with its machine. If you
  cannot verify something, say so and add it to `DECISIONS.md` §5.
- Commit before starting a substantial change, so it can be reverted cleanly.
- **The maintainer** is fluent in Python but has no JavaScript or frontend
  experience. Keep the frontend plain and thin, so that a fault can be traced
  without knowing JavaScript; logic belongs in Python.
- When asked to review or check something, present the findings and a plan
  first; change files only after approval.
- The next step and the open points with their deadlines are in the concept
  (§16 and §15 of the highest version under `../Konzept/`).
