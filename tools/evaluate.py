#!/usr/bin/env python3
"""Measure how well clones of recorded drives drive, against the acceptance
of the spike's second day (concept §16, DECISIONS.md §16). Runs in the venv:

    pdm run python tools/evaluate.py tests/data/*vorsichtig.json tests/data/*zügig.json
    pdm run python tools/evaluate.py RUN.json ... --json results.json

Every run is cloned on its own, as a visitor's would be, in four variants
(network or linear policy, each with and without mirrored examples) and
with several training seeds, since the network depends strongly on its
seed. Each clone drives the same sets of new transports, so that variants
are compared and not their luck, and once more the teacher's own
transports, from the states the teacher started in.

Teacher and clone are measured the same way: success by task.advance(),
transport time to calm, largest swing amplitude while running. Every figure
recorded from this tool names the machine it ran on.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from teach_by_showing.cloning import MLP, LinearPolicy, Rollout, dataset, rollout, train
from teach_by_showing.crane import SIM_HZ, CraneState, FloatArray, swing_amplitude
from teach_by_showing.recording import Run, load_run
from teach_by_showing.task import Transport, advance, sample_transports

EPISODE: Final[int] = 10  # transports per episode, as in the acceptance
ENOUGH: Final[int] = 8  # "most of ten": a clone that stops looks like a bug
VARIANTS: Final[tuple[tuple[str, bool], ...]] = (
    ("linear", False),
    ("linear", True),
    ("mlp", False),
    ("mlp", True),
)


@dataclass(frozen=True)
class Outcome:
    """Per transport: success, seconds to calm (None if not reached), and
    the largest swing amplitude while running, in degrees."""

    success: list[bool]
    seconds: list[float | None]
    swing: list[float]


def variant_name(kind: str, mirror: bool) -> str:
    return f"{kind}{' mirrored' if mirror else ''}"


def outcome_of(transport: Transport, max_swing: FloatArray) -> Outcome:
    frames: list[int] = transport.duration_frames().tolist()
    return Outcome(
        transport.succeeded.tolist(),
        [f / SIM_HZ if f >= 0 else None for f in frames],
        np.degrees(max_swing).tolist(),
    )


def _row_state(run: Run, rows: np.ndarray, frame: int) -> CraneState:
    return CraneState(
        frame,
        run.column("x")[rows],
        run.column("v")[rows],
        run.column("theta")[rows],
        run.column("omega")[rows],
    )


def teacher_transports(run: Run) -> tuple[CraneState, FloatArray]:
    """The state each of the teacher's transports began in, and its target.
    A later transport begins where the pause before it left the crane,
    which is not quite at rest."""
    transport: np.ndarray = np.asarray(run.transport)
    running: np.ndarray = run.running_mask()
    firsts: list[int] = [
        int(np.flatnonzero((transport == i) & running)[0]) for i in range(run.transports)
    ]
    columns: dict[str, FloatArray] = {}
    for name in ("x", "v", "theta", "omega"):
        # The row before a transport's first; the first transport begins
        # with the recording, in the state before row 0.
        before: list[float] = [
            float(getattr(run.initial, name)[0]) if f == 0 else run.columns[name][f - 1]
            for f in firsts
        ]
        columns[name] = np.asarray(before)
    begin: CraneState = CraneState(
        0, columns["x"], columns["v"], columns["theta"], columns["omega"]
    )
    return begin, run.column("target")[np.asarray(firsts)]


def teacher_outcome(run: Run) -> Outcome:
    """The teacher's transports judged by the same advance() as a clone's."""
    transport: np.ndarray = np.asarray(run.transport)
    running: np.ndarray = run.running_mask()
    amp: FloatArray = swing_amplitude(_row_state(run, np.arange(run.frames), 0), run.crane)
    success: list[bool] = []
    seconds: list[float | None] = []
    swing: list[float] = []
    for i in range(run.transports):
        rows: np.ndarray = np.flatnonzero((transport == i) & running)
        tr: Transport = Transport.begin(run.column("target")[rows[:1]], 0)
        for k, row in enumerate(rows, start=1):
            tr = advance(tr, _row_state(run, np.asarray([row]), k), run.crane, run.task)
        one: Outcome = outcome_of(tr, np.asarray([float(amp[rows].max())]))
        success += one.success
        seconds += one.seconds
        swing += one.swing
    return Outcome(success, seconds, swing)


def evaluate(run: Run, seeds: int, target_sets: int, epochs: int = 2000) -> dict[str, Any]:
    """Every variant, trained with `seeds` seeds, each clone driving
    `target_sets` episodes of EPISODE new transports and the teacher's own."""
    n: int = target_sets * EPISODE
    start, target = sample_transports(np.random.default_rng(1000), run.crane, run.task, n)
    own_state, own_target = teacher_transports(run)
    variants: dict[str, Any] = {}
    for kind, mirror in VARIANTS:
        f, u = dataset([run], mirror=mirror)
        new: list[dict[str, Any]] = []
        own: list[dict[str, Any]] = []
        seconds: list[float] = []
        for seed in range(seeds):
            rng: np.random.Generator = np.random.default_rng(seed)
            initial: MLP | LinearPolicy = (
                MLP.initial(rng) if kind == "mlp" else LinearPolicy.initial(rng)
            )
            t0: float = time.perf_counter()
            policy, _ = train(initial, f, u, epochs=epochs)
            seconds.append(time.perf_counter() - t0)
            r: Rollout = rollout(
                policy, CraneState.at_rest(start), target, run.crane, run.task
            )
            new.append(asdict(outcome_of(r.transport, r.max_swing)))
            r = rollout(policy, own_state, own_target, run.crane, run.task)
            own.append(asdict(outcome_of(r.transport, r.max_swing)))
        variants[variant_name(kind, mirror)] = {
            "new": new,
            "own": own,
            "train_seconds": seconds,
        }
    return {"label": run.label, "teacher": asdict(teacher_outcome(run)), "variants": variants}


def episode_successes(new: Sequence[dict[str, Any]]) -> list[int]:
    """Successes per episode of EPISODE transports, over all seeds."""
    counts: list[int] = []
    for outcome in new:
        s: list[bool] = outcome["success"]
        counts += [sum(s[i : i + EPISODE]) for i in range(0, len(s), EPISODE)]
    return counts


def _median_seconds(outcomes: Sequence[dict[str, Any]]) -> float:
    values: list[float] = [s for o in outcomes for s in o["seconds"] if s is not None]
    return statistics.median(values) if values else float("nan")


def _median_swing(outcomes: Sequence[dict[str, Any]]) -> float:
    return statistics.median([s for o in outcomes for s in o["swing"]])


def report(results: Sequence[dict[str, Any]], cautious: str, quick: str) -> list[str]:
    lines: list[str] = []
    for res in results:
        teacher: dict[str, Any] = res["teacher"]
        lines.append(
            f"{res['label']}: teacher {sum(teacher['success'])}/{len(teacher['success'])},"
            f" {_median_seconds([teacher]):.1f} s, swing {_median_swing([teacher]):.1f}°"
            " (medians over the teacher's transports)"
        )
        for name, v in res["variants"].items():
            counts: list[int] = episode_successes(v["new"])
            own: int = statistics.median(sum(o["success"]) for o in v["own"])
            lines += [
                f"  {name:16s} new: {statistics.median(counts):.0f}/{EPISODE} median"
                f" (range {min(counts)}–{max(counts)}),"
                f" {sum(c >= ENOUGH for c in counts)}/{len(counts)} episodes"
                f" with ≥{ENOUGH}; {_median_seconds(v['new']):.1f} s,"
                f" swing {_median_swing(v['new']):.1f}°",
                f"  {'':16s} own: {own:.0f}/{len(teacher['success'])} median,"
                f" {_median_seconds(v['own']):.1f} s, swing {_median_swing(v['own']):.1f}°;"
                f" training {statistics.median(v['train_seconds']):.2f} s median",
            ]
    by_label: dict[str, dict[str, Any]] = {r["label"]: r for r in results}
    if cautious in by_label and quick in by_label:
        lines.append(f"style, {cautious} against {quick}, per seed on new transports:")
        for name in by_label[cautious]["variants"]:
            slow: list[dict[str, Any]] = by_label[cautious]["variants"][name]["new"]
            fast: list[dict[str, Any]] = by_label[quick]["variants"][name]["new"]
            # Seeds whose clones reached no target on one side have no time.
            pairs: list[tuple[float, float]] = [
                (
                    _median_seconds([a]) / _median_seconds([b]),
                    _median_swing([a]) / _median_swing([b]),
                )
                for a, b in zip(slow, fast, strict=True)
            ]
            t: list[float] = [p[0] for p in pairs if np.isfinite(p[0])] or [float("nan")]
            s: list[float] = [p[1] for p in pairs]
            lines.append(
                f"  {name:16s} time ×{statistics.median(t):.2f}"
                f" (range {min(t):.2f}–{max(t):.2f}),"
                f" swing ×{statistics.median(s):.2f} (range {min(s):.2f}–{max(s):.2f})"
            )
    return lines


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Clone recorded drives and measure how the clones drive."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="recorded runs (JSON)")
    parser.add_argument("--seeds", type=int, default=10, help="training seeds; default: 10")
    parser.add_argument(
        "--target-sets", type=int, default=5, help=f"episodes of {EPISODE} new transports"
    )
    parser.add_argument("--cautious", default="vorsichtig", help="label of the cautious run")
    parser.add_argument("--quick", default="zügig", help="label of the quick run")
    parser.add_argument("--json", type=Path, help="write every single figure to this file")
    args: argparse.Namespace = parser.parse_args()

    results: list[dict[str, Any]] = [
        evaluate(load_run(path), args.seeds, args.target_sets) for path in args.runs
    ]
    print("\n".join(report(results, args.cautious, args.quick)))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
