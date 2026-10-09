"""Evaluation tool (DECISIONS.md §16). Its arithmetic, on scripted teachers;
the figures for real drives come from runs on the deployment machine and
are recorded in DECISIONS.md with their machine."""

from __future__ import annotations

import importlib.util
import sys
from functools import cache
from types import ModuleType
from typing import Any

import numpy as np

from teach_by_showing.crane import SIM_HZ
from teach_by_showing.recording import Run
from tests.support import ROOT
from tests.teachers import cautious, quick, teach


def load_evaluate() -> ModuleType:
    # tools/ is not a package; the script must stay runnable on its own.
    path = ROOT / "tools" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", path)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


ev: ModuleType = load_evaluate()


@cache
def run_of(name: str) -> Run:
    run: Run = teach({"cautious": cautious, "quick": quick}[name])
    run.label = name
    return run


def test_the_teacher_is_judged_like_a_clone() -> None:
    """Success by advance(), time from the start to the end of the calm
    second: the same as the live session counted while recording."""
    run: Run = run_of("quick")
    outcome = ev.teacher_outcome(run)
    transport: np.ndarray = np.asarray(run.transport)
    running: np.ndarray = run.running_mask()
    assert outcome.success == [True] * run.transports
    for i, seconds in enumerate(outcome.seconds):
        assert seconds == np.sum((transport == i) & running) / SIM_HZ
    assert all(0.0 < s < 45.0 for s in outcome.swing)


def test_the_teachers_transports_begin_where_the_teacher_began() -> None:
    run: Run = run_of("quick")
    begin, target = run.transport_begins()
    assert begin.n == run.transports
    assert begin.x[0] == run.initial.x[0]
    transport: np.ndarray = np.asarray(run.transport)
    for i in range(run.transports):
        first: int = int(np.flatnonzero((transport == i) & run.running_mask())[0])
        assert target[i] == run.columns["target"][first]
        if i > 0:
            assert begin.x[i] == run.columns["x"][first - 1]


def test_successes_are_counted_per_episode_of_ten() -> None:
    outcome: dict[str, Any] = {"success": [True] * 10 + [True] * 7 + [False] * 3}
    assert ev.episode_successes([outcome, outcome]) == [10, 7, 10, 7]


def results() -> list[dict[str, Any]]:
    # Few epochs: these tests check the bookkeeping, not how well clones drive.
    return [
        ev.evaluate(run_of(name), seeds=2, target_sets=1, epochs=200)
        for name in ("cautious", "quick")
    ]


def test_the_evaluation_is_reproducible_apart_from_the_clock() -> None:
    def strip(res: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for r in res:
            for v in r["variants"].values():
                v.pop("train_seconds")
        return res

    assert strip(results()) == strip(results())


def test_the_report_names_every_variant_and_the_style() -> None:
    text: str = "\n".join(ev.report(results(), "cautious", "quick"))
    for kind, mirror in ev.VARIANTS:
        assert ev.variant_name(kind, mirror) in text
    assert "style, cautious against quick" in text
    assert f"with ≥{ev.ENOUGH}" in text
