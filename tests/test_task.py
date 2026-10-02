"""Transport task (DECISIONS.md §7)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from teach_by_showing.crane import SIM_HZ, CraneParams, CraneState, FloatArray, step
from teach_by_showing.task import (
    TaskParams,
    Transport,
    advance,
    is_calm,
    sample_targets,
    sample_transports,
)

CRANE: CraneParams = CraneParams()
TASK: TaskParams = TaskParams()


def test_sampling_is_reproducible_from_the_seed() -> None:
    a = sample_transports(np.random.default_rng(3), CRANE, TASK, 50)
    b = sample_transports(np.random.default_rng(3), CRANE, TASK, 50)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_sampled_transports_are_admissible_and_go_both_ways() -> None:
    start, target = sample_transports(np.random.default_rng(0), CRANE, TASK, 5000)
    lo: float = CRANE.track_min + TASK.margin
    hi: float = CRANE.track_max - TASK.margin
    for pos in (start, target):
        assert np.all((pos >= lo) & (pos <= hi))
    assert np.all(np.abs(target - start) >= TASK.min_distance - 1e-12)
    assert np.any(target > start) and np.any(target < start)


def test_sampling_refuses_a_track_too_short() -> None:
    with pytest.raises(ValueError):
        sample_transports(np.random.default_rng(0), CRANE, replace(TASK, min_distance=5.0), 1)


def at_target(target: float, theta: float = 0.0, omega: float = 0.0) -> CraneState:
    s: CraneState = CraneState.at_rest(np.array([target]))
    return replace(s, theta=np.array([theta]), omega=np.array([omega]))


def test_load_passing_the_vertical_is_not_calm() -> None:
    """θ = 0 but swinging hard: the criterion uses the swing amplitude, not θ."""
    assert is_calm(at_target(5.0), np.array([5.0]), CRANE, TASK)[0]
    assert not is_calm(at_target(5.0, omega=0.5), np.array([5.0]), CRANE, TASK)[0]


def test_calm_load_succeeds_after_calm_seconds() -> None:
    s: CraneState = at_target(5.0)
    tr: Transport = Transport.begin(np.array([5.0]), s.frame)
    frames: int = 0
    while not tr.succeeded[0]:
        s = step(s, np.zeros(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
        frames += 1
    assert frames == TASK.calm_frames == SIM_HZ
    assert tr.duration_frames()[0] == SIM_HZ


def test_interrupted_calm_starts_over() -> None:
    s: CraneState = at_target(5.0)
    tr: Transport = Transport.begin(np.array([5.0]), s.frame)
    for _ in range(TASK.calm_frames - 1):
        s = step(s, np.zeros(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    s = replace(s, omega=np.array([0.5]))
    tr = advance(tr, s, CRANE, TASK)
    assert tr.calm_frames[0] == 0
    assert not tr.succeeded[0]


def test_transport_times_out_and_then_stays_finished() -> None:
    s: CraneState = at_target(2.0)
    tr: Transport = Transport.begin(np.array([8.0]), s.frame)
    for _ in range(TASK.limit_frames):
        s = step(s, np.zeros(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    assert tr.timed_out(s.frame, TASK)[0] and tr.done(s.frame, TASK)[0]
    # Arriving after the limit does not count.
    s = at_target(8.0)
    s = replace(s, frame=TASK.limit_frames + 1)
    for _ in range(2 * TASK.calm_frames):
        s = step(s, np.zeros(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    assert not tr.succeeded[0]
    assert tr.duration_frames()[0] == -1


def test_succeeded_transport_is_frozen() -> None:
    s: CraneState = at_target(5.0)
    tr: Transport = Transport.begin(np.array([5.0]), s.frame)
    for _ in range(TASK.calm_frames):
        s = step(s, np.zeros(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    done: dict[str, object] = tr.to_dict()
    for _ in range(SIM_HZ):
        s = step(s, np.ones(1), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    assert tr.to_dict() == done


def test_batch_tracks_each_transport_separately() -> None:
    """Three cranes: one calm at its target, one swinging there, one far away."""
    s: CraneState = CraneState.at_rest(np.array([5.0, 5.0, 1.0]))
    s = replace(s, omega=np.array([0.0, 0.5, 0.0]))
    targets: FloatArray = np.array([5.0, 5.0, 8.0])
    tr: Transport = Transport.begin(targets, s.frame)
    for _ in range(TASK.calm_frames):
        s = step(s, np.zeros(3), CRANE)
        tr = advance(tr, s, CRANE, TASK)
    assert tr.succeeded.tolist() == [True, False, False]


def test_transport_and_params_survive_json() -> None:
    tr: Transport = Transport.begin(np.array([3.0, 7.5]), 12)
    tr = advance(tr, at_target(3.0), CRANE, TASK)
    back: Transport = Transport.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert back.to_dict() == tr.to_dict()
    assert TaskParams.from_dict(json.loads(json.dumps(TASK.to_dict()))) == TASK


@pytest.mark.parametrize("start", [0.0, 0.3, 5.0, 9.8, 10.0])
def test_targets_from_anywhere_on_the_track(start: float) -> None:
    """The live loop draws the next target from wherever the trolley stands,
    including the ends after a timed-out transport."""
    starts: FloatArray = np.full(500, start)
    target: FloatArray = sample_targets(np.random.default_rng(0), starts, CRANE, TASK)
    assert np.all(np.abs(target - starts) >= TASK.min_distance - 1e-12)
    assert np.all((target >= TASK.margin) & (target <= CRANE.track_max - TASK.margin))
