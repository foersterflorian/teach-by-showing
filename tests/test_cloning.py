"""Behavioural cloning: features, dataset, training, rollout
(DECISIONS.md §15)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from teach_by_showing.cloning import (
    INPUTS,
    MLP,
    LinearPolicy,
    Rollout,
    dataset,
    policy_from_dict,
    rollout,
    train,
)
from teach_by_showing.crane import CraneParams, CraneState, FloatArray
from teach_by_showing.features import features
from teach_by_showing.recording import Run, load_run
from teach_by_showing.task import TaskParams, sample_transports
from tests.support import ROOT
from tests.teachers import cautious, quick, teach

PARAMS: CraneParams = CraneParams()
TASK: TaskParams = TaskParams()


@cache
def taught(name: str) -> Run:
    return teach({"cautious": cautious, "quick": quick}[name])


def state(x: float, v: float = 0.0, theta: float = 0.0, omega: float = 0.0) -> CraneState:
    return CraneState(0, np.array([x]), np.array([v]), np.array([theta]), np.array([omega]))


def test_features_are_zero_for_a_calm_load_at_the_target() -> None:
    f: FloatArray = features(state(4.0), np.array([4.0]), PARAMS)
    assert f.shape == (1, INPUTS)
    assert np.all(f == 0.0)


def test_distance_saturates_far_from_the_target() -> None:
    f: FloatArray = features(state(0.5), np.array([9.5]), PARAMS)
    assert 0.99 < f[0, 0] < 1.0


def test_features_ignore_the_trolley_speed() -> None:
    """The speed gives away the teacher's last command; with it, the clone
    of a cautious drive copied its own motion and never started (§15)."""
    a: FloatArray = features(state(3.0, v=0.0), np.array([7.0]), PARAMS)
    b: FloatArray = features(state(3.0, v=1.0), np.array([7.0]), PARAMS)
    assert np.array_equal(a, b)


def test_mirroring_the_scene_negates_every_feature() -> None:
    f: FloatArray = features(state(3.0, 0.4, 0.1, -0.3), np.array([7.0]), PARAMS)
    g: FloatArray = features(state(7.0, -0.4, -0.1, 0.3), np.array([3.0]), PARAMS)
    assert np.allclose(f, -g)


def test_the_dataset_leaves_out_pauses_and_mirrors_on_request() -> None:
    run: Run = taught("quick")
    f, u = dataset([run])
    assert f.shape == (int(np.sum(run.running_mask())), INPUTS)
    assert np.array_equal(u, run.column("command")[run.running_mask()])
    fm, um = dataset([run], mirror=True)
    assert np.array_equal(fm, np.concatenate([f, -f]))
    assert np.array_equal(um, np.concatenate([u, -u]))


@pytest.mark.parametrize("kind", [MLP, LinearPolicy])
def test_training_lowers_the_loss_and_is_reproducible(kind: type[MLP | LinearPolicy]) -> None:
    f, u = dataset([taught("quick")])
    a, losses = train(kind.initial(np.random.default_rng(3)), f, u, epochs=300)
    b, _ = train(kind.initial(np.random.default_rng(3)), f, u, epochs=300)
    assert losses[-1] < 0.5 * losses[0]
    assert all(np.array_equal(p, q) for p, q in zip(a.params(), b.params(), strict=True))


@pytest.mark.parametrize("kind", [MLP, LinearPolicy])
def test_a_policy_survives_json(kind: type[MLP | LinearPolicy]) -> None:
    p: MLP | LinearPolicy = kind.initial(np.random.default_rng(0))
    q: MLP | LinearPolicy = policy_from_dict(json.loads(json.dumps(p.to_dict())))
    assert type(q) is kind
    assert all(np.array_equal(a, b) for a, b in zip(p.params(), q.params(), strict=True))


def clone_drives(name: str, kind: type[MLP | LinearPolicy]) -> Rollout:
    """The clone of a scripted teacher on ten transports it never saw."""
    run: Run = taught(name)
    f, u = dataset([run])
    policy, _ = train(kind.initial(np.random.default_rng(0)), f, u)
    start, target = sample_transports(np.random.default_rng(1), PARAMS, TASK, 10)
    return rollout(policy, CraneState.at_rest(start), target, PARAMS, TASK)


@pytest.mark.parametrize("kind", [MLP, LinearPolicy])
@pytest.mark.parametrize("name", ["cautious", "quick"])
def test_the_clone_of_a_scripted_teacher_reaches_new_targets(
    name: str, kind: type[MLP | LinearPolicy]
) -> None:
    assert int(clone_drives(name, kind).transport.succeeded.sum()) >= 8


@pytest.mark.parametrize("kind", [MLP, LinearPolicy])
def test_the_clones_keep_their_teachers_style(kind: type[MLP | LinearPolicy]) -> None:
    """The style the concept promises: cautious is slower and swings less."""
    slow: Rollout = clone_drives("cautious", kind)
    fast: Rollout = clone_drives("quick", kind)
    both: np.ndarray = slow.transport.succeeded & fast.transport.succeeded
    assert np.median(slow.transport.duration_frames()[both]) > 1.3 * np.median(
        fast.transport.duration_frames()[both]
    )
    assert np.median(slow.max_swing) < 0.7 * np.median(fast.max_swing)


@pytest.mark.parametrize("path", sorted((ROOT / "tests" / "data").glob("*.json")))
def test_training_on_a_recorded_teacher_runs(path: Path) -> None:
    """Whether the clones of real drives succeed is measured, not tested
    (tools/evaluate.py); here only that the chain runs on them."""
    f, u = dataset([load_run(path)], mirror=True)
    policy, losses = train(MLP.initial(np.random.default_rng(0)), f, u, epochs=200)
    assert np.all(np.isfinite(losses)) and losses[-1] < losses[0]
    assert all(np.all(np.isfinite(p)) for p in policy.params())
