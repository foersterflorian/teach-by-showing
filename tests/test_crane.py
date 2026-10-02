"""Crane physics (DECISIONS.md §6)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from teach_by_showing.crane import (
    DT,
    SIM_HZ,
    CraneParams,
    CraneState,
    FloatArray,
    step,
    swing_amplitude,
)

PARAMS: CraneParams = CraneParams()
UNDAMPED: CraneParams = replace(PARAMS, damping=0.0)


def swinging(theta0: float, x: float = 5.0) -> CraneState:
    """One crane standing still with its load released at angle theta0."""
    s: CraneState = CraneState.at_rest(np.array([x]))
    return replace(s, theta=np.array([theta0]))


def run(
    state: CraneState, params: CraneParams, seconds: float, command: float = 0.0
) -> list[CraneState]:
    states: list[CraneState] = [state]
    u: FloatArray = np.full(state.n, command)
    for _ in range(round(seconds * SIM_HZ)):
        states.append(step(states[-1], u, params))
    return states


def test_small_angle_period_matches_analytic() -> None:
    """The rope length is chosen for a period of 2.5 s (§6); the integrator
    must reproduce it."""
    states: list[CraneState] = run(swinging(0.05), UNDAMPED, 20.0)
    theta: FloatArray = np.array([s.theta[0] for s in states])
    # Upward zero crossings, refined by linear interpolation.
    k: FloatArray = np.flatnonzero((theta[:-1] < 0) & (theta[1:] >= 0)).astype(np.float64)
    i: np.ndarray = k.astype(int)
    crossings: FloatArray = (k + theta[i] / (theta[i] - theta[i + 1])) * DT
    period: float = float(np.mean(np.diff(crossings)))
    assert period == pytest.approx(2 * math.pi / PARAMS.omega0, rel=2e-3)
    assert period == pytest.approx(2.5, abs=0.01)


def test_energy_stays_bounded_without_damping() -> None:
    """Semi-implicit Euler, not explicit Euler: a free swing must neither gain
    nor lose energy over minutes."""
    g, length = UNDAMPED.gravity, UNDAMPED.rope_length

    def energy(s: CraneState) -> float:
        th, om = float(s.theta[0]), float(s.omega[0])
        return g * length * (1 - math.cos(th)) + 0.5 * length**2 * om**2

    states: list[CraneState] = run(swinging(0.5), UNDAMPED, 120.0)
    e0: float = energy(states[0])
    drift: float = max(abs(energy(s) - e0) for s in states) / e0
    assert drift < 0.01


def test_damping_halves_amplitude_in_expected_time() -> None:
    half_life: float = 2 * math.log(2) / PARAMS.damping
    states: list[CraneState] = run(swinging(0.05), PARAMS, half_life)
    ratio: float = float(swing_amplitude(states[-1], PARAMS)[0] / 0.05)
    assert ratio == pytest.approx(0.5, abs=0.02)


def test_drive_respects_speed_and_acceleration_limits() -> None:
    """The safety layer under every driver (§6): wild commands, including
    values far outside −1..1, never exceed the drive's limits."""
    rng: np.random.Generator = np.random.default_rng(1)
    s: CraneState = CraneState.at_rest(np.full(8, 5.0))
    for _ in range(20 * SIM_HZ):
        nxt: CraneState = step(s, rng.uniform(-5, 5, size=8), PARAMS)
        assert np.all(np.abs(nxt.v) <= PARAMS.max_speed + 1e-12)
        assert np.all(np.abs(nxt.v - s.v) <= PARAMS.max_accel * DT + 1e-12)
        s = nxt


def test_command_is_clipped() -> None:
    a: CraneState = run(CraneState.at_rest(np.array([2.0])), PARAMS, 3.0, command=1.0)[-1]
    b: CraneState = run(CraneState.at_rest(np.array([2.0])), PARAMS, 3.0, command=7.0)[-1]
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize(("x0", "command", "end"), [(5.0, 1.0, 10.0), (5.0, -1.0, 0.0)])
def test_trolley_brakes_early_and_stops_at_the_end(
    x0: float, command: float, end: float
) -> None:
    """Full speed towards an end stop: the drive slows down in time and the
    trolley comes to rest at the end, never beyond it."""
    states: list[CraneState] = run(CraneState.at_rest(np.array([x0])), PARAMS, 30.0, command)
    xs: FloatArray = np.array([s.x[0] for s in states])
    assert np.all((xs >= PARAMS.track_min) & (xs <= PARAMS.track_max))
    assert states[-1].x[0] == pytest.approx(end, abs=0.01)
    assert abs(states[-1].v[0]) < 0.01


def test_batch_equals_individual_cranes() -> None:
    """One code path for one crane and for a swarm (§6): n cranes in one batch
    behave exactly like n separate runs."""
    rng: np.random.Generator = np.random.default_rng(7)
    n: int = 5
    x0: FloatArray = rng.uniform(1, 9, size=n)
    commands: FloatArray = rng.uniform(-1, 1, size=(10 * SIM_HZ, n))

    batch: CraneState = CraneState.at_rest(x0)
    singles: list[CraneState] = [CraneState.at_rest(x0[i : i + 1]) for i in range(n)]
    for u in commands:
        batch = step(batch, u, PARAMS)
        singles = [step(s, u[i : i + 1], PARAMS) for i, s in enumerate(singles)]

    for i, s in enumerate(singles):
        assert batch.x[i] == s.x[0]
        assert batch.v[i] == s.v[0]
        assert batch.theta[i] == s.theta[0]
        assert batch.omega[i] == s.omega[0]


def test_step_does_not_mutate_its_input() -> None:
    s: CraneState = swinging(0.3)
    before: dict[str, object] = s.to_dict()
    step(s, np.array([1.0]), PARAMS)
    assert s.to_dict() == before


def test_time_counts_frames() -> None:
    states: list[CraneState] = run(swinging(0.0), PARAMS, 100.0)
    assert states[-1].frame == 100 * SIM_HZ
    assert states[-1].t == pytest.approx(100.0, abs=1e-12)


def test_state_and_params_survive_json_exactly() -> None:
    """All state is JSON-serialisable (CLAUDE.md), without loss: a recorded
    run replays bit for bit."""
    s: CraneState = run(CraneState.at_rest(np.array([1.0, 6.0])), PARAMS, 4.3, 0.7)[-1]
    back: CraneState = CraneState.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.to_dict() == s.to_dict()
    assert (
        step(back, np.array([-1.0, 1.0]), PARAMS).to_dict()
        == step(s, np.array([-1.0, 1.0]), PARAMS).to_dict()
    )

    p: CraneParams = replace(PARAMS, rope_length=2.0)
    assert CraneParams.from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_swing_amplitude_is_constant_over_a_free_swing() -> None:
    """|θ| drops to zero twice per period; the amplitude must not, or a load
    swinging through the vertical would count as calm."""
    states: list[CraneState] = run(swinging(0.05), UNDAMPED, 2.5)
    amps: FloatArray = np.array([swing_amplitude(s, UNDAMPED)[0] for s in states])
    assert np.all(np.abs(amps - 0.05) < 0.05 * 0.01)
