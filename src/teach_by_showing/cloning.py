"""Behavioural cloning: a small model learns to map what the crane looks
like to what the teacher did (DECISIONS.md §15). By default a linear policy;
a small network (MLP) stays available.

Plain NumPy, full batch, a fixed number of Adam steps and a generator passed
in, so that the same runs and seed give the same agent bit for bit. Policies
are batched: one call drives n cranes, the swarm of the evolution as well as
a single agent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, Self

import numpy as np

from teach_by_showing.crane import SIM_HZ, CraneParams, CraneState, FloatArray, step
from teach_by_showing.features import NAMES, features
from teach_by_showing.profile import Tally
from teach_by_showing.recording import Run
from teach_by_showing.task import TaskParams, Transport, advance

SCHEMA: Final[int] = 1
INPUTS: Final[int] = len(NAMES)
# The stand's choice, measured in DECISIONS.md §16: the linear policy reached
# every target from both teaching drives and kept their style for every
# seed; the network depended on its seed. Mirrored since 2026-10-22: without
# it a cautious drive's small distance weight let a learnt offset hold the
# agent 0.27 m beside its target, so that it never arrived; mirrored, the
# offset learns itself to zero and every recorded drive arrives (§16).
DEFAULT_KIND: Final[str] = "linear"
DEFAULT_MIRROR: Final[bool] = True
HIDDEN: Final[int] = 8  # 41 weights; concept §5.1 estimated 50–80 for five inputs


class Policy(Protocol):
    """Features (n, INPUTS) to commands (n,), each in −1..1."""

    def act(self, f: FloatArray) -> FloatArray: ...

    def params(self) -> list[FloatArray]: ...

    def loss_and_grads(self, f: FloatArray, u: FloatArray) -> tuple[float, list[FloatArray]]:
        """Mean squared error against the teacher's commands u, and its
        gradient for each array of params(), in the same order."""
        ...

    def with_params(self, params: list[FloatArray]) -> Self: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MLP:
    """INPUTS → HIDDEN → 1, tanh in both layers: 41 weights."""

    w1: FloatArray  # (INPUTS, HIDDEN)
    b1: FloatArray  # (HIDDEN,)
    w2: FloatArray  # (HIDDEN,)
    b2: FloatArray  # () — an array, so that Adam treats every weight alike

    @classmethod
    def initial(cls, rng: np.random.Generator) -> Self:
        # Scaled by fan-in, so that tanh starts in its linear range.
        return cls(
            rng.normal(0.0, 1.0 / np.sqrt(INPUTS), (INPUTS, HIDDEN)),
            np.zeros(HIDDEN),
            rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), HIDDEN),
            np.zeros(()),
        )

    def act(self, f: FloatArray) -> FloatArray:
        return np.tanh(np.tanh(f @ self.w1 + self.b1) @ self.w2 + self.b2)

    def hidden(self, f: FloatArray) -> FloatArray:
        """Activations of the hidden layer, (n, HIDDEN): what the network
        display shows (concept §6.2)."""
        return np.tanh(f @ self.w1 + self.b1)

    def params(self) -> list[FloatArray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def loss_and_grads(self, f: FloatArray, u: FloatArray) -> tuple[float, list[FloatArray]]:
        h: FloatArray = self.hidden(f)
        y: FloatArray = np.tanh(h @ self.w2 + self.b2)
        err: FloatArray = y - u
        dz2: FloatArray = 2.0 * err / err.shape[0] * (1.0 - y * y)
        dz1: FloatArray = np.outer(dz2, self.w2) * (1.0 - h * h)
        grads: list[FloatArray] = [
            f.T @ dz1,
            dz1.sum(axis=0),
            h.T @ dz2,
            np.asarray(dz2.sum()),
        ]
        return float(np.mean(err * err)), grads

    def with_params(self, params: list[FloatArray]) -> Self:
        return type(self)(*params)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "kind": "mlp"} | {
            name: getattr(self, name).tolist() for name in ("w1", "b1", "w2", "b2")
        }


@dataclass(frozen=True, slots=True)
class LinearPolicy:
    """tanh(w·f + b): the fallback if the network drives too restlessly
    (concept §5.2 item 7)."""

    w: FloatArray  # (INPUTS,)
    b: FloatArray  # ()

    @classmethod
    def initial(cls, rng: np.random.Generator) -> Self:
        return cls(rng.normal(0.0, 1.0 / np.sqrt(INPUTS), INPUTS), np.zeros(()))

    def act(self, f: FloatArray) -> FloatArray:
        return np.tanh(f @ self.w + self.b)

    def params(self) -> list[FloatArray]:
        return [self.w, self.b]

    def loss_and_grads(self, f: FloatArray, u: FloatArray) -> tuple[float, list[FloatArray]]:
        y: FloatArray = self.act(f)
        err: FloatArray = y - u
        dz: FloatArray = 2.0 * err / err.shape[0] * (1.0 - y * y)
        return float(np.mean(err * err)), [f.T @ dz, np.asarray(dz.sum())]

    def with_params(self, params: list[FloatArray]) -> Self:
        return type(self)(*params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "linear",
            "w": self.w.tolist(),
            "b": self.b.tolist(),
        }


def policy_from_dict(data: dict[str, Any]) -> MLP | LinearPolicy:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"policy has schema {data.get('schema')!r}, expected {SCHEMA}")

    def arr(key: str) -> FloatArray:
        return np.asarray(data[key], dtype=np.float64)

    if data.get("kind") == "mlp":
        return MLP(arr("w1"), arr("b1"), arr("w2"), arr("b2"))
    if data.get("kind") == "linear":
        return LinearPolicy(arr("w"), arr("b"))
    raise ValueError(f"unknown policy kind {data.get('kind')!r}")


def dataset(runs: Sequence[Run], mirror: bool = False) -> tuple[FloatArray, FloatArray]:
    """Features and commands of every frame in which a transport was
    running; pauses are left out, since the teacher is not driving then.

    The features come from the recorded states through features(), the
    same function the agent sees while driving. With `mirror`, every
    example is added once more reflected left to right: the task is
    symmetric, and every feature and the command change sign."""
    fs: list[FloatArray] = []
    us: list[FloatArray] = []
    for run in runs:
        keep = run.running_mask()
        state: CraneState = CraneState(
            0,
            run.column("x")[keep],
            run.column("v")[keep],
            run.column("theta")[keep],
            run.column("omega")[keep],
        )
        fs.append(features(state, run.column("target")[keep], run.crane))
        us.append(run.column("command")[keep])
    f: FloatArray = np.concatenate(fs)
    u: FloatArray = np.concatenate(us)
    if mirror:
        return np.concatenate([f, -f]), np.concatenate([u, -u])
    return f, u


def train[P: Policy](
    policy: P,
    f: FloatArray,
    u: FloatArray,
    epochs: int = 2000,
    lr: float = 0.01,
    on_epoch: Callable[[int, P], None] | None = None,
) -> tuple[P, FloatArray]:
    """Adam on the full batch for a fixed number of epochs, starting from
    `policy` (fresh from MLP.initial(rng) or LinearPolicy.initial(rng)).
    Returns the trained policy and the loss before each epoch. `on_epoch`
    sees the policy before each epoch and, with `epochs`, the trained one:
    the stand keeps a few of them to show the learning in slow motion
    (DECISIONS.md §21)."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    params: list[FloatArray] = policy.params()
    m: list[FloatArray] = [np.zeros_like(p) for p in params]
    v: list[FloatArray] = [np.zeros_like(p) for p in params]
    losses: FloatArray = np.empty(epochs)
    for epoch in range(epochs):
        if on_epoch is not None:
            on_epoch(epoch, policy)
        loss, grads = policy.loss_and_grads(f, u)
        losses[epoch] = loss
        t: int = epoch + 1
        new: list[FloatArray] = []
        for p, g, mi, vi in zip(params, grads, m, v, strict=True):
            mi[...] = beta1 * mi + (1.0 - beta1) * g
            vi[...] = beta2 * vi + (1.0 - beta2) * g * g
            m_hat = mi / (1.0 - beta1**t)
            v_hat = vi / (1.0 - beta2**t)
            new.append(p - lr * m_hat / (np.sqrt(v_hat) + eps))
        params = new
        policy = policy.with_params(params)
    if on_epoch is not None:
        on_epoch(epochs, policy)
    return policy, losses


@dataclass(frozen=True, slots=True)
class Rollout:
    """Outcome of n transports driven by a policy."""

    transport: Transport
    # The style profile's sums (§22), counted while the drive runs.
    tally: Tally

    @property
    def max_swing(self) -> FloatArray:
        """Largest swing amplitude while running, rad."""
        return self.tally.max_swing


@dataclass(slots=True)
class Driving:
    """A rollout under way, advanced a few steps at a time, so that the
    stand can drive one beside its paced loop without stalling it (§23).
    rollout() is the same drive taken in one go."""

    policy: Policy
    state: CraneState
    transport: Transport
    tally: Tally
    crane: CraneParams
    task: TaskParams

    @classmethod
    def begin(
        cls,
        policy: Policy,
        state: CraneState,
        target: FloatArray,
        crane: CraneParams,
        task: TaskParams,
    ) -> Self:
        return cls(
            policy,
            state,
            Transport.begin(target, state.frame),
            Tally.begin(target),
            crane,
            task,
        )

    @property
    def done(self) -> bool:
        return bool(self.transport.done(self.state.frame, self.task).all())

    def drive(self, steps: int) -> bool:
        """At most `steps` frames; True once every transport has ended."""
        for _ in range(steps):
            if self.done:
                break
            running = ~self.transport.done(self.state.frame, self.task)
            command: FloatArray = self.policy.act(
                features(self.state, self.transport.target, self.crane)
            )
            self.state = step(self.state, command, self.crane)
            self.transport = advance(self.transport, self.state, self.crane, self.task)
            self.tally = self.tally.update(
                command, self.state, running, self.crane, self.task
            )
        return self.done

    def result(self) -> Rollout:
        return Rollout(self.transport, self.tally)


def rollout(
    policy: Policy,
    state: CraneState,
    target: FloatArray,
    crane: CraneParams,
    task: TaskParams,
) -> Rollout:
    """Drive n transports at once from `state` until every one has succeeded
    or timed out, with the same step() and advance() as the live crane.
    Usually the cranes start at rest (CraneState.at_rest); replaying a
    teacher's transport starts from the state the teacher started in."""
    driving: Driving = Driving.begin(policy, state, target, crane, task)
    while not driving.drive(SIM_HZ):
        pass
    return driving.result()


def clone(
    runs: Sequence[Run],
    rng: np.random.Generator,
    kind: str = DEFAULT_KIND,
    mirror: bool = DEFAULT_MIRROR,
    on_epoch: Callable[[int, Policy], None] | None = None,
) -> tuple[MLP | LinearPolicy, FloatArray]:
    """An agent from recorded drives, the way the stand makes one; returns
    it with its loss curve."""
    f, u = dataset(runs, mirror=mirror)
    if kind == "linear":
        return train(LinearPolicy.initial(rng), f, u, on_epoch=on_epoch)
    if kind == "mlp":
        return train(MLP.initial(rng), f, u, on_epoch=on_epoch)
    raise ValueError(f"unknown policy kind {kind!r}")
