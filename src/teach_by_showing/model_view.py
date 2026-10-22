"""The learned model as the stand shows it (DECISIONS.md §21): one neuron,
its three inputs and their weights, the formula, and the learning replayed
in slow motion. Builds the `model` part of a state message; knows nothing
about HTTP or the frontend.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from teach_by_showing.cloning import MLP, LinearPolicy, Policy, clone
from teach_by_showing.crane import FloatArray
from teach_by_showing.recording import Run
from teach_by_showing.texts import Texts, number

# The words are in texts.py: the inputs' labels name only the quantities, no
# interpretation of what a weight means for the style (decided 2026-10-16);
# that is in model_explain.
# A weight of this size draws the thickest connection. The quick teaching
# drive gives 1.8, the cautious one 0.3 (§21).
WEIGHT_SCALE: Final[float] = 2.0
# Snapshots of the training, spaced evenly on a logarithmic scale of the
# epochs: the cautious drive's weights settle within 50 of 2000 epochs and
# would show no movement on a linear scale.
SNAPSHOTS: Final[int] = 48


def _signed(value: float, words: Texts) -> str:
    return ("−" if value < 0 else "+") + number(abs(value), 2, words)


def formula(policy: LinearPolicy, words: Texts) -> str:
    """The policy as one line, in the language's number style and with real
    minus signs."""
    terms: list[tuple[float, str]] = [
        (float(w), f" · {name}") for w, name in zip(policy.w, words.model_short, strict=True)
    ]
    terms.append((float(policy.b), ""))
    text: str = ""
    for k, (value, name) in enumerate(terms):
        if k == 0:
            sign: str = "−" if value < 0 else ""
        else:
            sign = " − " if value < 0 else " + "
        text += f"{sign}{number(abs(value), 2, words)}{name}"
    return f"{words.command_label} = tanh({text})"


@dataclass(frozen=True, slots=True)
class LearningTrace:
    """The learning as recorded for the slow motion: the policy and the loss
    at a few epochs, and how long the real training took."""

    epochs: tuple[int, ...]
    policies: tuple[LinearPolicy, ...]
    losses: tuple[float, ...]
    seconds: float

    @property
    def points(self) -> int:
        return len(self.epochs)


def snapshot_epochs(epochs: int, count: int = SNAPSHOTS) -> frozenset[int]:
    spaced: FloatArray = np.geomspace(1, epochs, count - 1)
    return frozenset({0, *np.round(spaced).astype(int).tolist()})


def learn(
    runs: Sequence[Run], rng: np.random.Generator, epochs: int = 2000
) -> tuple[Policy, LearningTrace | None]:
    """Clone the runs as cloning.clone() does, keeping snapshots for the
    slow motion. None for a policy the display cannot show (the network)."""
    keep: frozenset[int] = snapshot_epochs(epochs)
    seen: list[tuple[int, Policy]] = []

    def on_epoch(epoch: int, policy: Policy) -> None:
        if epoch in keep:
            seen.append((epoch, policy))

    begin: float = time.perf_counter()
    policy, losses = clone(runs, rng, on_epoch=on_epoch)
    seconds: float = time.perf_counter() - begin
    if isinstance(policy, MLP):
        return policy, None
    # The loss before epoch k is losses[k]; after the last epoch, the
    # trained policy's, which train() does not compute.
    final: float = float(losses[-1])
    trace: LearningTrace = LearningTrace(
        tuple(e for e, _ in seen),
        tuple(p for _, p in seen if isinstance(p, LinearPolicy)),
        tuple(float(losses[e]) if e < len(losses) else final for e, _ in seen),
        seconds,
    )
    return policy, trace


def model_message(
    policy: LinearPolicy,
    words: Texts,
    inputs: FloatArray | None = None,
    output: float | None = None,
    trace: LearningTrace | None = None,
    shown: int = 0,
) -> dict[str, Any]:
    """The `model` part of a state message. With `inputs` and `output` the
    neuron is shown at work; with `trace`, its learning up to snapshot
    `shown`, and the weights of that snapshot."""
    message: dict[str, Any] = {
        "labels": list(words.model_labels),
        "output_label": words.command_label,
        "w": policy.w.tolist(),
        # Formatted here, so that the scripts hold no wording or number style.
        "w_text": [_signed(float(w), words) for w in policy.w],
        "b": float(policy.b),
        "weight_scale": WEIGHT_SCALE,
        "inputs": None if inputs is None else inputs.tolist(),
        "output": output,
        "formula": formula(policy, words),
        "explain": words.model_explain,
        "loss": None,
    }
    if trace is not None:
        # Positions in 0..1 on a logarithmic epoch axis and a loss axis from
        # 0 to the largest loss, so that the frontend only scales them.
        last: int = max(trace.epochs)
        top: float = max(trace.losses)
        message["loss"] = {
            "label": words.loss_label,
            "axis": words.loss_axis,
            "points": [
                [float(np.log1p(e) / np.log1p(last)), loss / top]
                for e, loss in zip(
                    trace.epochs[: shown + 1], trace.losses[: shown + 1], strict=True
                )
            ],
        }
    return message
