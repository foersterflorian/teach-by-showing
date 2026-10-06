"""Demonstrations as files: every frame a teacher drove, with the command
applied and the state it led to (DECISIONS.md §13).

Recording turns imitation learning into an offline problem: a run can be
trained on, replayed and tested without hardware. The module knows nothing
about HTTP, the frontend or the AI layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
from numpy.typing import NDArray

from teach_by_showing.crane import CraneParams, CraneState, FloatArray
from teach_by_showing.task import TaskParams

# Raised whenever the meaning of a field changes, so that an old file is
# refused instead of misread.
SCHEMA: Final[int] = 1
COLUMNS: Final[tuple[str, ...]] = ("command", "x", "v", "theta", "omega", "target")


@dataclass(slots=True)
class Run:
    """One recorded demonstration of a single crane.

    Row k holds the command applied in frame `first_frame + k` and the state
    that step produced; `initial` is the state before row 0. Columns, not a
    list of frame objects: a file of 5000 frames stays readable and loads
    straight into arrays."""

    label: str
    seed: int | None
    created: str
    crane: CraneParams
    task: TaskParams
    initial: CraneState
    columns: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in COLUMNS}
    )
    # Index of the transport each frame belongs to, counted from 0.
    transport: list[int] = field(default_factory=list)
    # False for frames after a transport has ended, during the pause before
    # the next target: the teacher is not driving then, so training skips them.
    running: list[bool] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.transport)

    @property
    def transports(self) -> int:
        return self.transport[-1] + 1 if self.transport else 0

    def append(
        self, command: float, state: CraneState, target: float, transport: int, running: bool
    ) -> None:
        values: dict[str, float] = {
            "command": command,
            "x": float(state.x[0]),
            "v": float(state.v[0]),
            "theta": float(state.theta[0]),
            "omega": float(state.omega[0]),
            "target": target,
        }
        for name in COLUMNS:
            self.columns[name].append(values[name])
        self.transport.append(transport)
        self.running.append(running)

    def column(self, name: str) -> FloatArray:
        return np.asarray(self.columns[name], dtype=np.float64)

    def running_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.running, dtype=np.bool_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "label": self.label,
            "seed": self.seed,
            "created": self.created,
            "crane": self.crane.to_dict(),
            "task": self.task.to_dict(),
            "initial": self.initial.to_dict(),
            "columns": self.columns,
            "transport": self.transport,
            "running": self.running,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"run has schema {data.get('schema')!r}, expected {SCHEMA}")
        columns: dict[str, list[float]] = {
            name: [float(v) for v in data["columns"][name]] for name in COLUMNS
        }
        transport: list[int] = [int(v) for v in data["transport"]]
        running: list[bool] = [bool(v) for v in data["running"]]
        if any(len(c) != len(transport) for c in columns.values()) or len(running) != len(
            transport
        ):
            raise ValueError("run columns differ in length")
        return cls(
            str(data["label"]),
            None if data["seed"] is None else int(data["seed"]),
            str(data["created"]),
            CraneParams.from_dict(data["crane"]),
            TaskParams.from_dict(data["task"]),
            CraneState.from_dict(data["initial"]),
            columns,
            transport,
            running,
        )


def run_filename(created: str, label: str) -> str:
    """`<created>-<label>.json`, with everything but letters, digits and
    hyphens dropped from both, so that a label can never leave the folder."""

    def clean(text: str) -> str:
        return re.sub(r"[^\w-]|_", "", text.replace(" ", "-")) or "run"

    return f"{clean(created)}-{clean(label)}.json"


def save_run(run: Run, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Path = directory / run_filename(run.created, run.label)
    # Written to a temporary name first, so that a crash never leaves half a
    # file that looks like a run.
    partial: Path = path.with_suffix(".partial")
    partial.write_text(json.dumps(run.to_dict()), encoding="utf-8")
    partial.replace(path)
    return path


def load_run(path: Path) -> Run:
    return Run.from_dict(json.loads(path.read_text(encoding="utf-8")))
