"""Measurement tool (DECISIONS.md §8, §12, §18). Only the arithmetic of its
reports; the figures themselves come from runs against a live server and are
recorded in DECISIONS.md with their machine."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from teach_by_showing.live import LiveSession
from tests.support import ROOT


def load_measure() -> ModuleType:
    # tools/ is not a package; the script must stay runnable on its own.
    path = ROOT / "tools" / "measure_ws.py"
    spec = importlib.util.spec_from_file_location("measure_ws", path)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["measure_ws"] = module
    spec.loader.exec_module(module)
    return module


measure: ModuleType = load_measure()


def test_a_steady_60_hz_stream_reports_no_skips_and_no_drift() -> None:
    samples: list[object] = [measure.Sample(i / 60, 100 + i, 0.0) for i in range(601)]
    report: str = "\n".join(measure.timing_report(samples, 60.0))
    assert "60.00 Hz, 0 frames skipped" in report
    assert "median 16.67 ms" in report
    assert "+0.0 ms over 10.0 s" in report


def test_a_skipped_frame_is_counted_and_shows_as_a_long_interval() -> None:
    frames: list[int] = [*range(10), *range(11, 20)]
    samples: list[object] = [measure.Sample(f / 60, f, 0.0) for f in frames]
    report: str = "\n".join(measure.timing_report(samples, 60.0))
    assert "1 frames skipped" in report
    assert "max 33.33 ms" in report


def test_the_trace_shows_the_range_of_the_command_per_bin() -> None:
    commands: list[float] = [0.0] * 30 + [1.0] * 30 + [-0.5] * 30
    samples: list[object] = [measure.Sample(i / 60, i, u) for i, u in enumerate(commands)]
    lines: list[str] = measure.trace_report(samples, 0.5)
    assert lines == [
        "   0.0 s  +0.00 .. +0.00",
        "   0.5 s  +1.00 .. +1.00",
        "   1.0 s  -0.50 .. -0.50",
    ]


def test_percentile_takes_the_nearest_rank() -> None:
    values: list[float] = [float(v) for v in range(101)]
    assert measure.percentile(values, 0) == 0.0
    assert measure.percentile(values, 95) == 95.0
    assert measure.percentile(values, 100) == 100.0


def test_roundtrip_expects_the_command_the_server_shapes() -> None:
    """Since the input curve (§14) a state shows the shaped command, not the
    deflection sent; the tool waited for the raw value and hung until
    2026-10-09."""
    session: LiveSession = LiveSession.start(np.random.default_rng(0))
    config: dict[str, Any] = session.config_message()
    session.set_input(0.3125)
    assert measure.expected_command(config, 0.3125) == session.effective_command != 0.3125


def test_roundtrip_refuses_the_stands_steps() -> None:
    """In the stand's steps an agent drives while the stand waits (§18); no
    state would ever show the command, and the tool would wait forever."""
    with pytest.raises(SystemExit, match="--free"):
        measure.expected_command({"mode": "stand", "input": {"expo": 0.0}}, 0.5)
