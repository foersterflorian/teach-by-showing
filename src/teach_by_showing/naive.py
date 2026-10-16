"""The naive driver: what a crane does without anyone teaching it
(DECISIONS.md §6, §19). Full speed towards the target, braking just in time
to stop there, the swing ignored. It reaches the target and leaves the load
swinging, which is the contrast the stand's agents are measured against.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from teach_by_showing.crane import CraneParams, CraneState, FloatArray

# Braking starts a little early, so that the trolley does not overshoot by
# the distance it covers in the frame the decision is late.
BRAKING_MARGIN: Final[float] = 0.02


def naive_command(state: CraneState, target: FloatArray, crane: CraneParams) -> FloatArray:
    """Full speed while the target is further than the braking distance,
    then zero, which the drive turns into braking at its limit."""
    braking: FloatArray = state.v**2 / (2 * crane.max_accel)
    gap: FloatArray = target - state.x
    command: FloatArray = np.where(np.abs(gap) > braking + BRAKING_MARGIN, np.sign(gap), 0.0)
    return command
