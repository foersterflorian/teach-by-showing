"""What a learned driver sees: the crane's state relative to its target, as a
few numbers of order one (DECISIONS.md §15).

The trolley's speed is deliberately not among them: the drive reaches the
commanded speed within a fraction of a second, so the speed gives away the
teacher's last command and a model learns to copy it instead of driving.

Batched like the crane: one row per crane. The scales are fixed, not fitted
to a teacher's data, so that every agent of the day shares one input space.
The module knows nothing about learning, HTTP or the frontend.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from teach_by_showing.crane import CraneParams, CraneState, FloatArray

# Beyond about this distance the driver goes full speed whatever the exact
# distance; tanh saturates there and resolves the last metres finely.
DISTANCE_SCALE: Final[float] = 2.0  # m
# About 11°: a well-driven swing stays within ±1, a naive one reaches 2–3.
SWING_SCALE: Final[float] = 0.2  # rad
NAMES: Final[tuple[str, ...]] = ("distance", "angle", "angular_speed")


def features(state: CraneState, target: FloatArray, params: CraneParams) -> FloatArray:
    """Shape (n, 3): distance to the target (saturated), rope angle and
    angular speed. All three change sign when the scene is mirrored left to
    right, which cloning.dataset() relies on."""
    return np.stack(
        [
            np.tanh((np.asarray(target, dtype=np.float64) - state.x) / DISTANCE_SCALE),
            state.theta / SWING_SCALE,
            state.omega / (params.omega0 * SWING_SCALE),
        ],
        axis=1,
    )
