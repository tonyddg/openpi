"""Action-time constraint weights for inference-time guided RTC."""

from __future__ import annotations

import numpy as np


def prefix_weights(*, delay: int, overlap: int, horizon: int,
                   schedule: str = "exp") -> np.ndarray:
    """Length-H weights: [0,delay) pinned; [delay,overlap) tapered; rest 0.

    delay is a *predicted* number of control ticks consumed by inference, not
    a claim about how many actions the robot has actually executed.
    """
    if not (0 <= delay <= overlap <= horizon):
        raise ValueError(f"Require 0 <= delay <= overlap <= horizon, got {delay, overlap, horizon}")
    if schedule not in {"exp", "linear", "hard"}:
        raise ValueError(f"Unknown schedule: {schedule}")
    weights = np.zeros(horizon, dtype=np.float32)
    weights[:delay] = 1.0
    if overlap > delay:
        x = np.arange(overlap - delay, dtype=np.float32) / (overlap - delay)
        if schedule == "linear":
            weights[delay:overlap] = 1.0 - x
        elif schedule == "exp":
            weights[delay:overlap] = np.expm1(2.0 * (1.0 - x)) / np.expm1(2.0)
        else:
            weights[delay:overlap] = 1.0
    return weights
