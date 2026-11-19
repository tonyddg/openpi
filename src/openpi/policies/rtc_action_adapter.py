"""Adapter between model `actions`, policy output action keys, and raw robot inputs.

RTC session/RPC action tensors are always [H, D] in *robot output space*.
The public Policy output action key may be different (e.g. `action`).
`input_action_key` selects where to inject those robot-space actions BEFORE
running the policy's original input transforms to recover model-space actions.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np


def validate_action_key(key: str, *, label: str) -> str:
    if not isinstance(key, str) or not key or key.strip() != key:
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    if label == "output_action_key" and key == "rtc":
        raise ValueError("output_action_key='rtc' is reserved for client diagnostic metadata")
    return key


def extract_robot_actions(
    output: Mapping[str, Any], *, output_action_key: str, horizon: int
) -> np.ndarray:
    """Select the policy's robot-space action chunk without changing its output."""
    key = validate_action_key(output_action_key, label="output_action_key")
    if not isinstance(output, Mapping):
        raise ValueError(f"Policy output must be a dict, got {type(output).__name__}")
    if key not in output:
        raise ValueError(
            f"Policy output missing configured action key {key!r}; "
            f"available keys: {list(output)}. "
            "Set --output-action-key to the key returned by data_transforms.outputs."
        )
    try:
        actions = np.asarray(output[key], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Policy output {key!r} must be a numeric [H, D] action tensor") from exc
    if actions.ndim != 2 or actions.shape[0] != horizon or actions.shape[1] < 1:
        raise ValueError(
            f"Policy output {key!r} must have shape [H, D] with H={horizon}; "
            f"got {actions.shape}. Split/multi-field action outputs need a custom adapter."
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"Policy output {key!r} contains non-finite actions")
    return actions.copy()


def inject_robot_actions(raw: dict, *, input_action_key: str, actions: np.ndarray) -> dict:
    """Inject into the *raw observation* key expected by the repack/input pipeline.

    Supports a direct top-level key (including literal dots), or a slash path
    for an actual nested dictionary, as used by flatten_dict/RepackTransform.
    """
    key = validate_action_key(input_action_key, label="input_action_key")
    if key in raw or "/" not in key:
        raw[key] = actions.copy()
        return raw
    parts = key.split("/")
    if not all(parts):
        raise ValueError(f"Invalid input_action_key path: {key!r}")
    cursor = raw
    for name in parts[:-1]:
        if name not in cursor:
            cursor[name] = {}
        if not isinstance(cursor[name], dict):
            raise ValueError(f"Cannot inject {key!r}: segment {name!r} is not a dict")
        cursor = cursor[name]
    cursor[parts[-1]] = actions.copy()
    return raw


def prepare_model_prefix(
    input_transform: Callable[[dict], dict],
    obs: dict,
    aligned_robot_actions: np.ndarray,
    *,
    input_action_key: str,
    horizon: int,
    model_action_dim: int,
) -> tuple[dict, np.ndarray]:
    """Re-encode robot-space previous chunk using the current observation.

    Crucial for DeltaActions: old robot-space actions must be converted using
    the *current* state, then normalized/padded, not directly passed to guidance.
    """
    raw = copy.deepcopy(obs)
    inject_robot_actions(raw, input_action_key=input_action_key,
                         actions=np.asarray(aligned_robot_actions, dtype=np.float32))
    converted = input_transform(raw)
    if not isinstance(converted, dict) or "actions" not in converted:
        keys = list(converted) if isinstance(converted, dict) else type(converted).__name__
        raise ValueError(
            f"Input transforms did not produce model-space 'actions' from raw key "
            f"{input_action_key!r}; output keys: {keys}. "
            "Set --input-action-key to the raw key consumed by the input/repack transforms "
            "(which may differ from --output-action-key)."
        )
    model_prefix = np.asarray(converted.pop("actions"), dtype=np.float32)
    expected_shape = (horizon, model_action_dim)
    if model_prefix.shape != expected_shape or not np.isfinite(model_prefix).all():
        raise ValueError(
            f"Model prefix actions expected finite {expected_shape}, got {model_prefix.shape}"
        )
    return converted, model_prefix
