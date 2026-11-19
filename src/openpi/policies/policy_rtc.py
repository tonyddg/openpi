"""Wrap an already-loaded upstream Policy; reuse its exact robot/model transforms.

Requires JAX/NNX Pi0 or Pi0.5. Torch weights are intentionally rejected.
"""

from __future__ import annotations

import copy
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0_rtc import sample_actions_rtc
from openpi.models.rtc_guidance import prefix_weights
from openpi.shared import nnx_utils


class RTCPolicy:
    def __init__(self, base_policy, *, schedule: str = "exp", max_guidance_weight: float = 5.0):
        if getattr(base_policy, "_is_pytorch_model", False):
            raise TypeError("This RTC sampler requires the JAX/NNX Pi0.5 checkpoint, not PyTorch safetensors")
        model = base_policy._model
        if not hasattr(model, "embed_prefix") or not hasattr(model, "embed_suffix"):
            raise TypeError("RTC sampler only supports upstream Pi0/Pi0.5 flow models")
        self._base = base_policy
        self._model = model
        self._schedule = schedule
        self._max_weight = float(max_guidance_weight)
        # module_jit requires a bound NNX method; no patch to the model class.
        rtc_method = types.MethodType(sample_actions_rtc, model)
        self._sample = nnx_utils.module_jit(rtc_method)

    @property
    def action_horizon(self) -> int:
        return int(self._model.action_horizon)

    def infer(self, obs: dict, *, previous_chunk: np.ndarray | None = None,
              previous_start: int = 0, inference_delay: int = 0) -> dict:
        if previous_chunk is None:
            return self._base.infer(obs)

        H = self.action_horizon
        prev = np.asarray(previous_chunk, dtype=np.float32)
        if prev.ndim != 2 or prev.shape[0] != H:
            raise ValueError(f"Expected previous robot-space chunk [H,D] where H={H}, got {prev.shape}")
        if not 0 <= previous_start < H:
            raise ValueError("Invalid previous_start")
        overlap = H - previous_start
        if not 0 <= inference_delay <= overlap:
            raise ValueError("Expected inference delay must fit previous overlap")
        aligned = np.empty_like(prev)
        aligned[:overlap] = prev[previous_start:]
        aligned[overlap:] = prev[-1]

        # Crucial: rebase DeltaActions to THIS observation's state and apply the
        # SAME normalization/padding as the original training input pipeline.
        raw = copy.deepcopy(obs)
        raw["actions"] = aligned.copy()
        converted = self._base._input_transform(raw)
        model_prefix = converted.pop("actions", None)
        if model_prefix is None:
            raise ValueError("Input transforms dropped actions; adapt the robot-specific repack mapping")
        model_prefix = np.asarray(model_prefix, dtype=np.float32)
        if model_prefix.shape != (H, int(self._model.action_dim)):
            raise ValueError(f"Model-space prefix has wrong shape: {model_prefix.shape}")
        weights = prefix_weights(delay=inference_delay, overlap=overlap,
                                 horizon=H, schedule=self._schedule)

        model_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], converted)
        observation = _model.Observation.from_dict(model_inputs)
        self._base._rng, rng = jax.random.split(self._base._rng)
        kwargs = dict(self._base._sample_kwargs)
        kwargs.update(prefix_actions=jnp.asarray(model_prefix)[None, ...],
                      prefix_weights=jnp.asarray(weights),
                      max_guidance_weight=jnp.asarray(self._max_weight))
        start = time.monotonic()
        model_actions = jax.block_until_ready(self._sample(rng, observation, **kwargs))
        elapsed = (time.monotonic() - start) * 1000
        outputs = {"state": model_inputs["state"], "actions": model_actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0]), outputs)
        outputs = self._base._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": elapsed}
        outputs["rtc_active"] = True
        outputs["rtc_overlap"] = overlap
        outputs["rtc_delay_estimated"] = inference_delay
        return outputs
