"""Guided Pi0/Pi0.5 sampler compatible with the upstream JAX/NNX model.

Kept as a free function so the upstream Pi0 implementation and checkpoint
format remain unchanged. Adapt this sampler if upstream sample_actions changes.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


def sample_actions_rtc(model, rng, observation: _model.Observation, *,
                       prefix_actions, prefix_weights,
                       max_guidance_weight=5.0, num_steps=10, noise=None):
    """Sample actions under an aligned, *normalized model-space* soft prefix.

    Both prefix arrays are required. First chunk uses the stock Pi0 sampler.
    OpenPI convention: t=1 noisy; t=0 clean, dx/dt = velocity.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch, model.action_horizon, model.action_dim))
    if prefix_actions.shape != noise.shape:
        raise ValueError(f"Expected prefix_actions {noise.shape}, got {prefix_actions.shape}")
    if prefix_weights.shape != (model.action_horizon,):
        raise ValueError("Expected per-step prefix weights of shape (action_horizon,)")

    # Prefix is computed once; JAX VJP below only differentiates w.r.t. the noisy action.
    tokens, mask, ar_mask = model.embed_prefix(observation)
    _, cache = model.PaliGemma.llm(
        [tokens, None], mask=make_attn_mask(mask, ar_mask),
        positions=jnp.cumsum(mask, axis=-1) - 1,
    )

    def velocity(x, t):
        suffix, suffix_mask, suffix_ar, conditioning = model.embed_suffix(
            observation, x, jnp.broadcast_to(t, (batch,))
        )
        suffix_self_mask = make_attn_mask(suffix_mask, suffix_ar)
        suffix_to_prefix = einops.repeat(mask, "b p -> b q p", q=suffix.shape[1])
        all_mask = jnp.concatenate([suffix_to_prefix, suffix_self_mask], axis=-1)
        suffix_position = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm(
            [None, suffix], mask=all_mask, positions=suffix_position,
            kv_cache=cache, adarms_cond=[None, conditioning],
        )
        return model.action_out_proj(out[:, -model.action_horizon:])

    dt = -1.0 / num_steps

    def step(carry):
        x, t = carry

        def denoised(y):
            v = velocity(y, t)
            return y - t * v, v

        x_clean, pullback, v = jax.vjp(denoised, x, has_aux=True)
        # Pull the current estimated clean sample toward the previous chunk.
        error = (prefix_actions - x_clean) * prefix_weights[None, :, None]
        correction = pullback(error)[0]
        remaining = jnp.maximum(t, 1e-3)
        completed = jnp.maximum(1.0 - t, 1e-3)
        scale = (t / completed) * ((t * t + (1.0 - t) ** 2) / (remaining * remaining))
        weight = jnp.minimum(scale, jnp.asarray(max_guidance_weight))
        v_guided = v - weight * correction  # Negative sign: OpenPI time runs 1 -> 0.
        return (x + dt * v_guided, t + dt)

    def cond(carry):
        return carry[1] >= -dt / 2

    result, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0)))
    return result
