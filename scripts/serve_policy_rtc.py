"""Serve standard Pi0.5 + guided RTC from one JAX checkpoint and one port.

Example:
  uv run scripts/serve_policy_rtc.py --config pi05_aloha \\
      --checkpoint-dir checkpoints/pi05_aloha --port 8000 --s 20 --d 5
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import tyro

from openpi.policies import policy_config
from openpi.policies.policy_rtc import RTCPolicy
from openpi.serving.rtc_session_manager import RTCConfig
from openpi.serving.rtc_websocket_server import RTCWebsocketPolicyServer
from openpi.training import config as train_config


@dataclasses.dataclass
class Args:
    config: str
    checkpoint_dir: str
    host: str = "127.0.0.1"
    port: int = 8000
    s: int = 20
    d: int = 5
    control_period_s: float = 0.02
    num_steps: int = 10
    max_guidance_weight: float = 5.0
    schedule: str = "exp"
    warmup: bool = True


def main(args: Args):
    config = train_config.get_config(args.config)
    if not getattr(config.model, "pi05", False):
        raise ValueError("Expected a Pi0.5 JAX/NNX training config (pi05=True)")
    base = policy_config.create_trained_policy(
        config, args.checkpoint_dir, sample_kwargs={"num_steps": args.num_steps},
    )
    H = int(base._model.action_horizon)
    rtc_config = RTCConfig(
        H=H, s=args.s, d=args.d, control_period_s=args.control_period_s,
        max_guidance_weight=args.max_guidance_weight, schedule=args.schedule,
    )
    guided = RTCPolicy(base, schedule=args.schedule,
                       max_guidance_weight=args.max_guidance_weight)
    # Compile both JAX branches before admitting robots. The guided branch
    # performs VJP and can have a substantially different compilation cost.
    if args.warmup:
        fake_obs = config.model.fake_obs(batch_size=1)
        rng = jax.random.key(42)
        jax.block_until_ready(base._sample_actions(rng, fake_obs, num_steps=args.num_steps))
        jax.block_until_ready(guided._sample(
            rng, fake_obs, num_steps=args.num_steps,
            prefix_actions=jnp.zeros((1, H, int(base._model.action_dim))),
            prefix_weights=jnp.ones((H,)),
            max_guidance_weight=jnp.asarray(args.max_guidance_weight),
        ))
        logging.info("Standard and RTC JAX samplers warmup completed")
    server = RTCWebsocketPolicyServer(
        base, guided, config=rtc_config, host=args.host, port=args.port,
    )
    logging.info("Serving Pi0.5 with H=%d, s=%d, d=%d", H, args.s, args.d)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
