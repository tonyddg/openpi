"""Benchmark batch inference latency for different batch sizes.

Usage:
    uv run python app/maniskill/bench_batch.py [--host 127.0.0.1] [--port 8000]
"""

import dataclasses
import time
import logging
import numpy as np

from openpi_client import websocket_client_policy as _wcp
from openpi.mshab.utility import ROT_TYPE, STATE_ROT_LENGTH_DICT


def make_mshab_example(state_rot_type: ROT_TYPE = "rot_6d") -> dict:
    state_length = 4 + STATE_ROT_LENGTH_DICT[state_rot_type]
    return {
        "observation/state": np.random.rand(state_length).astype(np.float32),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _make_batched_obs(obs_list: list[dict]) -> dict:
    """Stack a list of observation dicts into a batched dict (axis=0 for arrays, list for strings)."""
    result = {}
    for key in obs_list[0]:
        val = obs_list[0][key]
        if isinstance(val, dict):
            result[key] = _make_batched_obs([o[key] for o in obs_list])
        elif isinstance(val, str):
            result[key] = [o[key] for o in obs_list]
        else:
            result[key] = np.stack([np.asarray(o[key]) for o in obs_list], axis=0)
    return result


@dataclasses.dataclass
class BenchResult:
    batch_size: int
    jit_ms: float       # first call (includes JIT for new batch size)
    steady_ms: float    # average of subsequent calls (stable)
    per_sample_ms: float  # steady / batch_size


def run_benchmark(client: _wcp.WebsocketClientPolicy, batch_sizes: list[int], warmup: int = 4, repeats: int = 10):
    """Run benchmark for each batch size, recording JIT and steady timing separately."""
    results: list[BenchResult] = []

    max_bs = max(batch_sizes)
    all_obs = [make_mshab_example() for _ in range(max_bs)]

    for bs in batch_sizes:
        batched_obs = _make_batched_obs(all_obs[:bs])

        # Time each call individually
        all_times = []
        for trial in range(warmup + repeats):
            start = time.perf_counter()
            if bs == 1:
                client.infer(all_obs[0])
            else:
                client.infer_batched(batched_obs)
            elapsed = (time.perf_counter() - start) * 1000
            all_times.append(elapsed)

        # First call includes JIT, rest are steady
        jit_ms = all_times[0]
        steady_ms = np.mean(all_times[warmup:])
        results.append(BenchResult(bs, jit_ms, steady_ms, steady_ms / bs))

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    print(f"Connecting to {args.host}:{args.port}...")
    client = _wcp.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"max_batch_size from server: {client._max_batch_size}")
    print()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    if client._max_batch_size is not None and client._max_batch_size < 128:
        batch_sizes.append(client._max_batch_size * 2)

    results = run_benchmark(client, batch_sizes)

    # Print table
    print(f"{'batch':>5s}  {'jit_ms':>9s}  {'steady_ms':>10s}  {'per_samp_ms':>12s}")
    print("-" * 44)
    for r in results:
        print(f"{r.batch_size:>5d}  {r.jit_ms:>8.0f}  {r.steady_ms:>9.1f}  {r.per_sample_ms:>11.1f}")

    # Speedup summary (batch > 1 only, and 128 if present)
    base = results[0].per_sample_ms
    print(f"\nSpeedup vs batch=1 (per-sample latency):")
    for r in results:
        if r.batch_size == 1:
            print(f"  batch=  1: 1.0x (baseline)")
        else:
            speedup = base / r.per_sample_ms
            print(f"  batch={r.batch_size:>3d}: {speedup:.1f}x")


if __name__ == "__main__":
    main()
