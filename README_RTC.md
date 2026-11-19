# OpenPI π0.5 Stateful RTC — additive patch

**Target:** upstream [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi), **JAX/NNX Pi0.5** and original `policy_config.create_trained_policy()`. This is a source integration patch, **not** a tested binary or pretrained checkpoint. It was prepared against the upstream public interfaces visible in November 2026; no exact upstream commit SHA was checked out in this environment.

## Design

* The original `pi0.py`, `policy.py`, `policy_config.py`, `websocket_client_policy.py`, `ActionChunkBroker`, and `serve_policy.py` are **unchanged**.
* A new RTC server accepts **both** the original plain `infer(obs)` message (standard output dict) and `op=infer/reset/ack` RTC messages.
* Server owns `H` (checkpoint action horizon), `s` (fixed start offset in control ticks), and `d` (estimated delay; updated by EMA from ACK feedback). `s` is **always relative to the predicted chunk's observation anchor**, not response arrival.
* Server keeps physical/robot-space action chunks **per `session_id`**, and makes a result active **only when client ACKs installation**.
* Client returns one action per `infer(obs)` call, starts its next RPC only at `anchor+s`, and **skips the expired part** of the new chunk when it arrives. `reset()` must receive `reset_ack` before another episode.
* The model sampler uses VJP guidance during flow-matching denoising. A previous robot-space chunk is rebased through the **same current input transforms** (including `DeltaActions`, normalization and padding) to the model action space. First inference uses the stock Standard sampler.

### Added files

```text
src/openpi/models/rtc_guidance.py
src/openpi/models/pi0_rtc.py
src/openpi/policies/policy_rtc.py
src/openpi/serving/rtc_session_manager.py
src/openpi/serving/rtc_websocket_server.py
packages/openpi-client/src/openpi_client/rtc_websocket_policy.py
packages/openpi-client/src/openpi_client/rtc_action_chunk_broker.py
scripts/serve_policy_rtc.py
examples/rtc_client_example.py
tests/test_rtc_session_manager.py
tests/test_rtc_broker.py
tests/test_rtc_server_dispatch.py
tests/test_rtc_websocket.py
```

## Installation

From your upstream OpenPI root, **copy** the new `src/`, `packages/`, `scripts/`, `examples/`, `tests/` folders into the same relative locations. This patch doesn't overwrite stock files. Keep OpenPI's `uv` environment and install its regular dependencies. The `openpi-client` package must be installed in the robot environment, including the two additional client modules.

```bash
# From the upstream openpi repository root:
uv run scripts/serve_policy_rtc.py \
  --config pi05_aloha \
  --checkpoint-dir /path/to/your/jax/pi05/checkpoint \
  --host 127.0.0.1 --port 8000 \
  --s 20 --d 5 --num-steps 10 \
  --control-period-s 0.02 --schedule exp \
  --max-guidance-weight 5
```

`H` comes from the checkpoint, not an independent command line value. Default `H=50` examples assume your model config has action horizon 50. The server validates `1 <= d < s` and `d < H-s`, providing slack for late results.

The default is to **warm up both JAX standard and RTC paths before accepting clients**. RTC's VJP backward pass is expensive; you should benchmark this on your GPU. You can skip warmup only for development via `--no-warmup` (check `uv run ... --help` for Tyro options). **PyTorch `model.safetensors` checkpoints are not supported by this patch.**

## Robot-side interface

```python
from openpi_client.rtc_action_chunk_broker import RTCActionChunkBroker

policy = RTCActionChunkBroker(host="127.0.0.1", port=8000)
# Constructor already calls reset() and checks the server ACK.
policy.reset()  # optional episode boundary reset

for tick in range(1000):
    obs = get_observation()           # the same dict used by original OpenPI
    action = policy.infer(obs)["actions"]
    execute_action(action)            # exactly one successful control tick
```

The `get_observation` / `execute_action` functions are robot-specific and not supplied. This example is **not** a standalone robot driver. Control code MUST call `robot.safe_stop()` (appropriate for that specific robot) on a deadline, RPC, or physical execution error; this library intentionally raises rather than silently replaying stale actions. If a physical action failed or a controller skipped a tick, do not use the logical call count as if it were confirmed execution: stop/reconcile/reset the episode.

For Standard inference on the **same RTC-enabled port**:

```python
from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ws = WebsocketClientPolicy(host="127.0.0.1", port=8000)
H = ws.get_server_metadata()["rtc"]["H"]
policy = ActionChunkBroker(ws, action_horizon=H)
# Standard: original infer(obs), no RTC state.
```

For dynamic mode changes: **halt the robot and reset** before constructing/switching a broker. Do not share in-progress action buffers across modes.

## Protocol and exact logical start

Connect as with the upstream server: the first binary WebSocket frame is the server's msgpack-numpy `metadata`. RTC extension:

```python
# first invocation after reset (synchronous); initial anchor = 0
{"op": "infer", "session_id": "robot1", "epoch": 1,
 "request_id": 1, "chunk_id": None, "start_step": 0, "obs": obs0}

# response (actions: NumPy [H, action_dim])
{"op": "infer_result", "epoch": 1, "request_id": 1,
 "chunk_id": 0, "anchor_step": 0, "actions": actions0}

# must ACK the chunk installation before the next guided inference
{"op": "ack", "session_id": "robot1", "epoch": 1, "request_id": 2,
 "chunk_id": 0, "installed_step": 0}

# at logical control step 20 (s = 20)
{"op": "infer", "session_id": "robot1", "epoch": 1,
 "request_id": 3, "chunk_id": 0, "start_step": 20, "obs": obs20}

# if response arrives at step 25, play actions1[5] (NOT actions1[0]);
# then ACK {"chunk_id": 1, "installed_step": 25}.
# the following request is at step 40, i.e. anchor 20 + s, NOT step 45.
```

Reset is *explicit*:

```python
# Request:
{"op": "reset", "session_id": "robot1", "request_id": 100}
# Response:
{"op": "reset_ack", "session_id": "robot1", "request_id": 100,
 "epoch": 2, "previous_chunk_discarded": True, "status": "ready"}
```

Reset increments `epoch`, clears active and pending chunks, invalidates in-flight tickets. **It does not abort an already-running JAX GPU kernel**, nor does it physically stop the robot; the caller is responsible for robot-level safe stop. The result of any old inference is rejected by the server using `epoch`.

### Why `ack` matters

Until the client confirms installing a chunk, the server keeps it in a `pending` slot and does not use it as a reference on the next guided request. A late result that the robot never used must not become `previous_chunk`.

### Source timing semantics

`H=50, s=20, d=5`: at logical step 20 the client captures the observation and **queues** the request. The robot can keep executing old actions. If new output arrives at step 25, new action index = `25 - 20 = 5`. The server's constraint weights are 1.0 for the first `d` predicted ticks and taper over the previous chunk's valid overlap (`H-s`). The server estimates future `d` via client-observed installation offsets (EMA), but the **actual** `d` can differ.

Python threads and network scheduling **cannot guarantee an on-the-wire packet at an exact real-time deadline**. This implementation guarantees the observation/decision is associated with logical `anchor+s`, validates that offset server-side, and rejects excessively late switches; use a real-time executor if you require hard control deadlines.

## Validation

Tests for the session state machine, broker, async reset and live WebSocket protocol (fake policies/actions, no Pi0.5 weights):

```bash
uv run pytest -q tests/test_rtc_*.py
tests/test_rtc_server_dispatch.py
tests/test_rtc_websocket.py
```

The source patch was syntax-checked and these mock tests were run. **It has not yet been tested with actual Pi0.5 weights, JAX/NNX inference on GPU, image/state transforms for your exact robot, or a real robot.** Before real-hardware use, benchmark both cold and warm RTC latency and verify: action-space conversions, chunk-boundary joint errors, tail overlap, no empty action queue, and reset when a GPU request remains in flight.

## Operational limitations

1. **JAX/NNX only**; no torch trained-RTC or torch guided sampler.
2. The server has a **global model inference lock** because a single Policy RNG is shared. Concurrent sessions are isolated, but inference calls serialize; peak load can cause missed deadlines.
3. The RTC RPC client uses a **new WebSocket connection per request**, keeping reset independent of in-flight inference. This is simple/correct but adds handshake overhead; optimize to two persistent sockets and a request-ID dispatcher for production.
4. Sessions are held in process memory, with no persistence or TTL eviction; use a single trusted server, bounded number of sessions, authentication/TLS for remote access, and a session expiry strategy for deployment.
5. Client logical `step` counts returned actions, **not confirmed hardware execution**. For hardware that drops/defers commands, integrate a real `executed_step` confirmation with the robot controller or stop/reset on inconsistency.
6. A custom `RepackTransform` can drop the injected `actions` field. In that case, the `RTCPolicy` **raises** rather than silently applying guidance in an incorrect action space; adapt your robot-specific repack mapping.
7. If GPU response arrives at or after the following RTC request start (`anchor+s`), the client fails closed. `s/d` must be tuned to real deployment latency.

## Source references

* Upstream OpenPI `Pi0.sample_actions`, `Policy.infer`, model transforms, WebSocket server, and client source.
* Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow Policies* (2025).
* An independent community example of guided Pi0.5 VJP inference: `saifahmadgit/openpi_franka` (`pi0_rtc.py`).
