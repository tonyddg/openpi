"""RTC broker with the original BasePolicy.infer(obs) -> one-action interface.

The caller MUST call infer exactly once per successfully executed control tick.
The logical tick increments when an action is returned; real hardware must
halt/reset if it fails to execute that action. No "hold stale action" fallback.
"""

from __future__ import annotations

import copy
from concurrent.futures import Future, ThreadPoolExecutor
import uuid

import numpy as np

from openpi_client import base_policy
from openpi_client.rtc_websocket_policy import RTCWebsocketRPC


class RTCDeadlineError(RuntimeError):
    """Fail closed: the control loop must enter its robot-specific safe state."""


class RTCActionChunkBroker(base_policy.BasePolicy):
    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000,
                 rpc=None, session_id: str | None = None):
        self._rpc = rpc if rpc is not None else RTCWebsocketRPC(host=host, port=port)
        self._metadata = self._rpc.get_server_metadata()
        cfg = self._metadata["rtc"]
        if int(cfg["protocol_version"]) != 1:
            raise ValueError("Unsupported RTC protocol version")
        self.H, self.s = int(cfg["H"]), int(cfg["s"])
        self._session_id = session_id or uuid.uuid4().hex
        self._inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-infer")
        self._control_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-ack")
        self._generation = 0
        self._request_counter = 0
        self._epoch: int | None = None
        self._step = 0
        self._active: np.ndarray | None = None
        self._chunk_id: int | None = None
        self._anchor = 0
        self._pending: Future | None = None
        self._pending_info: tuple[int, int, int, int] | None = None
        self._ack: Future | None = None
        self.reset()  # establish a fresh epoch before first inference

    def get_server_metadata(self) -> dict:
        return self._metadata

    def _request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def reset(self) -> None:
        # This call can proceed while old GPU work is running: RPC uses a new socket.
        # The server increments the epoch; stale workers may finish but cannot commit.
        self._generation += 1
        self._active = None
        self._chunk_id = None
        self._anchor = 0
        self._step = 0
        self._pending = None
        self._pending_info = None
        self._ack = None
        self._epoch = None
        answer = self._rpc.reset(self._session_id, self._request_id())
        if answer.get("op") != "reset_ack" or not answer.get("previous_chunk_discarded"):
            raise RTCDeadlineError(f"Remote reset was not acknowledged: {answer}")
        self._epoch = int(answer["epoch"])

    def _check_chunk(self, reply: dict, *, request_id: int,
                     start_step: int, epoch: int) -> tuple[int, np.ndarray]:
        if (reply.get("op") != "infer_result" or
            reply.get("request_id") != request_id or
            reply.get("epoch") != epoch or
            reply.get("anchor_step") != start_step):
            raise RTCDeadlineError(f"Unexpected/late RTC response: {reply}")
        actions = np.asarray(reply["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self.H or not np.isfinite(actions).all():
            raise RTCDeadlineError("Invalid action array or NaNs in RTC response")
        return int(reply["chunk_id"]), actions.copy()

    def _first_chunk(self, obs: dict) -> None:
        assert self._epoch is not None
        request_id = self._request_id()
        reply = self._rpc.infer(
            session_id=self._session_id, epoch=self._epoch, request_id=request_id,
            start_step=0, chunk_id=None, obs=copy.deepcopy(obs),
        )
        cid, actions = self._check_chunk(reply, request_id=request_id,
                                         start_step=0, epoch=self._epoch)
        # The first action may only be issued after the server confirms this
        # chunk is the active history for the new epoch.
        ack = self._rpc.ack(
            session_id=self._session_id, epoch=self._epoch,
            request_id=self._request_id(), chunk_id=cid, installed_step=0,
        )
        if ack.get("op") != "ack_ok" or ack.get("chunk_id") != cid:
            raise RTCDeadlineError(f"Initial chunk ACK failed: {ack}")
        self._active, self._chunk_id, self._anchor = actions, cid, 0

    def _poll_ack(self):
        if self._ack is not None and self._ack.done():
            reply = self._ack.result()  # propagate failed ACK
            if reply.get("op") != "ack_ok" or reply.get("chunk_id") != self._chunk_id:
                raise RTCDeadlineError(f"ACK did not confirm active chunk: {reply}")
            self._ack = None

    def _poll_inference(self):
        if self._pending is None or not self._pending.done():
            return
        reply = self._pending.result()  # propagate network/server errors
        assert self._pending_info is not None
        gen, epoch, request_id, start_step = self._pending_info
        self._pending = None
        self._pending_info = None
        if gen != self._generation or epoch != self._epoch:
            return  # Reset while inference was in flight: ignore old data.
        cid, actions = self._check_chunk(reply, request_id=request_id,
                                         start_step=start_step, epoch=epoch)
        idx = self._step - start_step
        if not 0 <= idx < self.s:
            raise RTCDeadlineError(
                f"RTC chunk arrived at index {idx}; missed the next fixed request start (s={self.s})"
            )
        self._active, self._chunk_id, self._anchor = actions, cid, start_step
        self._ack = self._control_pool.submit(
            self._rpc.ack, session_id=self._session_id, epoch=epoch,
            request_id=self._request_id(), chunk_id=cid, installed_step=self._step,
        )

    def _at_request_start(self) -> bool:
        return self._active is not None and self._step == self._anchor + self.s

    def infer(self, obs: dict) -> dict:
        """Called at the *start* of one control tick; returns its single action.

        At step anchor+s the observation is captured and inference is queued.
        Python scheduling cannot guarantee nanosecond-precise socket send time.
        """
        if self._epoch is None:
            raise RTCDeadlineError("Remote session not initialized; reset required")
        if self._active is None:
            self._first_chunk(obs)
        else:
            self._poll_ack()
            self._poll_inference()
            if self._at_request_start():
                if self._pending is not None:
                    raise RTCDeadlineError("Previous RPC still pending at next request start")
                if self._ack is not None:
                    raise RTCDeadlineError("Server did not ACK new chunk before next request start")
                assert self._epoch is not None
                req_id = self._request_id()
                self._pending_info = (self._generation, self._epoch, req_id, self._step)
                self._pending = self._inference_pool.submit(
                    self._rpc.infer, session_id=self._session_id,
                    epoch=self._epoch, request_id=req_id, start_step=self._step,
                    chunk_id=self._chunk_id, obs=copy.deepcopy(obs),
                )
        index = self._step - self._anchor
        if self._active is None or not 0 <= index < self.H:
            raise RTCDeadlineError("Active chunk exhausted while new chunk unavailable: STOP robot")
        action = self._active[index].copy()
        self._step += 1
        return {"actions": action, "rtc": {
            "epoch": self._epoch, "chunk_id": self._chunk_id,
            "action_index": index, "logical_step": self._step - 1,
        }}

    def close(self) -> None:
        self._inference_pool.shutdown(wait=False, cancel_futures=True)
        self._control_pool.shutdown(wait=False, cancel_futures=True)
