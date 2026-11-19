"""Thread-safe, per-robot RTC chunk bookkeeping (no model dependencies).

A generated chunk is *pending* until the client explicitly ACKs installation.
A reset bumps the epoch; a GPU task from an old epoch can finish, but its
result cannot be committed. Fixed RTC request points are expressed in logical
control ticks, never in wall-clock/network arrival time.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

import numpy as np


class RTCProtocolError(ValueError):
    """Invalid RTC request or violated timing/session invariant."""


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    H: int = 50
    s: int = 20
    d: int = 5
    control_period_s: float = 0.02
    max_guidance_weight: float = 5.0
    schedule: str = "exp"

    def __post_init__(self) -> None:
        if not (self.H > 2 and 1 <= self.d < self.s and self.d < self.H - self.s):
            raise ValueError("Require H>2, 1<=d<s, d<H-s; choose s to allow latency slack")
        if self.control_period_s <= 0:
            raise ValueError("control_period_s must be positive")
        if self.max_guidance_weight <= 0:
            raise ValueError("max_guidance_weight must be positive")
        if self.schedule not in {"exp", "linear", "hard"}:
            raise ValueError("Unsupported RTC prefix schedule")

    def metadata(self) -> dict[str, Any]:
        return {
            "supported": True,
            "protocol_version": 1,
            "H": self.H,
            "s": self.s,
            "d_estimated": self.d,
            "control_period_s": self.control_period_s,
            "max_inflight_requests": 1,
            "reset_supported": True,
        }


@dataclasses.dataclass
class Chunk:
    chunk_id: int
    anchor_step: int
    actions: np.ndarray


@dataclasses.dataclass
class Session:
    epoch: int = 0
    active: Chunk | None = None
    pending: Chunk | None = None
    inflight: int | None = None
    last_request_id: int = -1
    next_chunk_id: int = 0
    d_ema: float | None = None


@dataclasses.dataclass(frozen=True)
class InferenceTicket:
    session_id: str
    epoch: int
    request_id: int
    anchor_step: int
    previous_chunk: np.ndarray | None
    previous_start: int
    d_estimated: int


class RTCSessionManager:
    """Only this class may modify session state; model inference runs outside the lock."""

    def __init__(self, config: RTCConfig):
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def _session(self, session_id: str) -> Session:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise RTCProtocolError("session_id must be a nonempty string (<=128 chars)")
        return self._sessions.setdefault(session_id, Session())

    def reset(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            session.epoch += 1
            session.active = None
            session.pending = None
            session.inflight = None
            session.last_request_id = -1
            session.next_chunk_id = 0
            session.d_ema = None
            return {"op": "reset_ack", "epoch": session.epoch,
                    "previous_chunk_discarded": True, "status": "ready"}

    def begin(self, *, session_id: str, epoch: int, request_id: int,
              anchor_step: int, chunk_id: int | None) -> InferenceTicket:
        with self._lock:
            session = self._session(session_id)
            if epoch != session.epoch:
                raise RTCProtocolError("Stale epoch: reset/reconnect before further inference")
            if session.inflight is not None:
                raise RTCProtocolError("Only one in-flight inference per session is allowed")
            if session.pending is not None:
                raise RTCProtocolError("Previous generated chunk has not been ACKed")
            if request_id <= session.last_request_id:
                raise RTCProtocolError("request_id must increase within an epoch")
            if session.active is None:
                if anchor_step != 0 or chunk_id is not None:
                    raise RTCProtocolError("First request after reset must be at step 0 with chunk_id=null")
                prev = None
                prev_start = 0
            else:
                if chunk_id != session.active.chunk_id:
                    raise RTCProtocolError("chunk_id does not match the confirmed active chunk")
                required_step = session.active.anchor_step + self.config.s
                if anchor_step != required_step:
                    raise RTCProtocolError(f"RTC request must start at step {required_step}, got {anchor_step}")
                prev = session.active.actions.copy()
                prev_start = self.config.s
            d_est = self.config.d if session.d_ema is None else round(session.d_ema)
            d_est = int(np.clip(d_est, 1, min(self.config.s - 1, self.config.H - self.config.s - 1)))
            session.inflight = request_id
            session.last_request_id = request_id
            return InferenceTicket(session_id, epoch, request_id, anchor_step,
                                   prev, prev_start, d_est)

    def abort(self, ticket: InferenceTicket) -> None:
        with self._lock:
            session = self._session(ticket.session_id)
            if session.epoch == ticket.epoch and session.inflight == ticket.request_id:
                session.inflight = None

    def finish(self, ticket: InferenceTicket, actions: np.ndarray) -> dict[str, Any]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != self.config.H:
            self.abort(ticket)
            raise RTCProtocolError(f"Expected actions [H,D] with H={self.config.H}, got {actions.shape}")
        if not np.isfinite(actions).all():
            self.abort(ticket)
            raise RTCProtocolError("Nonfinite action values from policy")
        with self._lock:
            session = self._session(ticket.session_id)
            if session.epoch != ticket.epoch or session.inflight != ticket.request_id:
                return {"op": "stale", "epoch": ticket.epoch,
                        "request_id": ticket.request_id}
            chunk = Chunk(session.next_chunk_id, ticket.anchor_step, actions.copy())
            session.next_chunk_id += 1
            session.pending = chunk
            session.inflight = None
            return {"op": "infer_result", "epoch": ticket.epoch,
                    "request_id": ticket.request_id, "chunk_id": chunk.chunk_id,
                    "anchor_step": chunk.anchor_step, "actions": chunk.actions.copy()}

    def ack(self, *, session_id: str, epoch: int, chunk_id: int,
            installed_step: int) -> dict[str, Any]:
        with self._lock:
            session = self._session(session_id)
            if session.epoch != epoch:
                raise RTCProtocolError("ACK belongs to an old epoch")
            if session.active is not None and session.active.chunk_id == chunk_id:
                return {"op": "ack_ok", "epoch": epoch, "chunk_id": chunk_id}
            chunk = session.pending
            if chunk is None or chunk.chunk_id != chunk_id:
                raise RTCProtocolError("Unknown or uncommitted chunk_id")
            if not chunk.anchor_step <= installed_step < chunk.anchor_step + self.config.s:
                raise RTCProtocolError("Chunk installation missed its next RTC start; reset required")
            latency = installed_step - chunk.anchor_step
            if chunk.anchor_step != 0:
                # Client-observed end-to-end latency. It informs the NEXT request only.
                session.d_ema = (float(latency) if session.d_ema is None
                                 else 0.8 * session.d_ema + 0.2 * latency)
            session.active = chunk
            session.pending = None
            return {"op": "ack_ok", "epoch": epoch, "chunk_id": chunk_id,
                    "next_request_step": chunk.anchor_step + self.config.s}

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """For tests/diagnostics; not a client protocol method."""
        with self._lock:
            session = self._session(session_id)
            return {"epoch": session.epoch,
                    "active_id": None if session.active is None else session.active.chunk_id,
                    "pending_id": None if session.pending is None else session.pending.chunk_id,
                    "inflight": session.inflight,
                    "d_ema": session.d_ema}
