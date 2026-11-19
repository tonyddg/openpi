"""RTC RPC over stock msgpack-numpy WebSockets; no stock client changes.

Each RPC uses a short-lived connection. It allows reset on a different socket
while another socket waits for GPU inference. Optimizing to two persistent
connections is possible after the correctness path is validated.
"""

from __future__ import annotations

import websockets.sync.client

from openpi_client import msgpack_numpy


class RTCRemoteError(RuntimeError):
    pass


class RTCWebsocketRPC:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, *,
                 api_key: str | None = None, timeout_s: float = 30.0):
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._metadata = self._call(None)
        if not self._metadata.get("rtc", {}).get("supported", False):
            raise RTCRemoteError("Connected server does not advertise RTC support")

    def get_server_metadata(self) -> dict:
        return self._metadata

    def _call(self, payload: dict | None):
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        with websockets.sync.client.connect(
            self.uri, compression=None, max_size=None,
            open_timeout=self.timeout_s, additional_headers=headers,
        ) as ws:
            metadata = self._read(ws)
            if payload is None:
                return metadata
            ws.send(msgpack_numpy.Packer().pack(payload))
            result = self._read(ws)
        if not isinstance(result, dict):
            raise RTCRemoteError(f"Unexpected response: {type(result)!r}")
        if result.get("op") in ("error", "stale"):
            raise RTCRemoteError(result.get("message", str(result)))
        return result

    def _read(self, ws):
        packet = ws.recv(timeout=self.timeout_s)
        if isinstance(packet, str):
            raise RTCRemoteError(packet)
        return msgpack_numpy.unpackb(packet)

    def reset(self, session_id: str, request_id: int) -> dict:
        return self._call({"op": "reset", "session_id": session_id, "request_id": request_id})

    def infer(self, *, session_id: str, epoch: int, request_id: int,
              start_step: int, chunk_id: int | None, obs: dict) -> dict:
        return self._call({"op": "infer", "session_id": session_id, "epoch": epoch,
                           "request_id": request_id, "start_step": start_step,
                           "chunk_id": chunk_id, "obs": obs})

    def ack(self, *, session_id: str, epoch: int, request_id: int,
            chunk_id: int, installed_step: int) -> dict:
        return self._call({"op": "ack", "session_id": session_id, "epoch": epoch,
                           "request_id": request_id, "chunk_id": chunk_id,
                           "installed_step": installed_step})
