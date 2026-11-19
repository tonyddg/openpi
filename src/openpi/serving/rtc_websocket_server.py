"""Additive WebSocket server for stateless legacy inference + stateful RTC RPC.

Each RTC request may use a separate connection. While one connection awaits a
long GPU inference, an independent connection can process reset immediately.
The epoch guard discards late GPU results. GPU work itself is not preempted.
"""

from __future__ import annotations

import asyncio
import http
import logging
import threading
import time

import websockets
import websockets.asyncio.server as ws_server

from openpi_client import msgpack_numpy
from openpi.serving.rtc_session_manager import RTCConfig, RTCProtocolError, RTCSessionManager

log = logging.getLogger(__name__)


class RTCWebsocketPolicyServer:
    def __init__(self, base_policy, rtc_policy, *, config: RTCConfig,
                 host: str = "127.0.0.1", port: int = 8000):
        if int(rtc_policy.action_horizon) != config.H:
            raise ValueError("RTCConfig.H must match the checkpoint's action_horizon")
        self.base_policy = base_policy
        self.rtc_policy = rtc_policy
        self.manager = RTCSessionManager(config)
        self.config = config
        self.host = host
        self.port = port
        self._model_lock = threading.Lock()  # protect shared RNG/model invocation
        self.metadata = {**base_policy.metadata, "rtc": config.metadata()}

    def _run_infer(self, ticket, obs):
        with self._model_lock:
            if ticket.previous_chunk is None:
                return self.base_policy.infer(obs)
            return self.rtc_policy.infer(
                obs, previous_chunk=ticket.previous_chunk,
                previous_start=ticket.previous_start,
                inference_delay=ticket.d_estimated,
            )

    def _run_legacy(self, obs):
        with self._model_lock:
            return self.base_policy.infer(obs)

    async def _dispatch(self, request):
        if not isinstance(request, dict):
            raise RTCProtocolError("Expected a msgpack dictionary")
        op = request.get("op")
        if op is None:
            # Exact stock WebsocketClientPolicy compatibility: input and output
            # remain plain observation / result dictionaries.
            return await asyncio.to_thread(self._run_legacy, request)
        session_id = request.get("session_id")
        if op == "reset":
            result = self.manager.reset(session_id)
            return {**result, "session_id": session_id,
                    "request_id": request.get("request_id")}
        if op == "ack":
            result = self.manager.ack(
                session_id=session_id, epoch=request["epoch"],
                chunk_id=request["chunk_id"], installed_step=request["installed_step"],
            )
            return {**result, "session_id": session_id,
                    "request_id": request.get("request_id")}
        if op != "infer":
            raise RTCProtocolError(f"Unsupported op: {op}")
        ticket = self.manager.begin(
            session_id=session_id, epoch=request["epoch"],
            request_id=request["request_id"], anchor_step=request["start_step"],
            chunk_id=request.get("chunk_id"),
        )
        start = time.monotonic()
        try:
            output = await asyncio.to_thread(self._run_infer, ticket, request["obs"])
            reply = self.manager.finish(ticket, output["actions"])
        except Exception:
            self.manager.abort(ticket)
            raise
        if reply["op"] == "infer_result":
            reply["model_timing"] = output.get("policy_timing", {})
            reply["rtc_active"] = bool(ticket.previous_chunk is not None)
            reply["server_elapsed_ms"] = (time.monotonic() - start) * 1000
        return {**reply, "session_id": session_id}

    async def _handler(self, ws: ws_server.ServerConnection):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self.metadata))  # same handshake as upstream
        try:
            async for raw in ws:
                request = None
                try:
                    request = msgpack_numpy.unpackb(raw)
                    answer = await self._dispatch(request)
                except (RTCProtocolError, ValueError, KeyError, TypeError) as exc:
                    answer = {"op": "error", "message": str(exc),
                              "request_id": request.get("request_id") if isinstance(request, dict) else None}
                except Exception:
                    log.exception("RTC inference failed")
                    answer = {"op": "error", "message": "Internal inference error; see server log"}
                await ws.send(packer.pack(answer))
        except websockets.ConnectionClosed:
            pass

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with ws_server.serve(self._handler, self.host, self.port,
                                   compression=None, max_size=None,
                                   process_request=self._health_check) as server:
            log.info("RTC server listening on %s:%s", self.host, self.port)
            await server.serve_forever()

    @staticmethod
    def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None
