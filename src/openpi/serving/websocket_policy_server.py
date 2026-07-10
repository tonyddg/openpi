import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from openpi import transforms as _transforms

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._max_batch_size = max_batch_size
        if max_batch_size is not None:
            policy._max_batch_size = max_batch_size
        self._metadata = dict(metadata or {})
        if max_batch_size is not None:
            self._metadata["max_batch_size"] = max_batch_size
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        loop = asyncio.get_event_loop()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                batch_size = obs.pop("_batch_size", None) if isinstance(obs, dict) else None

                infer_time = time.monotonic()

                if batch_size is not None:
                    # --- Batch mode ---
                    chunk_count = 0
                    chunk_size = self._max_batch_size or batch_size
                    all_actions = []
                    for i in range(0, batch_size, chunk_size):
                        end = min(i + chunk_size, batch_size)
                        chunk = _transforms.split_dict(obs, batch_size)[i:end]
                        actions = await loop.run_in_executor(None, self._policy.infer_batch, chunk)
                        all_actions.extend(actions)
                        chunk_count += 1

                    infer_time = time.monotonic() - infer_time
                    logger.info("batch=%d, chunks=%d, infer=%.0fms", batch_size, chunk_count, infer_time * 1000)

                    for action in all_actions:
                        action["server_timing"] = {
                            "infer_ms": infer_time * 1000,
                        }
                        if prev_total_time is not None:
                            action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    batched_result = _transforms.stack_dicts(all_actions)
                    batched_result["_batch_size"] = batch_size
                    await websocket.send(packer.pack(batched_result))
                else:
                    # --- Single mode ---
                    action = await loop.run_in_executor(None, self._policy.infer, obs)
                    infer_time = time.monotonic() - infer_time
                    logger.info("batch=1, infer=%.0fms", infer_time * 1000)

                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))

                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
