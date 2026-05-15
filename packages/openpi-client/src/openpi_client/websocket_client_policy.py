import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


def _infer_batch_size(obs: Dict) -> int:  # noqa: UP006
    """Infer the batch size from a batched observation dict."""
    for val in obs.values():
        if isinstance(val, dict):
            return _infer_batch_size(val)
        if isinstance(val, list):
            return len(val)
        if isinstance(val, np.ndarray) and val.ndim > 0:
            return val.shape[0]
    return 1


def _stack_dicts(samples: list[Dict]) -> Dict:  # noqa: UP006
    """Stack a list of per-sample dicts into a batched dict."""
    if not samples:
        return {}
    result = {}
    for key in samples[0]:
        val = samples[0][key]
        if isinstance(val, dict):
            result[key] = _stack_dicts([s[key] for s in samples])
        elif isinstance(val, str):
            result[key] = [s[key] for s in samples]
        else:
            result[key] = np.stack([np.asarray(s[key]) for s in samples], axis=0)
    return result


def _split_dict(data: Dict, batch_size: int) -> List[Dict]:  # noqa: UP006
    """Split a batched dict along axis 0 into a list of per-sample dicts."""
    if batch_size <= 0:
        return []
    results: List[Dict] = [{} for _ in range(batch_size)]
    for key, val in data.items():
        if isinstance(val, dict):
            sub_results = _split_dict(val, batch_size)
            for i in range(batch_size):
                results[i][key] = sub_results[i]
        elif isinstance(val, list):
            for i in range(batch_size):
                results[i][key] = val[i]
        elif isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == batch_size:
            for i in range(batch_size):
                results[i][key] = val[i]
        else:
            for i in range(batch_size):
                results[i][key] = val
    return results


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
        self._max_batch_size: Optional[int] = self._server_metadata.get("max_batch_size")

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                    # 避免尝试走代理导致的错误
                    proxy=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:  # noqa: UP006
        """Infer actions for a list of observations (BasePolicy interface).

        Delegates to infer_batched() for efficient batch transport.
        """
        max_bs = self._max_batch_size
        if max_bs is None:
            return [self.infer(obs) for obs in obs_list]

        results: List[Dict] = []
        for i in range(0, len(obs_list), max_bs):
            chunk = obs_list[i : i + max_bs]
            batched = self.infer_batched(_stack_dicts(chunk))
            results.extend(_split_dict(batched, len(chunk)))
        return results

    def infer_batched(self, obs: Dict) -> Dict:  # noqa: UP006
        """Infer actions for a batched observation dict.

        Input:  {"state": (N, s), "image": {(N, h, w, 3)}, "prompt": ["t1", ...]}
        Output: {"actions": (N, H, A), "policy_timing": {...}, "server_timing": {...}}

        Automatically splits batches larger than max_batch_size.
        """
        max_bs = self._max_batch_size
        if max_bs is None:
            # Fallback: split to individual observations, infer, re-stack.
            batch_size = _infer_batch_size(obs)
            results = [self.infer(o) for o in _split_dict(obs, batch_size)]
            return _stack_dicts(results)

        batch_size = _infer_batch_size(obs)
        all_chunks: List[Dict] = []
        for i in range(0, batch_size, max_bs):
            chunk_size = min(max_bs, batch_size - i)
            chunk_list = _split_dict(obs, batch_size)[i : i + chunk_size]
            chunk_batch = _stack_dicts(chunk_list)
            chunk_batch["_batch_size"] = chunk_size
            self._ws.send(self._packer.pack(chunk_batch))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            result = msgpack_numpy.unpackb(response)
            result.pop("_batch_size", None)
            all_chunks.append(result)

        if len(all_chunks) == 1:
            return all_chunks[0]
        return _stack_dicts(
            [s for chunk in all_chunks for s in _split_dict(chunk, _infer_batch_size(chunk))]
        )

    @override
    def reset(self) -> None:
        pass
