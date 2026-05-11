from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import numpy as np
import logging

from openpi.mshab.utility import ROT_TYPE, STATE_ROT_LENGTH_DICT

def make_mshab_example(
    state_rot_type: ROT_TYPE = "rot_6d",
) -> dict:
    """
    Creates a random input example for mshab
    字典键值规范参见 src/openpi/policies/mshab_policy.py
    """

    state_length = 4 + STATE_ROT_LENGTH_DICT[state_rot_type]
    return {
        "observation/state": np.random.rand(state_length),
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "prompt": "do something",
    }

if __name__ == "__main__":

    host: str = "0.0.0.0"
    port: int = 8000
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    for i in range(5):
        element = make_mshab_example()
        infer_res = client.infer(element)
        print(f"infer res: {infer_res}")
