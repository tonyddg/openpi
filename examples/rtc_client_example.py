"""Integration sketch: replace get_observation/execute_action with your robot API.

Every successful call to execute_action must correspond to exactly one infer.
At the first exception, stop at the robot controller's safe state and reset.
"""

from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.rtc_action_chunk_broker import RTCActionChunkBroker
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def create_client(mode: str, host: str, port: int):
    if mode == "rtc":
        return RTCActionChunkBroker(host=host, port=port)  # auto reset + metadata
    if mode == "standard":
        ws = WebsocketClientPolicy(host=host, port=port)
        H = ws.get_server_metadata()["rtc"]["H"]
        return ActionChunkBroker(ws, action_horizon=H)
    raise ValueError(mode)


def run_episode(client, get_observation, execute_action, safe_stop, num_steps: int,
                *, action_key: str = "actions"):
    client.reset()
    try:
        for _ in range(num_steps):
            output = client.infer(get_observation())
            execute_action(output[action_key])
    except Exception:  # include robot execution errors
        safe_stop()
        client.reset()
        raise


# Example, after robot initialization:
# client = create_client("rtc", "127.0.0.1", 8000)
# action_key = "action"  # if configured with --output-action-key action
# run_episode(client, robot.observe, robot.execute_action, robot.stop, 1000, action_key=action_key)
