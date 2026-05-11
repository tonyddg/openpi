from typing import Literal


ROT_TYPE = Literal["axis_angle", "rot_6d", "quat", "rpy_euler"]
STATE_ROT_LENGTH_DICT = {
    "axis_angle": 3,
    "rot_6d": 6,
    "quat": 4,
    "rpy_euler": 3
}