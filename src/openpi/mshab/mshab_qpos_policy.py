import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

MSHABQposRepackTransform = transforms.Group(
    inputs=[
        transforms.RepackTransform(
            {
                "observation/image": "head_rgb",
                "observation/wrist_image": "gripper_rgb",
                "observation/state": "qpos",
                "observation/state_6d": "pi0_state",

                # 获取夹爪动作
                "actions/ref": "pi0_actions_ref",
                "actions/qpos": "policy_action",
                "prompt": "prompt",
            }
        )
    ]
)

@dataclasses.dataclass(frozen=True)
class MSHABQposInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    # 使用与 6d 输出策略相同的 eef 位姿观测
    use_6d_state: bool = False

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        # Detect batch mode: if state is 2D, image_masks must have a batch dimension.
        
        if self.use_6d_state:
            state = data["observation/state_6d"]
        else:
            # 补充缺少的夹爪动作
            state = np.hstack([data["observation/state"], data["observation/state_6d"][-1:]])

        if state.ndim == 2:
            batch_size = state.shape[0]
            _true = np.ones(batch_size, dtype=bool)
            _false = np.zeros(batch_size, dtype=bool)
        else:
            _true = np.True_
            _false = np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": _true,
                "left_wrist_0_rgb": _true,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": _true if self.model_type == _model.ModelType.PI0_FAST else _false,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        # 补充缺少的夹爪动作
        if ("actions/qpos" in data) and ("actions/ref" in data):
            inputs["actions"] = np.hstack([data["actions/qpos"], data["actions/ref"][:, -1:]])

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class MSHABQposOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"policy_action": np.asarray(data["actions"][..., :11])}