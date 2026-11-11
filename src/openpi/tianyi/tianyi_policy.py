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

@dataclasses.dataclass
class ImageProcessConfig():
    # 是否裁剪图像, 取 False 时配置中相关参数无效
    is_clip_image: bool = False
    # 裁剪区左上 X 坐标
    clip_lx: int = 0
    # 裁剪区左上 Y 坐标
    clip_ly: int = 0
    # 裁剪区右下 X 坐标
    clip_rx: int = 0
    # 裁剪区右下 Y 坐标
    clip_ry: int = 0

    def __post_init__(self):
        if self.is_clip_image is not None:
            if self.clip_rx <= self.clip_lx:
                raise ValueError(f"clip_lx is greater than clip_rx")
            if self.clip_ry <= self.clip_ly:
                raise ValueError(f"clip_ly is greater than clip_ry")

    def process(self, image):
        image = _parse_image(image) # 输入 c h w 输出 h w c 
        return image[self.clip_ly : self.clip_ry, self.clip_lx : self.clip_rx, :]

TianyiRepackTransform = transforms.Group(
    inputs=[
        transforms.RepackTransform(
            {
                "observation/image": "head_rgb",
                "observation/wrist_image": "gripper_rgb",
                "observation/state": "last_pi0_actions_ref",
                "actions/ref": "pi0_actions_ref",
                "prompt": "prompt",
            }
        )
    ]
)

@dataclasses.dataclass(frozen=True)
class TianyiInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    head_image_config: ImageProcessConfig = dataclasses.field(
        default_factory = lambda : ImageProcessConfig(is_clip_image = False,))
    gripper_image_config: ImageProcessConfig = dataclasses.field(
        default_factory = lambda : ImageProcessConfig(is_clip_image = False,))

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
        head_image = self.head_image_config.process(data["observation/image"])
        gripper_image = self.gripper_image_config.process(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        # Detect batch mode: if state is 2D, image_masks must have a batch dimension.
        state = data["observation/state"]
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
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": gripper_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(head_image),
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
        if "actions/ref" in data:
            inputs["actions"] = data["actions/ref"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class TianyiOutputs(transforms.DataTransformFn):
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
        return {"pi0_actions_ref": np.asarray(data["actions"][..., :7])}
