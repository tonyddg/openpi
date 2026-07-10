
## Issue 收集

https://github.com/Physical-Intelligence/openpi/issues/695

In the Pi-0.5 paper, it says that for action data, the model is trained to predict both joint positions and end-effector poses, and the text prompt includes <control mode> joint/end effector <control mode> to tell them apart. But in the PaligemmaTokenizer, the prompt looks like:
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
How should I add the <control_mode> here?

We only prepend the action format for end-effector control (no prompt for joint position actions) -- when you finetune on one particular action space, the model will quickly pick up on it, so I think it's safe to omit any additional identifier in the prompt.

---

https://github.com/Physical-Intelligence/openpi/discussions/302

That said, here are the nitty-gritty details:

joint position deltas: no change
end-effector control in a fixed global reference frame: "<control_mode> end effector </control_mode> {language_prompt}"
end-effector control in end-effector reference frame (i.e., deltas): "<control_mode> end effector cam frame </control_mode> {language_prompt}"
End-effector poses are encoded using XYZ position and Rot6D rotation (first two columns of the rotation matrix, flattened). The ordering is [x, y, z, *rot6d, gripper], so 9 dimensions total. For biarm setups, this sequence is repeated first for the left arm and then for the right arm. For end-effector deltas, the reference frame is lined up with the wrist camera such that +z is forward, +x is right, and +y is down.

---

https://github.com/Physical-Intelligence/openpi/issues/416

Hi!

I removed the delta actions since a user pointed out that Libero uses delta actions out of the box, so we do not need to apply an additional delta transform within the model itself. I re-trained a model without the delta transform and it performs well in Libero rollouts, so I think this change should be correct.

However, I realized that this change would break the pre-trained Libero checkpoints we provide, so I will revert it until we updated the checkpoints.

---

==Libero 的观测为末端位置 + 末端四元数姿态(转为轴角对) + 夹爪状态==

```py
# examples/libero/main.py:130 
# Libero 的观测为末端位置 + 末端四元数姿态(转为了轴角对) + 夹爪状态
# Finished executing previous action chunk -- compute new chunk
# Prepare observations dict
element = {
    "observation/image": img,
    "observation/wrist_image": wrist_img,
    "observation/state": np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ),
    "prompt": str(task_description),
}
```

gpt:

**如果看 OXE 在 OpenVLA 里的具体落地编码：**
- 最常见的是 `EEF_POS`，定义为 **`EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)`**，也就是 **7 维**。
- 还有一种 `EEF_R6`，定义为 **`EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)`**，也就是 **10 维**。
所以在很多 OXE 相关实现里，你实际看到的“末端执行器位姿动作”常常是
`[dx, dy, dz, droll, dpitch, dyaw, g]`，或者
`[dx, dy, dz, r6_1..r6_6, g]`。

**对 LIBERO 来说：**
- LIBERO 当前环境包装器默认控制器就是 **`OSC_POSE`**。
- robosuite 对 `OSC_POSE` 的定义是：**不含夹爪时动作维度是 6**，而且位置类控制默认把输入当成**相对当前状态的 delta**；对于末端执行器姿态控制，==旋转增量使用 **axis-angle `(ax, ay, az)`** 表示==。也就是说，控制器层面的 EE pose 动作本质上是：
  `Δposition(3) + Δrotation_axis_angle(3)`。

**因此，LIBERO 这类任务里最实用的理解就是：**
- **前 6 维**：末端执行器的 **delta pose**
  - 前 3 维：`dx, dy, dz`
  - 后 3 维：`dax, day, daz`（axis-angle 旋转增量）
- **第 7 维**：夹爪开合标量。

LIBERO 里如果你看 EEF state，它是 world frame 下的 EEF 位姿；而默认 Delta action 是“相对 robot base frame 的 EEF 增量”，并且属于 base/world 这种参考系下的固定系增量，不是末端自身 tool frame 的随动增量。

本体观测：移动底盘坐标系下的末端位置、四元数姿态、夹爪张开程度
动作：移动底盘坐标系下的末端位移变换（位置 + 固定欧拉角）+ 夹爪张开程度

## 代码阅读

### 微调流程

#### 转为 LeRobot 数据集

`examples/libero/convert_libero_data_to_lerobot.py --data_dir <数据集路径>`: Libero 到 LeRobot 数据集转换脚本
- LeRobot 数据集中，数据集的本地文件均位于 `$HF_LEROBOT_HOME` 下，数据集名称为 `repo_id`，为一个 `your_hf_username/<数据集名称>` 的字符串，同时也是 `$HF_LEROBOT_HOME` 下的同名文件夹
- 命令函参数 `data_dir` 用于指定原始 Libero 数据集文件夹

#### 定义数据处理流程

`src/openpi/training/config.py` 数据集预处理、模型输出输出预处理、模型训练配置

数据集预处理方法通过数据配置类 `DataConfig` 的工厂类 `DataConfigFactory` 的基类规定，主要属性有
- `repo_id` 对应 LeRobot 数据集的 `repo_id`（必要参数）
- `base_config` 数据集配置，通过 `DataConfig(prompt_from_task=True)` 创建，参数 `prompt_from_task` 含义为
    - 取 `True` 从 LeRobot 数据集的 `task` 元数据获取任务描述
    - 取 `False` 从 LeRobot 数据集的 `prompt` 属性获取任务描述（或者在预处理流程规定）
- `assets` 保存数据集的标准化统计信息，默认保存在 `$OPENPI_DATA_HOME/{repo_id}` 下

数据配置类 `DataConfig` 记录了数据在不同阶段的处理方法，这些处理方法通过工厂类 `DataConfigFactory` 的虚函数 `create` 具体定义
- 预处理流程通过 `_transforms.Group` 的 `inputs, outputs` 两个以类 `transforms.DataTransformFn` 为元素的列表定义输入侧与输出侧变换
    - `transforms.DataTransformFn` 可以通过继承基类重载 `__call__(data) -> dict` 方法自定义，也可使用 `src/openpi/transforms.py` 中已经定义好的
    - 通过 `push(inputs, outputs)` 方法可以在 `input` 的最后与 `outputs` 的最前处插入新的变换
- OpenPi 的数据处理流程为
    - 模型训练与推理输入：`repack_transforms.inputs` -> `data_transforms.inputs` -> `Normalize` -> `model_transforms.inputs`
    - 模型推理输出：`model_transforms.outputs` -> `Unnormalize` -> `data_transforms.outputs` -> `repack_transforms.outputs`
- `repack_transforms` 将数据集原始键值对映射为便于 `data_transforms` 处理的键值对
    - 通常使用 `_transforms.RepackTransform(dict)`，通过 {新键值:旧键值} 规定整理关系 
    - 对于原始数据集中如 `["a"]["b"]` 的多维字典将被整理为 `["a/b"]` 的一维字典，输出字典必定为一维的
    - OpenPi 会在 `repack_transforms` 前将 LeRobot 中语言指令的 `task` 字段转换为 `prompt` 或用内部 prompt 填充（依据 `DataConfig` 的 `prompt_from_task` 设置），因此建议使用 `prompt` 字段获取语言指令
- `data_transforms` 在标准化前，将输入（图像 / 动作等）转为适合学习的标准语义（在这个阶段，数据依然未 Batch 的 Numpy 形式）
    - 通常需要自定义 `transforms.DataTransformFn` 用于处理输入，输入经 `repack_transforms` 调整后的，以 Numpy 数组为键值，字符串为键名的字典 `data`，输出具有以下字段的字典
        - `image`：机器人图像观测，值为一个子字典，其键为视角名称可以取 `base_0_rgb`（头部）、`left_wrist_0_rgb`（或单臂手上相机）、`right_wrist_0_rgb`；值为 `H, W, C` 格式的 uint8 Numpy 数组图片，缺少的观测使用 `np.zeros_like(...)` 填充（转换函数可参考 `src/openpi/policies/libero_policy.py:_parse_image`），无论图片大小，在实际输入时将被调整为 224 x 224
        - `image_mask`：未使用图像掩膜，结构与 `image` 相同，但是如果图像未使用则对应子字典的键取 `np.False_` 反之取 `np.True_`
        - `state` 本体状态，一维的 Numpy 数组
        - `actions` 示教动作块，二维 `H, D` 的 Numpy 数组（在读取数据集时将自动处理为动作块形式）
        - `prompt` 语言指令，字符串
    - 推理时没有也不需要提供动作，因此需要先判断输入的字典中是否有动作相关的键
    - 由于 pi0 系列默认输出 32 维动作，因此还需要自定义 `transforms.DataTransformFn` 提取前 dim 个维度的动作，通常代码为 `return {"actions": np.asarray(data["actions"][:, :7])}`
- `model_transforms` 将标准化后的数据转为模型所需的张量，通常直接使用 `ModelTransformFactory()(model_config)` 创建
- 通过继承 `DataConfigFactory` 并实例化虚函数 `create`，并在这个函数中返回具体的 `base_config` 确定数据处理流程，对于定义好的三个变换，通过如下方式创建 `base_config`

```py
# 数据类的子类也要修饰成数据类
@dataclasses.dataclass(frozen=True)
class XXXDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        repack_transform = ...
        data_transforms = ...
        model_transforms = ...

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

#### 模型训练配置

`src/openpi/training/config.py:560` 数据集预处理、模型输出输出预处理、模型训练配置

模型训练配置通过类 `TrainConfig` 定义，基本参数有
- `name` 配置名称，该名称将用于模型的加载与训练相关脚本
- `data` 数据处理流程，在此处传入自定义的 `DataConfigFactory` 基类，并通过 `repo_id` 参数规定数据集
- `batch_size, num_train_steps` 批量大小与训练步长

通用训练参数
```py
batch_size=32, # LoRA 微调中，Batch Size 取 32 约用完 80G
# batch_size * step, 不按 epoch
num_train_steps=30_000,
# 用于全量, LoRA 微调使用默认的学习率调度参数 (即不显示传入该参数)
lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=10_000,
    peak_lr=5e-5,
    decay_steps=1_000_000,
    decay_lr=5e-5,
),
optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),

# 保存设置
checkpoint_base_dir="/mnt/dataset/xzmyuq/openpi_ckpts", # 保存路径
save_interval=1000, # 可覆盖保存间隔
keep_period=5000, # 长期保存间隔，LoRA 参数约 9G
```

模型选择参数（全量微调 pi0.5）
```py
model=pi0_config.Pi0Config(
    pi05=True, action_horizon=10, 
    discrete_state_input=False
),
ema_decay=0.999,
weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
```

模型选择参数（LoRA 微调 pi0.5）
```py
model=pi0_config.Pi0Config(
    pi05=True, # 启用 pi0.5
    action_horizon=10,  # 动作块长度
    discrete_state_input=False,
    # LoRA 微调
    # 取 gemma_2b 可以全量训练 VLM 主干
    paligemma_variant="gemma_2b_lora",
    # 取 gemma_300m 可以全量训练 Action Expert
    action_expert_variant="gemma_300m_lora",
),
# freeze_filter 通常要和 model 的 Pi0Config 保持一致, 再调用 get_freeze_filter 方法
freeze_filter=pi0_config.Pi0Config(
    pi05=True, action_horizon=10,
    # 必须保持一致 !!!
    paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
).get_freeze_filter(),
# Turn off EMA for LoRA finetuning.
ema_decay=None,
```

关于 pi0.5 中的 pi0_config.Pi0Config 的 `discrete_state_input` 参数
1. pi0.5 中，本体感知并不进入动作专家，而是合并到 prompt 中
2. 先将 state 每个维度归一化到 [-1, 1] 再依据分位映射为 0~255，得到一串字符串如 119 35 170 201 255
3. 然后与原始 prompt 合并得到实际 prompt 例如：`Task: tidy the house, State: 127 191 25; Action:`
4. `discrete_state_input` 取 False 时则不会有 State 部分, 即本体感知被忽略了

#### 微调脚本

```bash
# 分析数据集的统计量
uv run scripts/compute_norm_stats.py --config-name 配置名称

# 训练模型
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py 配置名称 --exp-name=my_experiment --overwrite
```

### 创建 LeRobot 数据集

`from lerobot.datasets.lerobot_dataset import LeRobotDataset` 导入 LeRobot 数据集模块

LeRobot 数据集中，将数值数据保存为 parquet 格式，图像保存为 mp4 格式的视频

首先需要使用一个配置字典规定数据集中单帧数据的各个字段的名称、形状、维度含义
- 配置字典为一维的，字典的键名即单帧数据中对应的键，需要用如 `a.b` 的方式表示单帧数据中的 `[a][b]`（键名中不可以出现 `/`）
- 字典的值为一个子配置字典，包含以下键
    - `dtype` 字符串数据类型，一般取 `float32`（实数）或 `image`（图片）
    - `shape` 元组，数据形状；`names` 字符串列表，各维含义
- 虽然配置字典没有出现，但 LeRobot 数据集中，每个数据帧必须有 `task` 字段说明任务的语言指令

配置字典示例如下

```py
features = {
    "observation.image.head": {
        "dtype": "image", "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.image.wrist": {
        "dtype": "image", "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (8,), "names": ["state"],
    },
    "actions": {
        "dtype": "float32", "shape": (7,), "names": ["actions"],
    },
},
```

以上配置字典对应的单帧数据为

```py
frame = {
    "observation.image.head": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.image.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
    "observation.state": np.random.randn(8).astype("float32"),
    "actions": np.random.randn(7).astype("float32"),
    "task": "..."
}
```

通过 `LeRobotDataset.create(...)` 方法创建数据集，返回数据集对象，常用参数有
- `repo_id` 数据集名称，为一个 `your_hf_username/<数据集名称>` 的字符串，数据集将保存在 `$HF_LEROBOT_HOME` 下的同名文件夹
- `features` 数据集的配置字典
- `fps, robot_tyoe` 元数据，采样帧率（影响视频形式的图像保存）与机器人类型
- `image_writer_threads` 保存线程数，通常取 `2 / 4 * 相机数`

LeRobot 数据集中数据分为 `frame` 与 `episode` 两个层次，先通过 `add_frame(frame)` 方法按时间顺序直接向数据集插入多个数据帧，然后通过 `save_episode()` 方法将前面所有数据帧保存为一个 Episode，最后通过 `finalize` 方法保存整个数据集。示例代码如下

```py
for episode in range(num_episodes):
    for step in range(num_steps):
        dataset.add_frame({
            "observation.image.head": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation.image.wrist": np.zeros((256, 256, 3), dtype=np.uint8),
            "observation.state": np.random.randn(8).astype("float32"),
            "actions": np.random.randn(7).astype("float32"),
            "task": "..."
        })
    dataset.save_episode()
dataset.finalize()
```

### 推理流程

#### 服务端

通过以下命令加载 checkpoint 并启动 openpi 推理服务端, 通过环境变量 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` 避免占用过多显存

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config 训练配置 --policy.dir 模型文件夹
```

例如 

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_lora_ref_action_trajectories_tidy_house_all_state_rot_6d_action_axis_angle --policy.dir /mnt/dataset/xzmyuq/openpi_ckpts/pi05_lora_ref_action_trajectories_tidy_house_all_state_rot_6d_action_axis_angle/pi05_lora_ref_action_trajectories_tidy_house_all_state_rot_6d_action_axis_angle/29999
```

#### 客户端

通过 `uv add "git+https://github.com/Physical-Intelligence/openpi#subdirectory=packages/openpi-client"` 安装 openpi 客户端子模块

```python
# 导入客户端类
from openpi_client import websocket_client_policy as _websocket_client_policy

# 创建客户端类
_websocket_client_policy.WebsocketClientPolicy(host, port)
```

通过方法 `res = client.infer(element)` 获取动作块
- `element` 单帧观测 (不需要动作), 为以 Numpy 数组为值的字典, 字典格式对应 `transforms.RepackTransform` 输入的字典
- `res` 推理结果字典, 动作块所在的键为 `dataclasses` 的参数 `action_sequence_keys` 指定 (H x A), 其他键值为推理所用时间
