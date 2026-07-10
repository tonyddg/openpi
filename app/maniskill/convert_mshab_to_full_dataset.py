from pathlib import Path
import random
import shutil
from typing import List, Literal, Optional, Tuple
import tyro
import h5py
import json
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.mshab.scan_mshab_trajectories import collect_trajectory_records, sample_language_instruction, TrajectoryRecord, extract_furniture_name

# ARMMOVE 的动作配置字典
FETCH_ACT_DICT = dict(
    # 将 [7 自由度关节, 腰部, 底盘] 的动作映射为原始运动空间
    fullact_map = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12],
    # 读取 [7 自由度关节, 腰部] 的关节状态
    qpos_map = [5, 7, 8, 9, 10, 11, 12, 3],
    # 读取 [7 自由度关节, 腰部] 的关节速度 (基于底盘维度 0-3 被裁剪的 MSHAB 原始关节速度)
    qvel_map = [2, 4, 5, 6, 7, 8, 9, 0],
)

def random_split_indices(
    A: int,
    B: int,
    seed: Optional[int] = None,
    shuffle_remaining: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    从 [0, 1, ..., A-1] 中随机选择 B 个不重复的数。

    返回:
        selected: 随机选中的 B 个数，可作为测试集 indices
        remaining: 剩余的 A-B 个数，可作为训练集 indices

    参数:
        A: 总数量
        B: 要随机选择的数量
        seed: 随机种子，用于复现实验
        shuffle_remaining: 是否打乱 remaining
    """

    if A < 0:
        raise ValueError("A must be non-negative")

    if B < 0 or B > A:
        raise ValueError("B must satisfy 0 <= B <= A")

    rng = random.Random(seed)

    all_indices = list(range(A))

    selected = rng.sample(all_indices, B)
    selected_set = set(selected)

    remaining = [idx for idx in all_indices if idx not in selected_set]

    if shuffle_remaining:
        rng.shuffle(remaining)

    return selected, remaining

# 用非升序的索引元组从 h5py 切片
def h5_take(dset, row_idx, col_idx):
    col_idx = np.asarray(col_idx)

    unique_idx, inverse = np.unique(col_idx, return_inverse=True)
    values = np.asarray(dset[row_idx, unique_idx])

    return values[inverse]

global_episode_id = 0
def append_mshab_trajectory_to_dataset(
    record: TrajectoryRecord,

    train_lerobot_dataset: LeRobotDataset,
    val_lerobot_dataset: LeRobotDataset,

    train_detach_dataset: Optional[h5py.Group] = None,
    val_detach_dataset: Optional[h5py.Group] = None,

    val_set_ratio: float = 0.05,
    seed: int = 42
):
    episode_total_length = 0
    episode_valid_nums = 0 

    print(f"resolve record: {record}")

    assert record.json_path is not None
    assert record.h5_path is not None

    with open(record.json_path, 'r') as f:
        episode_info_list = list(json.load(f)["episodes"])
    num_episode_info = len(episode_info_list)
    _, val_eps_idx = random_split_indices(
        num_episode_info, int(num_episode_info * (1 - val_set_ratio)), seed
    )
    val_eps_idx = set(val_eps_idx)

    with h5py.File(record.h5_path, 'r') as h5_file:
        # 记录全局 episode 信息
        global global_episode_id

        for local_episode_idx, episode_info in enumerate(episode_info_list):
            if local_episode_idx in val_eps_idx:
                dataset = val_lerobot_dataset
                detach_dataset = val_detach_dataset
            else:
                dataset = train_lerobot_dataset
                detach_dataset = train_detach_dataset

            episode_id = int(episode_info["episode_id"])
            
            episode_group = h5_file[f"traj_{episode_id}"]
            assert isinstance(episode_group, h5py.Group)

            # 判断轨迹是否可用
            episode_step_success = np.asarray(episode_group["success"][:]) # type: ignore
            episode_step_fail = np.asarray(episode_group["fail"][:]) # type: ignore
            # 获取第一次成功或失败的步数
            where_success_step = np.where(episode_step_success)[0]
            success_step = int(where_success_step[0]) if where_success_step.size > 0 else None
            where_fail_step = np.where(episode_step_fail)[0]
            fail_step = int(where_fail_step[0]) if where_fail_step.size > 0 else None
            if success_step is None:
                print(f"episode_id: {episode_id} is invalid: no success")
                continue
            if fail_step is not None and fail_step <= success_step:
                print(f"episode_id: {episode_id} is invalid: fail before success")
                continue

            episode_valid_nums += 1
            episode_total_length += success_step
            print(f"episode_id: {episode_id} has length {success_step}")

            # 依据模板生成任务指令
            episode_recption = episode_group["info/target_receptacles"][0] # type: ignore
            target_recption = extract_furniture_name(episode_recption)
            print(f"episode_recption: {episode_recption}, target_recption: {target_recption}")

            task = sample_language_instruction(record, target = target_recption)

            for frame_idx in range(success_step):

                if frame_idx == 0:
                    last_action = np.zeros(len(FETCH_ACT_DICT["fullact_map"]), dtype=np.float32)
                else:
                    last_action = np.asarray(
                        episode_group["actions"][frame_idx - 1, FETCH_ACT_DICT["fullact_map"]], # type: ignore
                        dtype=np.float32,
                    )

                policy_action = np.asarray(episode_group["actions"][frame_idx, FETCH_ACT_DICT["fullact_map"]]) # type: ignore
                # MSHAB 记录的动作讲直接保存模型输出, 需要压缩到有效范围 [-1, 1] 内
                policy_action_clip = np.clip(policy_action, -1, 1)

                pi0_actions_ref_clip = np.array(episode_group["info/pi0_eef_ref_action"][frame_idx + 1, :]) # type: ignore
                # MSHAB 记录的动作讲直接保存模型输出, 需要压缩到有效范围 [-1, 1] 内
                pi0_actions_ref_clip[-1] = np.clip(pi0_actions_ref_clip[-1], -1, 1)

                pi0_actions_abs_clip = np.array(episode_group["info/pi0_eef_abs_action"][frame_idx + 1, :]) # type: ignore
                # MSHAB 记录的动作讲直接保存模型输出, 需要压缩到有效范围 [-1, 1] 内
                pi0_actions_abs_clip[-1] = np.clip(pi0_actions_abs_clip[-1], -1, 1)

                dataset.add_frame(
                    {
                        # 视觉观测: 长 201, 为 t0 到 t200 (作完最后一步动作再记录一张图片)
                        "gripper_rgb": np.asarray(episode_group["obs/sensor_data/fetch_hand/rgb"][frame_idx, :, :, :]), # type: ignore
                        "head_rgb": np.asarray(episode_group["obs/sensor_data/fetch_head/rgb"][frame_idx, :, :, :]), # type: ignore
                        # "gripper_depth": np.asarray(episode_group["obs/sensor_data/fetch_hand/depth"][frame_idx, :, :, :]), # type: ignore
                        # "head_depth": np.asarray(episode_group["obs/sensor_data/fetch_head/depth"][frame_idx, :, :, :]), # type: ignore
                        # 相机位姿, 用于生成像点云: 长 201, 为 t0 到 t200
                        "gripper_camera_t": np.asarray(episode_group["info/gripper_camera_t"][frame_idx, :, :]), # type: ignore
                        "head_camera_t": np.asarray(episode_group["info/head_camera_t"][frame_idx, :, :]), # type: ignore
                        # 本体观测: 长 201, 为 t0 到 t200
                        "qpos": h5_take(episode_group["info/qpos"], frame_idx, FETCH_ACT_DICT["qpos_map"]), # type: ignore
                        "qvel": h5_take(episode_group["obs/agent/qvel"], frame_idx, FETCH_ACT_DICT["qvel_map"]), # type: ignore
                        "last_action": last_action, # type: ignore

                        # state 长 201, 但索引为 0 的状态无效 (执行动作前的 evaluate), 为 t0 到 t199 
                        "pi0_state": np.asarray(episode_group["info/pi0_eef_state"][frame_idx + 1, :]), # type: ignore
                        # action 长 201, 但索引为 0 的动作无效 (reset 前后的 eff 差), 为 t0 到 t199 
                        "pi0_actions_ref": pi0_actions_ref_clip,
                        "pi0_actions_abs": pi0_actions_abs_clip,
                        
                        # 拟合动作
                        "policy_action": policy_action_clip,

                        # 元信息
                        "frame_idx": np.array([frame_idx], dtype = np.uint32),
                        "episode_idx": np.array([global_episode_id], dtype = np.uint32), # 使用全局 episode 编号

                        "task": task,
                    }
                )
                
            if detach_dataset is not None:
                detach_episode_group = detach_dataset.create_group(
                    f"episode_{global_episode_id:06d}",
                )
                
                gripper_depth = np.asarray(episode_group["obs/sensor_data/fetch_hand/depth"][:success_step, :, :, :], dtype = np.uint16) # type: ignore
                detach_episode_group.create_dataset(
                    "gripper_depth", shape = gripper_depth.shape, dtype = np.uint16, data = gripper_depth,
                    # 每次通常读一张深度图
                    chunks = (1, 128, 128, 1),          
                    compression = "gzip",
                    compression_opts = 1,
                    shuffle = True,
                )
                
                head_depth = np.asarray(episode_group["obs/sensor_data/fetch_head/depth"][:success_step, :, :, :], dtype = np.uint16) # type: ignore
                detach_episode_group.create_dataset(
                    "head_depth", shape = head_depth.shape, dtype = np.uint16, data = head_depth,
                    # 每次通常读一张深度图
                    chunks = (1, 128, 128, 1),          
                    compression = "gzip",
                    compression_opts = 1,
                    shuffle = True,
                )
                
            dataset.save_episode()
            global_episode_id += 1


    print(f"episode total nums: {num_episode_info}, episode valid nums: {episode_valid_nums}, episode mean length: {episode_total_length / episode_valid_nums}")

from openpi.mshab.utility import ROT_TYPE, STATE_ROT_LENGTH_DICT

# uv run app/maniskill/convert_mshab_to_full_dataset.py 
# --mshab_trajectroy_root /mnt/dataset/xzmyuq/dataset/mshab_custom_trajectory/trajectories_tidy_house_all_obs_rot6d_act_axs 
# --repo_name trajectories_tidy_house_all 
# --no-is_name_by_filter --state_rot_type rot_6d --action_rot_type axis_angle

def main(
    # 数据集根目录
    mshab_trajectroy_root: str,
    # 数据集名称
    repo_name: str = "your_hf_username/mshab",
    *, 

    # 依据 filter 补充数据集名称
    is_name_by_filter: bool = True,
    task_filter: Optional[list[str]] = None,
    subtask_filter: Optional[list[str]] = None,
    obj_name_filter: Optional[list[str]] = None,

    # 依据 rot type 补充数据集名称
    is_name_by_rot_type: bool = True,
    pi0_state_rot_type: ROT_TYPE = "axis_angle",
    pi0_action_rot_type: ROT_TYPE = "axis_angle",

    # 验证集比例
    val_set_ratio: float = 0.05,
    seed: int = 42,
):
    
    state_length = 4 + STATE_ROT_LENGTH_DICT[pi0_state_rot_type]
    action_length = 4 + STATE_ROT_LENGTH_DICT[pi0_action_rot_type]

    records = collect_trajectory_records(
        dataset_root = Path(mshab_trajectroy_root),
        task_filter = task_filter,
        subtask_filter = subtask_filter,
        obj_name_filter = obj_name_filter,
    )
    if len(records) == 0:
        raise RuntimeError(f"No records find in {mshab_trajectroy_root}")

    if is_name_by_filter:
        if task_filter is not None:
            for task in task_filter:
                repo_name += f"_{task}"
        else:
            repo_name += "_all"

        if subtask_filter is not None:
            for subtask in subtask_filter:
                repo_name += f"_{subtask}"
        else:
            repo_name += "_all"

        if obj_name_filter is not None:
            for obj_name in obj_name_filter:
                repo_name += f"_{obj_name}"
        else:
            repo_name += "_all"

    if is_name_by_rot_type:
        repo_name += f"_state_{pi0_state_rot_type}"
        repo_name += f"_action_{pi0_action_rot_type}"

    # 数据集设置
    FEATURES = {
        # 视觉观测
        "gripper_rgb": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "head_rgb": {
            "dtype": "image",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        # LeRobot 目前无法处理深度图
        # "gripper_depth": {
        #     "dtype": "int16",
        #     "shape": (128, 128, 1),
        #     "names": ["height", "width", "channel"],
        # },
        # "head_depth": {
        #     "dtype": "int16",
        #     "shape": (128, 128, 1),
        #     "names": ["height", "width", "channel"],
        # },
        # 用于生成像点云
        "gripper_camera_t": {
            "dtype": "float32",
            "shape": (4, 4),
            "names": ["row", "column"],
        },
        "head_camera_t": {
            "dtype": "float32",
            "shape": (4, 4),
            "names": ["row", "column"],
        },

        # 本体观测
        "qpos": {
            "dtype": "float32",
            "shape": (len(FETCH_ACT_DICT["qpos_map"]), ),
            "names": ["qpos"],
        },
        "qvel": {
            "dtype": "float32",
            "shape": (len(FETCH_ACT_DICT["qvel_map"]), ),
            "names": ["qvel"],
        },
        "last_action": {
            "dtype": "float32",
            "shape": (len(FETCH_ACT_DICT["fullact_map"]), ),
            "names": ["last_action"],
        },

        # 给定目标 (保持 gripper 使数据集能同时用于 pi0 训练)
        "pi0_actions_ref": {
            "dtype": "float32",
            "shape": (action_length, ),
            "names": [f"actions_ref with {pi0_action_rot_type} rot"],
        },
        "pi0_actions_abs": {
            "dtype": "float32",
            "shape": (action_length, ),
            "names": [f"actions_abs with {pi0_action_rot_type} rot"],
        },

        # 拟合动作
        "policy_action": {
            "dtype": "float32",
            "shape": (len(FETCH_ACT_DICT["fullact_map"]), ),
            "names": ["policy_action"],
        },

        # 保留项 pi0 本体感知 (使数据集能同时用于 pi0 训练)
        "pi0_state": {
            "dtype": "float32",
            "shape": (state_length, ),
            "names": [f"state with {pi0_state_rot_type} rot"],
        },

        # 用于定位深度图位置的元信息
        "episode_idx": {
            "dtype": "uint32",
            "shape": (1, ),
            "names": ["episode idx meta"],
        },
        "frame_idx": {
            "dtype": "uint32",
            "shape": (1, ),
            "names": ["frame idx meta"],
        }
    }

    def make_dataset(repo_name, suffix):
        use_repo_name = f"{repo_name}_{suffix}"
        output_path = HF_LEROBOT_HOME / use_repo_name
        if output_path.exists():
            shutil.rmtree(output_path)
        lerobot_dataset = LeRobotDataset.create(
            repo_id = use_repo_name,
            robot_type = "fetch",
            fps = 10,
            features = FEATURES,
            image_writer_threads = 10,
            image_writer_processes = 0,
        )
        detach_dataset = h5py.File(
            output_path / "detach.h5", 
            "w",
        )
        return (lerobot_dataset, detach_dataset)
    train_lerobot_dataset, train_detach_dataset = make_dataset(repo_name, "train")
    val_lerobot_dataset, val_detach_dataset = make_dataset(repo_name, "val")

    for i, record in enumerate(records):
        append_mshab_trajectory_to_dataset(
            record, train_lerobot_dataset, val_lerobot_dataset,
            train_detach_dataset, val_detach_dataset,
            val_set_ratio, seed + i
        )
    
    train_detach_dataset.close()
    val_detach_dataset.close()

if __name__ == "__main__":
    tyro.cli(main)
