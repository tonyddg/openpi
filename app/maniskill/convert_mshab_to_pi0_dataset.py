from pathlib import Path
import shutil
from typing import Literal, Optional
import tyro
import h5py
import json
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.mshab.scan_mshab_trajectories import collect_trajectory_records, sample_language_instruction, TrajectoryRecord, extract_furniture_name

def append_mshab_trajectory_to_dataset(
    record: TrajectoryRecord,
    dataset: LeRobotDataset,
):
    episode_total_length = 0
    episode_total_nums = 0
    episode_valid_nums = 0 

    print(f"resolve record: {record}")

    assert record.json_path is not None
    assert record.h5_path is not None

    with open(record.json_path, 'r') as f:
        episode_info_list = list(json.load(f)["episodes"])

    with h5py.File(record.h5_path, 'r') as h5_file:
        episode_total_nums += 1

        for episode_info in episode_info_list:
            episode_id = int(episode_info["episode_id"])
            
            episode_group = h5_file[f"traj_{episode_id}"]
            assert isinstance(episode_group, h5py.Group)

            # 判断轨迹是否可用
            episode_step_success = np.asarray(episode_group["success"][:]) # type: ignore
            episode_step_fail = np.asarray(episode_group["fail"][:]) # type: ignore
            # 获取第一次成功或失败的步数
            where_success_step = np.where(episode_step_success)[0]
            success_step = int(where_success_step[0]) if where_success_step.size > 0 else 199
            where_fail_step = np.where(episode_step_fail)[0]
            fail_step = int(where_fail_step[0]) if where_fail_step.size > 0 else 199
            if success_step <= fail_step and success_step >= 199:
                print(f"episode_id: {episode_id} is invalid")
                continue 
            episode_valid_nums += 1
            episode_total_length += success_step
            print(f"episode_id: {episode_id} has length {success_step}")

            # 依据模板生成任务指令
            episode_recption = episode_group["info/target_receptacles"][0] # type: ignore
            target_recption = extract_furniture_name(episode_recption)
            print(f"episode_recption: {episode_recption}, target_recption: {target_recption}")

            task = sample_language_instruction(record, target = target_recption)

            for i in range(success_step):
                dataset.add_frame(
                    {
                        # img: 长 201, 为 t0 到 t200 (作完最后一步动作再记录一张图片)
                        "fetch_hand": np.asarray(episode_group["obs/sensor_data/fetch_hand/rgb"][i, :, :, :]), # type: ignore
                        "fetch_head": np.asarray(episode_group["obs/sensor_data/fetch_head/rgb"][i, :, :, :]), # type: ignore
                        # state 长 201, 但索引为 0 的状态无效 (执行动作前的 evaluate), 为 t0 到 t199 
                        "state": np.asarray(episode_group["info/pi0_eef_state"][i + 1, :]), # type: ignore
                        # action 长 201, 但索引为 0 的动作无效 (reset 前后的 eff 差), 为 t0 到 t199 
                        "actions_abs": np.asarray(episode_group["info/pi0_eef_abs_action"][i + 1, :]), # type: ignore
                        "actions_ref": np.asarray(episode_group["info/pi0_eef_ref_action"][i + 1, :]), # type: ignore
                        
                        "task": task,
                    }
                )
            dataset.save_episode()

    print(f"episode total nums: {episode_total_nums}, episode valid nums: {episode_valid_nums}, episode mean length: {episode_total_length / episode_valid_nums}")

from openpi.mshab.utility import ROT_TYPE, STATE_ROT_LENGTH_DICT

# uv run app/maniskill/convert_mshab_to_lerobot.py 
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
    task_filter: Optional[str] = None,
    subtask_filter: Optional[str] = None,
    obj_name_filter: Optional[str] = None,

    # 依据 rot type 补充数据集名称
    is_name_by_rot_type: bool = True,
    state_rot_type: ROT_TYPE = "axis_angle",
    action_rot_type: ROT_TYPE = "axis_angle"
):
    
    state_length = 4 + STATE_ROT_LENGTH_DICT[state_rot_type]
    action_length = 4 + STATE_ROT_LENGTH_DICT[action_rot_type]

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
            repo_name += f"_{task_filter}"
        else:
            repo_name += "_all"

        if subtask_filter is not None:
            repo_name += f"_{subtask_filter}"
        else:
            repo_name += "_all"

        if obj_name_filter is not None:
            repo_name += f"_{obj_name_filter}"
        else:
            repo_name += "_all"

    if is_name_by_rot_type:
        repo_name += f"_state_{state_rot_type}"
        repo_name += f"_action_{action_rot_type}"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)
    dataset = LeRobotDataset.create(
        repo_id = repo_name,
        robot_type = "fetch",
        fps = 10,
        features = {
            "fetch_hand": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "fetch_head": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_length,),
                "names": [f"state with {state_rot_type} rot"],
            },
            "actions_abs": {
                "dtype": "float32",
                "shape": (action_length,),
                "names": [f"actions_abs with {action_rot_type} rot"],
            },
            "actions_ref": {
                "dtype": "float32",
                "shape": (action_length,),
                "names": [f"actions_ref with {action_rot_type} rot"],
            },
        },
        image_writer_threads = 10,
        image_writer_processes = 0,
    )

    for record in records:
        append_mshab_trajectory_to_dataset(
            record, dataset
        )

if __name__ == "__main__":
    tyro.cli(main)
