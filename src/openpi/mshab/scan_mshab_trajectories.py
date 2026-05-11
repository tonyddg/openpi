#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional
import re

@dataclass
class TrajectoryRecord:
    task: str
    subtask: str
    obj_name: str
    run_name: str
    h5_path: Optional[str]
    json_path: Optional[str]


def is_trajectory_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("*.h5")) or any(path.glob("*.json"))


def parse_task_triplet(
    root: Path, traj_dir: Path
) -> Optional[tuple[str, str, str, str]]:
    """
    Parse relative path:
        <root>/<task>/<subtask>/<split>/<obj_name>
    """
    try:
        rel = traj_dir.relative_to(root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 4:
        return None

    task, subtask, split, obj_name = parts[:4]
    return task, subtask, split, obj_name


def collect_runs_in_dir(path: Path) -> List[tuple[str, Optional[Path], Optional[Path]]]:
    """
    Collect all trajectory file groups in one directory.

    Group by file stem:
        20260410_201518.h5
        20260410_201518.json
    =>
        run_name = "20260410_201518"

    Returns:
        List of (run_name, h5_path, json_path), sorted by mtime descending.
    """
    grouped: Dict[str, Dict[str, Optional[Path]]] = {}

    for h5_file in path.glob("*.h5"):
        stem = h5_file.stem
        grouped.setdefault(stem, {"h5": None, "json": None})
        grouped[stem]["h5"] = h5_file

    for json_file in path.glob("*.json"):
        stem = json_file.stem
        grouped.setdefault(stem, {"h5": None, "json": None})
        grouped[stem]["json"] = json_file

    def group_mtime(item: tuple[str, Dict[str, Optional[Path]]]) -> float:
        _, files = item
        times = []
        if files["h5"] is not None:
            times.append(files["h5"].stat().st_mtime)
        if files["json"] is not None:
            times.append(files["json"].stat().st_mtime)
        return max(times) if times else 0.0

    sorted_items = sorted(grouped.items(), key=group_mtime, reverse=True)

    results: List[tuple[str, Optional[Path], Optional[Path]]] = []
    for run_name, files in sorted_items:
        results.append((run_name, files["h5"], files["json"]))
    return results


def collect_trajectory_records(
    dataset_root: Path,
    task_filter: Optional[str] = None,
    subtask_filter: Optional[str] = None,
    obj_name_filter: Optional[str] = None,
) -> List[TrajectoryRecord]:
    records: List[TrajectoryRecord] = []

    for path in dataset_root.rglob("*"):
        if not path.is_dir():
            continue
        if not is_trajectory_dir(path):
            continue

        parsed = parse_task_triplet(dataset_root, path)
        if parsed is None:
            continue

        task, subtask, _split, obj_name = parsed

        if task_filter is not None and task != task_filter:
            continue
        if subtask_filter is not None and subtask != subtask_filter:
            continue
        if obj_name_filter is not None and obj_name != obj_name_filter:
            continue

        runs = collect_runs_in_dir(path)
        for run_name, h5_path, json_path in runs:
            records.append(
                TrajectoryRecord(
                    task=task,
                    subtask=subtask,
                    obj_name=obj_name,
                    run_name=run_name,
                    h5_path=str(h5_path.resolve()) if h5_path else None,
                    json_path=str(json_path.resolve()) if json_path else None,
                )
            )

    records.sort(key=lambda x: (x.task, x.subtask, x.obj_name, x.run_name))
    return records

def obj_name_to_natural_language(obj_name: str) -> str:
    """
    Convert an MS-HAB object name like:
        024_bowl -> bowl
        003_cracker_box -> cracker box
        env-0_024_bowl -> bowl
        obj_3 -> obj 3

    Rules:
    1. Remove optional env prefix like 'env-0_'.
    2. Remove leading numeric ID like '024_'.
    3. Replace underscores/hyphens with spaces.
    4. Collapse repeated spaces.
    5. Lowercase output.
    """
    if not isinstance(obj_name, str):
        raise TypeError(f"obj_name must be str, got {type(obj_name)}")

    text = obj_name.strip().lower()

    # Remove optional env prefix, e.g. env-0_024_bowl
    text = re.sub(r"^env-\d+_", "", text)

    # Remove leading numeric ID, e.g. 024_bowl -> bowl
    text = re.sub(r"^\d+_", "", text)

    # Replace separators with spaces
    text = text.replace("_", " ").replace("-", " ")

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_furniture_name(s):
    s = s.decode('utf-8') if isinstance(s, bytes) else str(s)
    s = re.sub(r'_:\d+$', '', s)
    s = re.sub(r'_\d+$', '', s)
    return s.split('_')[-1]

PICK_TEMPLATES: List[str] = [
    "pick {obj}",
    "grasp {obj}",
    "pick up the {obj}",
    "reach for the {obj} and pick it up",
    "move to the {obj} and grasp it",
    "lift the {obj}",
]

PICK_TEMPLATES_WITH_TARGET: List[str] = [
    "pick the {obj} from the {target}",
    "grasp the {obj} in the {target}",
    "take the {obj} out of the {target}",
    "lift the {obj} from the {target}",
    "retrieve the {obj} from the {target}",
    "remove the {obj} from the {target}",
]

PLACE_TEMPLATES: List[str] = [
    "place {obj}",
    "put down the {obj}",
    "place the {obj} at the target",
    "move the {obj} to the goal and release it",
    "carry the {obj} and place it",
    "put the {obj} down at the target",
]

PLACE_TEMPLATES_WITH_TARGET: List[str] = [
    "place the {obj} in the {target}",
    "put the {obj} in the {target}",
    "drop the {obj} into the {target}",
    "set the {obj} in the {target}",
    "move the {obj} to the {target}",
    "put the {obj} down in the {target}",
]

DEFAULT_TEMPLATE_MAP: Dict[str, Dict[str, List[str]]] = {
    "pick": {
        "default": PICK_TEMPLATES,
        "target": PICK_TEMPLATES_WITH_TARGET,
    },
    "place": {
        "default": PLACE_TEMPLATES,
        "target": PLACE_TEMPLATES_WITH_TARGET,
    },
}

def sample_language_instruction(
    record: TrajectoryRecord,
    templates: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
    target: Optional[str] = None
) -> str:
    """
    Sample one language instruction for a trajectory record.

    Args:
        record: Trajectory metadata.
        templates: Optional custom templates, otherwise find in DEFAULT_TEMPLATE_MAP by record and target.
        rng: Optional random.Random instance for reproducibility.
        target: Source reception for pick or destination reception for place.

    Returns:
        A sampled natural-language instruction string.

    Raises:
        ValueError: if record.subtask has no available templates.
    """
    if templates is None:
        
        template_mode = "default" if (target is None or len(target) <= 1) else "target"
        subtask = record.subtask.strip().lower()
        try:
            templates = DEFAULT_TEMPLATE_MAP[subtask][template_mode]
        except:
            raise ValueError(f"No templates found for subtask: {record.subtask}")

    if rng is None:
        rng = random.Random()

    subtask = record.subtask.strip().lower()
    obj_text = obj_name_to_natural_language(record.obj_name)

    template = rng.choice(templates)
    return template.format(
        obj = obj_text,
        target = target,
        task = record.task,
        subtask = subtask,
        obj_name = record.obj_name,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan MS-HAB trajectory outputs and return all h5/json records."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="MS-HAB dataset root directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Only scan a specific task, e.g. tidy_house",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Only scan a specific subtask, e.g. pick / place / navigate / open / close",
    )
    parser.add_argument(
        "--obj-name",
        type=str,
        default=None,
        help="Only scan a specific object, e.g. 024_bowl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    records = collect_trajectory_records(
        dataset_root=dataset_root,
        task_filter=args.task,
        subtask_filter=args.subtask,
        obj_name_filter=args.obj_name,
    )
    data = [asdict(r) for r in records]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2 if args.pretty else None)

    print(json.dumps(data, ensure_ascii=False, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()