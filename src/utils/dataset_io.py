# Dataset loading and metadata helpers for LeRobot datasets.

import json
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_dataset(root: Path, repo_id: str | None = None) -> LeRobotDataset:
    if repo_id:
        return LeRobotDataset(repo_id, root=root)
    return LeRobotDataset(str(root))


def read_info_json(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    return json.loads(info_path.read_text())
