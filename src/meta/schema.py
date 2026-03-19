# Lightweight schema checks for dataset info.json and stats.json.

from typing import Any


REQUIRED_INFO_KEYS = {
    "robot_type",
    "total_episodes",
    "total_frames",
    "fps",
    "features",
}


def validate_info(info: dict[str, Any]) -> None:
    missing = REQUIRED_INFO_KEYS - set(info.keys())
    if missing:
        raise ValueError(f"info.json missing keys: {sorted(missing)}")


def validate_stats(stats: dict[str, Any]) -> None:
    if not isinstance(stats, dict) or not stats:
        raise ValueError("stats.json is empty or invalid")
