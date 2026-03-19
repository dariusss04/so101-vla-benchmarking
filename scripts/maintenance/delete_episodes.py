# Remove specific episodes from a local LeRobot dataset and optionally push the cleaned dataset to the Hub.

import os
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.utils import init_logging


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env var: {name}")
    return value


def _parse_episode_indices(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


DATASET_REPO_ID = _get_env("DATASET_REPO_ID", "your-username/your-dataset")
DATASET_ROOT = Path(_get_env("DATASET_ROOT", "./data/your-dataset"))
EPISODE_INDICES = _parse_episode_indices(_get_env("EPISODE_INDICES", ""))
PUSH_TO_HUB = _get_env("PUSH_TO_HUB", "false").lower() == "true"


def main():
    init_logging()
    logging.info("Starting delete-episodes operation")

    if not EPISODE_INDICES:
        raise ValueError("EPISODE_INDICES must be a comma-separated list, e.g. '0,1,5'")

    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
    )

    logging.info(f"Loaded dataset with {dataset.meta.total_episodes} episodes")

    output_dir = DATASET_ROOT
    dataset.root = Path(str(dataset.root) + "_old")

    new_dataset = delete_episodes(
        dataset=dataset,
        episode_indices=EPISODE_INDICES,
        output_dir=output_dir,
        repo_id=DATASET_REPO_ID,
    )

    logging.info(f"Delete complete. New episode count: {new_dataset.meta.total_episodes}")

    if PUSH_TO_HUB:
        logging.info("Pushing cleaned dataset to Hugging Face Hub")
        new_dataset.push_to_hub()

    logging.info("Done.")


if __name__ == "__main__":
    main()
