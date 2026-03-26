# Dataset Conversion (CALVIN)

This directory contains tools to convert the raw CALVIN dataset into the LeRobot format for VLA benchmarking.

## `convert_calvin_dataset.py`
Converts raw CALVIN folders (`training`, `validation`, `lang_annotations`) into a unified LeRobot dataset.

**Environment Variables:**
- `DATASET_PATH`: Path to the raw CALVIN dataset root.
- `OUT_DIR`: (Optional) Output path for converted dataset (defaults to `data/calvin_lerobot`).
- `TASKS`: JSON-formatted list of tasks to convert (e.g., `["push block", "lift block"]`).
- `MAPPING_JSON`: (Optional) Path to a JSON file for task relabeling.
- `HUB_REPO_ID`: (Optional) Hugging Face repo ID for auto-upload.
- `PUSH_TO_HUB`: (Optional) Set to `true` to push to HF Hub after conversion.

## `inspect_calvin_tasks.py`
Analyzes raw CALVIN data to report task frequencies and suggest conversion lists.

**Environment Variables:**
- `DATASET_PATH`: Path to raw data.
- `TOP_K`: (Optional) Number of top tasks to show.
- `MIN_TRAIN` / `MIN_VAL`: (Optional) Minimum occurrences for a task to be considered valid.