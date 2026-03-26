# Simulation Training

Training pipeline for SmolVLA using converted CALVIN datasets.

## `train_smolvla.py`
Custom training loop with true-resume support and Hub integration.

**Environment Variables:**
- `DATASET_PATH`: Local path or HF Repo ID of the converted LeRobot dataset.
- `OUTPUT_DIR`: Save directory for checkpoints.
- `PRETRAINED_MODEL`: (Optional) Path to base model or checkpoint to resume from.
- `HUB_REPO_ID`: (Optional) HF Hub repo for auto-pushing checkpoints.
- `PUSH_TO_HUB`: (Optional) Set to `true` to enable Hub syncing.
- `BATCH_SIZE`, `STEPS`, `LR`: (Optional) Hyperparameters.
- `SAVE_EVERY`: (Optional) Checkpoint frequency.
