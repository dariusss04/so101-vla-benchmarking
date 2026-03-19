# Training

Training entry points for ACT and SmolVLA on LeRobot datasets.

- `train_act.py` / `.sh` — ACT training with checkpoint push (true resume supported).
- `train_smolvla.py` / `.sh` — SmolVLA training with checkpoint push (true resume supported).

The `.sh` scripts use `lerobot-train` (LeRobot default pipeline), which **does not support true resume**. The Python scripts in this folder implement full checkpoint + optimizer state saving and can resume cleanly.

All paths, repo IDs, and hyperparameters are set via environment variables.
