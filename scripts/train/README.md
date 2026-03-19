# Training

Training entry points for ACT and SmolVLA on LeRobot datasets.

- `train_act.py` — ACT training with checkpoint push and **true resume** support.
- `train_smolvla.py` — SmolVLA training with checkpoint push and **true resume** support.
- `train_act.sh` — LeRobot default training (checkpoint push, **no true resume**).
- `train_smolvla.sh` — LeRobot default training (checkpoint push, **no true resume**).

The `.sh` scripts use `lerobot-train` (LeRobot default pipeline), which **does not support true resume**. The Python scripts in this folder implement full checkpoint + optimizer state saving and can resume cleanly.

All paths, repo IDs, and hyperparameters are set via environment variables.
