# Simulation Inference (Multistep)

Primary evaluation pipeline for running SmolVLA models in the CALVIN simulator.

## `inference_smolvla.py`
Executes long-horizon multistep evaluations across task sequences.

**Environment Variables:**
- `DATASET_PATH`: Path to the raw CALVIN dataset validation split.
- `TRAIN_FOLDER`: Path to the trained model checkpoint folder.
- `CHECKPOINT`: (Optional) Path to a specific checkpoint file.
- `DEVICE`: `cuda`, `mps`, or `cpu`.
- `EVAL_MATRIX`: (Optional) JSON matrix string for specific evaluation sequences.
- `RECORD_VIDEO_DIR`: (Optional) Directory to save evaluation videos.
- `EP_LEN`: (Optional) Max steps per subtask (default: 90).
- `DEBUG`: (Optional) Set to `true` for GUI visualization.