# Inference (no stats)

Minimal inference scripts for running models without logging metrics.

- `inference_act.py` — ACT inference loop.
- `inference_smolvla.py` — SmolVLA inference loop.

Both scripts use environment variables for paths, ports, and camera settings, and expose `SPEED_MULT` to control the effective loop rate.
