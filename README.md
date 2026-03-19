# Benchmarking VLAs on a Physical SO101 Robot

This repository contains the full **manual benchmarking pipeline** used in our VLA (Vision‑Language‑Action) experiments on a physical SO101 setup. It includes dataset metadata, training and inference scripts (ACT and SmolVLA), benchmarking metrics, and configuration files designed for reproducible evaluation.

## Project Goals

- Build a **clean, reproducible pipeline** for manual VLA benchmarking.
- Record and organize datasets in LeRobot format.
- Train ACT/SmolVLA models and evaluate them on physical tasks.
- Report success rates, error modes, and timing metrics for real‑world evaluation.

## Hardware & Setup

- **Robot:** SO101 (Follower + Leader for teleop)
- **Cameras:** front + wrist RGB (640×480)
- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) (v0.4.4)

## Datasets

Datasets are recorded via teleoperation and stored in LeRobot format. **Real‑world data is sensitive** to lighting, camera placement, and object appearance—models trained on one setup may not generalize to another.

See `data/README.md` for dataset details and download instructions.

You can also preview datasets with the LeRobot visualizer:  
https://huggingface.co/spaces/lerobot/visualize_dataset

## Models

We benchmark two model families:

- **ACT** (vision‑action model)
- **SmolVLA** (vision‑language‑action model)

Training scripts are provided in `scripts/train/` and support checkpointing for resume.

## Manual Benchmarking

Inference is performed on physical tasks such as stacking and pushing. Metrics include:

- success rate
- task completion time
- failure mode breakdown (failed grab, wrong target, overshoot, etc.)

Stats are recorded into CSV files during inference. See `inference/` and `configs/csv_schema.yaml`.

## Repository Structure

```
benchmarkingVlas/
  configs/            # YAML configs (tasks, cameras, defaults, csv schema)
  data/               # dataset metadata + download instructions
  inference/          # inference scripts with stats logging
  scripts/            # teleop, record, calibration, training, maintenance
  src/                # shared helpers (env, metadata, stats I/O)
```

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables (ports, repo IDs, paths). See `configs/` and the script READMEs for examples.

3. Run the desired script in `scripts/` or `inference/`.

## License

MIT License — see [LICENSE](LICENSE).
