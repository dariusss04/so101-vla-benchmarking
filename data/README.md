# Datasets

This folder contains **dataset metadata + download instructions** for the physical VLA benchmarks. All datasets were recorded manually via teleoperation and uploaded to the Hugging Face Hub.

**Why this matters:** VLA models are sensitive to real‑world conditions (lighting, camera placement, background clutter, object appearance). A model trained on one setup often degrades on a different setup unless you re‑record data under the new conditions.

If you want to reproduce or extend the datasets, use the scripts in `scripts/record/` to record new episodes and push them to your own Hugging Face dataset repo.

You can also **preview datasets visually** using the LeRobot dataset visualizer:  
https://huggingface.co/spaces/lerobot/visualize_dataset

## Available datasets

- `cube_stack/` — cube stacking benchmark. Includes task summary and download instructions.
- `cube_push/` — placeholder for the cube pushing benchmark (not yet available).
