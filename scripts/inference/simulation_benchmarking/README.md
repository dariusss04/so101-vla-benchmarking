# Specialized Simulation Inference

Localized and interactive testing tools for CALVIN simulation.

## `inference_smolvla_singlestep.py`
Evaluates model success on isolated, single-step tasks.

**Environment Variables:**
- Uses the same environment variables as the main `inference_smolvla.py`.
- `TARGET_TASKS`: (Optional) Specific tasks to evaluate.

## `inference_smolvla_interactive.py`
Allows manual control of the agent via keyboard-driven text instructions.

**Environment Variables:**
- `DATASET_PATH`: Path to raw CALVIN data.
- `TRAIN_FOLDER`: Path to model.
- `MAX_STEPS`: Max rollout length.
