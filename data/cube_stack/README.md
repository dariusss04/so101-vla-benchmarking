# Cube Stack Dataset

**Task:** pick up the red cube and place it on top of the blue cube.

**Hugging Face dataset:** [dariusss04/cube-stack](https://huggingface.co/datasets/dariusss04/cube-stack)

## Summary (from `info.json`)

- Episodes: **160**
- Frames: **28,761**
- FPS: **30**
- Robot: **SO101 follower**
- Cameras: **front + wrist**, 640×480 RGB, AV1 encoded

## Download

```bash
pip install -U huggingface_hub
huggingface-cli login

export DATASET_DIR=./data/cube_stack
huggingface-cli download dariusss04/cube-stack \
  --repo-type dataset \
  --local-dir "$DATASET_DIR" \
  --local-dir-use-symlinks False
```

## Notes

This dataset was recorded via teleoperation. Physical VLAs are sensitive to lighting, camera placement, and object appearance. If you deploy in a different setup, you should record a new dataset using `scripts/record/`.
