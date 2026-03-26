# SmolVLA inference with stats logging (some metrics are automatic, others are prompted after the run).

import os
import time
import logging
import csv
import json
import shutil
from pathlib import Path
from statistics import mean, pstdev

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env var: {name}")
    return value


MODEL_ID = _get_env("MODEL_ID", "smolvla-cube-stack")
CHECKPOINT_STEP = int(_get_env("CHECKPOINT_STEP", "100000"))
TRIAL_ID = int(_get_env("TRIAL_ID", "1"))
TASK_NAME = _get_env("TASK_NAME", "Pick up the red cube and place it on top of the blue cube")
ROBUSTNESS_CONDITION = _get_env("ROBUSTNESS_CONDITION", "static")
TIME_LIMIT_S = float(_get_env("TIME_LIMIT_S", "60"))

CAM_TIMEOUT_MS = int(_get_env("CAM_TIMEOUT_MS", "1000"))
MODEL_CALL_EVERY = int(_get_env("MODEL_CALL_EVERY", "3"))

BASE_MODEL_DIR = Path(_get_env("MODEL_BASE_DIR", "./models/smolvla/cube-stack"))
CSV_PATH = Path(_get_env("CSV_PATH", "./inference/smolvla/cube-stack/stats.csv"))

RECORD_FPS = int(_get_env("RECORD_FPS", "30"))
SPEED_MULT = float(_get_env("SPEED_MULT", "0.6"))
INFERENCE_FPS = max(1, int(RECORD_FPS * SPEED_MULT))

FOLLOWER_PORT = _get_env("FOLLOWER_PORT", "/dev/tty.usbmodemFOLLOWER")
FOLLOWER_ID = _get_env("FOLLOWER_ID", "FOLLOWER")

CAM_WIDTH = int(_get_env("CAM_WIDTH", "640"))
CAM_HEIGHT = int(_get_env("CAM_HEIGHT", "480"))
CAM_WARMUP_S = int(_get_env("CAM_WARMUP_S", "2"))
FRONT_CAM_INDEX = int(_get_env("FRONT_CAM_INDEX", "0"))
WRIST_CAM_INDEX = int(_get_env("WRIST_CAM_INDEX", "1"))

CAMERAS = {
    "front": OpenCVCameraConfig(index_or_path=FRONT_CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT, fps=RECORD_FPS, warmup_s=CAM_WARMUP_S),
    "wrist": OpenCVCameraConfig(index_or_path=WRIST_CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT, fps=RECORD_FPS, warmup_s=CAM_WARMUP_S),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dir(checkpoint_step: int) -> Path:
    ckpt_name = f"{checkpoint_step:06d}"
    model_dir = BASE_MODEL_DIR / "checkpoints" / ckpt_name / "pretrained_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir}")
    return model_dir


def sanitize_smolvla_config(model_dir: Path):
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return

    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        logging.warning(f"Could not read config.json: {e}")
        return

    bad_keys = {"compile_model", "compile_mode"}
    if not any(k in cfg for k in bad_keys):
        return

    backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(cfg_path, backup_path)

    for k in bad_keys:
        cfg.pop(k, None)

    cfg_path.write_text(json.dumps(cfg, indent=2))
    logging.warning("Removed unsupported SmolVLA config keys: compile_model/compile_mode")


def load_pre_post(model_dir: Path, device: torch.device):
    pre = PolicyProcessorPipeline.from_pretrained(
        model_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": str(device)}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    post = PolicyProcessorPipeline.from_pretrained(
        model_dir,
        config_filename="policy_postprocessor.json",
        overrides={"device_processor": {"device": str(device)}},
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return pre, post


def prompt_binary(prompt: str) -> int:
    while True:
        val = input(prompt).strip()
        if val in ("0", "1"):
            return int(val)
        print("Please enter 0 or 1.")


def prompt_int(prompt: str) -> int:
    while True:
        val = input(prompt).strip()
        if val.isdigit():
            return int(val)
        print("Please enter an integer.")


def write_csv_row(row: dict):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_PATH.exists()

    headers = [
        "model_id",
        "trial_id",
        "task_name",
        "robustness_condition",
        "time_limit_s",
        "duration_s",
        "completed_within_time",
        "success",
        "failed_to_grab",
        "dropped",
        "wrong_target",
        "incomplete_placement",
        "push_overshoot",
        "push_undershoot",
        "num_retries",
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
    ]

    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    register_third_party_plugins()
    init_logging()

    device = get_device()
    logging.info(f"Using device: {device}")

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_BASE_DIR not found: {BASE_MODEL_DIR}")

    model_dir = resolve_model_dir(CHECKPOINT_STEP)
    sanitize_smolvla_config(model_dir)

    policy = SmolVLAPolicy.from_pretrained(str(model_dir)).to(device).eval()
    pre, post = load_pre_post(model_dir, device)

    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=CAMERAS,
    )
    robot = SO101Follower(robot_cfg)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    logging.info("Connecting robot...")
    robot.connect()

    for cam in robot.cameras.values():
        _orig = cam.async_read
        cam.async_read = (
            lambda timeout_ms=CAM_TIMEOUT_MS, _orig=_orig: _orig(timeout_ms=timeout_ms)
        )

    policy.reset()
    pre.reset()
    post.reset()

    action_vals = []
    last_action_tensor = None
    step_idx = 0
    stop_reason = None
    error = None
    start_time = time.perf_counter()

    try:
        logging.info("Starting SmolVLA inference loop...")
        while True:
            loop_start = time.perf_counter()

            if loop_start - start_time >= TIME_LIMIT_S:
                stop_reason = "timeout"
                break

            obs = robot.get_observation()
            obs_frame = build_dataset_frame(dataset_features, obs, prefix="observation")

            if last_action_tensor is None or (step_idx % MODEL_CALL_EVERY == 0):
                last_action_tensor = predict_action(
                    observation=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=pre,
                    postprocessor=post,
                    use_amp=policy.config.use_amp,
                    task=TASK_NAME,
                    robot_type=robot.robot_type,
                )

            action_vals.extend(last_action_tensor.flatten().tolist())

            action_dict = make_robot_action(last_action_tensor, dataset_features)
            robot.send_action(action_dict)
            step_idx += 1

            dt = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / INFERENCE_FPS - dt, 0.0))

    except KeyboardInterrupt:
        stop_reason = "manual"

    except Exception as e:
        stop_reason = "error"
        error = e

    finally:
        robot.disconnect()
        print("Disconnected")

    if stop_reason == "error":
        print(f"Run failed due to error: {error}")
        return

    duration_s = time.perf_counter() - start_time
    completed_within_time = 1 if duration_s <= TIME_LIMIT_S else 0

    if action_vals:
        action_mean = mean(action_vals)
        action_std = pstdev(action_vals) if len(action_vals) > 1 else 0.0
        action_min = min(action_vals)
        action_max = max(action_vals)
    else:
        action_mean = float("nan")
        action_std = float("nan")
        action_min = float("nan")
        action_max = float("nan")

    print(f"Inference trial {TRIAL_ID} finished ({stop_reason}).")
    success = prompt_binary("Enter success (0/1): ")

    if success == 0:
        failed_to_grab = prompt_binary("Enter failed_to_grab (0/1): ")
        dropped = prompt_binary("Enter dropped (0/1): ")
        wrong_target = prompt_binary("Enter wrong_target (0/1): ")
        incomplete_placement = prompt_binary("Enter incomplete_placement (0/1): ")
        push_overshoot = prompt_binary("Enter push_overshoot (0/1): ")
        push_undershoot = prompt_binary("Enter push_undershoot (0/1): ")
    else:
        failed_to_grab = 0
        dropped = 0
        wrong_target = 0
        incomplete_placement = 0
        push_overshoot = 0
        push_undershoot = 0

    num_retries = prompt_int("Enter num_retries (integer): ")

    row = {
        "model_id": MODEL_ID,
        "trial_id": TRIAL_ID,
        "task_name": TASK_NAME,
        "robustness_condition": ROBUSTNESS_CONDITION,
        "time_limit_s": TIME_LIMIT_S,
        "duration_s": duration_s,
        "completed_within_time": completed_within_time,
        "success": success,
        "failed_to_grab": failed_to_grab,
        "dropped": dropped,
        "wrong_target": wrong_target,
        "incomplete_placement": incomplete_placement,
        "push_overshoot": push_overshoot,
        "push_undershoot": push_undershoot,
        "num_retries": num_retries,
        "action_mean": action_mean,
        "action_std": action_std,
        "action_min": action_min,
        "action_max": action_max,
    }

    write_csv_row(row)
    print(f"Saved stats to {CSV_PATH}")


if __name__ == "__main__":
    main()
