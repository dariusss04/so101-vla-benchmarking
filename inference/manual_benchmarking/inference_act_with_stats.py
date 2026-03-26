# ACT inference with stats logging (some metrics are automatic, others are prompted after the run).

import os
import time
import logging
import csv
from pathlib import Path
from statistics import mean, pstdev

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    transition_to_batch,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.factory import make_default_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env var: {name}")
    return value


MODEL_ID = _get_env("MODEL_ID", "act-cube-stack")
CHECKPOINT_STEP = int(_get_env("CHECKPOINT_STEP", "80000"))
TRIAL_ID = int(_get_env("TRIAL_ID", "1"))
TASK_NAME = _get_env("TASK_NAME", "Pick up the red cube and place it on top of the blue cube")
ROBUSTNESS_CONDITION = _get_env("ROBUSTNESS_CONDITION", "static")
TIME_LIMIT_S = float(_get_env("TIME_LIMIT_S", "60"))
SPEED_MULT = float(_get_env("SPEED_MULT", "0.5"))
BASE_MODEL_DIR = Path(_get_env("MODEL_BASE_DIR", "./models/act/cube-stack"))
DATASET_ROOT = Path(_get_env("DATASET_ROOT", "./data/cube-stack"))
CSV_PATH = Path(_get_env("CSV_PATH", "./inference/act/cube-stack/stats.csv"))

FOLLOWER_PORT = _get_env("FOLLOWER_PORT", "/dev/tty.usbmodemFOLLOWER")
FOLLOWER_ID = _get_env("FOLLOWER_ID", "FOLLOWER")

CAM_FPS = int(_get_env("CAM_FPS", "30"))
CAM_WIDTH = int(_get_env("CAM_WIDTH", "640"))
CAM_HEIGHT = int(_get_env("CAM_HEIGHT", "480"))
CAM_WARMUP_S = int(_get_env("CAM_WARMUP_S", "2"))
FRONT_CAM_INDEX = int(_get_env("FRONT_CAM_INDEX", "0"))
WRIST_CAM_INDEX = int(_get_env("WRIST_CAM_INDEX", "1"))

CAMERAS = {
    "front": OpenCVCameraConfig(index_or_path=FRONT_CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS, warmup_s=CAM_WARMUP_S),
    "wrist": OpenCVCameraConfig(index_or_path=WRIST_CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS, warmup_s=CAM_WARMUP_S),
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
    if model_dir.exists():
        return model_dir

    ckpt_root = BASE_MODEL_DIR / "checkpoints"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoints folder not found: {ckpt_root}")

    candidates = []
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            candidates.append(int(p.name))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_root}")

    latest = max(candidates)
    logging.warning(
        f"Requested checkpoint {ckpt_name} not found. Falling back to latest {latest:06d}."
    )
    return BASE_MODEL_DIR / "checkpoints" / f"{latest:06d}" / "pretrained_model"


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


def predict_action_chunk(obs_frame, policy, device, preprocessor, postprocessor, task, robot_type):
    obs_tensor = prepare_observation_for_inference(
        observation=obs_frame,
        device=device,
        task=task,
        robot_type=robot_type,
    )
    obs_tensor = preprocessor(obs_tensor)

    with torch.inference_mode():
        actions = policy.predict_action_chunk(obs_tensor)
        actions = postprocessor(actions)
    return actions.squeeze(0)


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
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    model_dir = resolve_model_dir(CHECKPOINT_STEP)

    ds = LeRobotDataset(str(DATASET_ROOT))
    fps = ds.meta.fps

    policy = ACTPolicy.from_pretrained(str(model_dir)).to(device).eval()
    pre, post = load_pre_post(model_dir, device)

    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=FOLLOWER_ID,
        cameras=CAMERAS,
    )
    robot = SO101Follower(robot_cfg)

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    logging.info("Connecting robot...")
    robot.connect()
    policy.reset()
    pre.reset()
    post.reset()

    action_queue = []
    next_action_time = time.perf_counter()
    dt_target = (1.0 / fps) / SPEED_MULT

    all_action_vals = []
    stop_reason = None
    error = None
    trial_start = time.perf_counter()
    last_obs = None

    try:
        logging.info("Starting policy inference loop (chunk playback)...")
        while True:
            now = time.perf_counter()

            if now - trial_start >= TIME_LIMIT_S:
                stop_reason = "timeout"
                break

            if not action_queue:
                obs = robot.get_observation()
                last_obs = obs
                obs_processed = robot_observation_processor(obs)
                obs_frame = build_dataset_frame(ds.features, obs_processed, prefix="observation")

                actions = predict_action_chunk(
                    obs_frame=obs_frame,
                    policy=policy,
                    device=device,
                    preprocessor=pre,
                    postprocessor=post,
                    task=TASK_NAME,
                    robot_type=robot.robot_type,
                )
                action_queue = [actions[i] for i in range(actions.shape[0])]

            if now < next_action_time:
                precise_sleep(next_action_time - now)
                continue

            action_tensor = action_queue.pop(0).unsqueeze(0)
            action_dict = make_robot_action(action_tensor, ds.features)

            all_action_vals.extend(list(action_dict.values()))

            robot_action_to_send = robot_action_processor((action_dict, last_obs))
            robot.send_action(robot_action_to_send)

            next_action_time += dt_target
            if now - next_action_time > 1.0:
                next_action_time = now + dt_target

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

    duration_s = time.perf_counter() - trial_start
    completed_within_time = 1 if duration_s <= TIME_LIMIT_S else 0

    if all_action_vals:
        action_mean = mean(all_action_vals)
        action_std = pstdev(all_action_vals) if len(all_action_vals) > 1 else 0.0
        action_min = min(all_action_vals)
        action_max = max(all_action_vals)
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
