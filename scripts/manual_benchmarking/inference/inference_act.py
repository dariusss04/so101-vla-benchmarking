# ACT inference. SPEED_MULT scales the effective loop rate: effective_hz = dataset_fps * SPEED_MULT.

import os
import time
import logging
from pathlib import Path

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


MODEL_BASE_DIR = Path(_get_env("MODEL_BASE_DIR", "./models/act/your-model"))
CHECKPOINT_STEP = _get_env("CHECKPOINT_STEP", "")
DATASET_ROOT = Path(_get_env("DATASET_ROOT", "./data/your-dataset"))
TASK_DESCRIPTION = _get_env("TASK_DESCRIPTION", "Your task description")
SPEED_MULT = float(_get_env("SPEED_MULT", "0.5"))

FOLLOWER_PORT = _get_env("FOLLOWER_PORT", "/dev/tty.usbmodemFOLLOWER")
FOLLOWER_ID = _get_env("FOLLOWER_ID", "FOLLOWER")

CAM_FPS = int(_get_env("CAM_FPS", "30"))
CAM_WIDTH = int(_get_env("CAM_WIDTH", "640"))
CAM_HEIGHT = int(_get_env("CAM_HEIGHT", "480"))
CAM_WARMUP_S = int(_get_env("CAM_WARMUP_S", "2"))
FRONT_CAM_INDEX = int(_get_env("FRONT_CAM_INDEX", "0"))
WRIST_CAM_INDEX = int(_get_env("WRIST_CAM_INDEX", "1"))

CAMERAS = {
    "front": OpenCVCameraConfig(
        index_or_path=FRONT_CAM_INDEX,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fps=CAM_FPS,
        warmup_s=CAM_WARMUP_S,
    ),
    "wrist": OpenCVCameraConfig(
        index_or_path=WRIST_CAM_INDEX,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fps=CAM_FPS,
        warmup_s=CAM_WARMUP_S,
    ),
}


def _resolve_model_dir(base_dir: Path, checkpoint_step: str) -> Path:
    if checkpoint_step:
        step = int(checkpoint_step)
        return base_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    return base_dir


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        actions = policy.predict_action_chunk(obs_tensor)  # (B, S, A)
        actions = postprocessor(actions)
    return actions.squeeze(0)  # (S, A)


def main():
    register_third_party_plugins()
    init_logging()

    device = get_device()
    logging.info(f"Using device: {device}")

    model_dir = _resolve_model_dir(MODEL_BASE_DIR, CHECKPOINT_STEP)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT not found: {DATASET_ROOT}")

    ds = LeRobotDataset(str(DATASET_ROOT))
    dataset_fps = ds.meta.fps
    effective_hz = dataset_fps * SPEED_MULT
    if effective_hz <= 0:
        raise ValueError("SPEED_MULT must be > 0")
    dt_target = 1.0 / effective_hz

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
    last_obs = None

    try:
        logging.info("Starting ACT inference loop...")
        while True:
            now = time.perf_counter()

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
                    task=TASK_DESCRIPTION,
                    robot_type=robot.robot_type,
                )
                action_queue = [actions[i] for i in range(actions.shape[0])]

            if now < next_action_time:
                precise_sleep(next_action_time - now)
                continue

            action_tensor = action_queue.pop(0).unsqueeze(0)
            action_dict = make_robot_action(action_tensor, ds.features)
            robot_action_to_send = robot_action_processor((action_dict, last_obs))
            robot.send_action(robot_action_to_send)

            next_action_time += dt_target
            if now - next_action_time > 1.0:
                next_action_time = now + dt_target

    except KeyboardInterrupt:
        print("\nStopping inference...")

    finally:
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
