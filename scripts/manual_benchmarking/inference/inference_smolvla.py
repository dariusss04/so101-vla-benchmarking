# SmolVLA inference. SPEED_MULT scales the effective loop rate: effective_hz = CAM_FPS * SPEED_MULT.

import os
import time
import logging
import json
from pathlib import Path

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


MODEL_BASE_DIR = Path(_get_env("MODEL_BASE_DIR", "./models/smolvla/your-model"))
CHECKPOINT_STEP = _get_env("CHECKPOINT_STEP", "")
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


def sanitize_smolvla_config(model_dir: Path):
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    if "compile_model" in cfg or "compile_mode" in cfg:
        cfg.pop("compile_model", None)
        cfg.pop("compile_mode", None)
        cfg_path.write_text(json.dumps(cfg, indent=2))


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


def main():
    register_third_party_plugins()
    init_logging()

    device = get_device()
    logging.info(f"Using device: {device}")

    model_dir = _resolve_model_dir(MODEL_BASE_DIR, CHECKPOINT_STEP)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

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

    effective_hz = CAM_FPS * SPEED_MULT
    if effective_hz <= 0:
        raise ValueError("SPEED_MULT must be > 0")
    dt_target = 1.0 / effective_hz

    logging.info("Connecting robot...")
    robot.connect()
    policy.reset()
    pre.reset()
    post.reset()

    try:
        logging.info("Starting SmolVLA inference loop...")
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            obs_frame = build_dataset_frame(dataset_features, obs, prefix="observation")

            action_tensor = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=pre,
                postprocessor=post,
                use_amp=policy.config.use_amp,
                task=TASK_DESCRIPTION,
                robot_type=robot.robot_type,
            )

            action_dict = make_robot_action(action_tensor, dataset_features)
            robot.send_action(action_dict)

            dt = time.perf_counter() - loop_start
            precise_sleep(max(dt_target - dt, 0.0))

    except KeyboardInterrupt:
        print("\nStopping inference...")

    finally:
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
