# Record teleoperated episodes with an SO101 leader/follower setup and two cameras.
# Configure dataset, ports, cameras, and recording options via environment variables.

import os
from pathlib import Path

from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.scripts.lerobot_record import record_loop


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val


def main():
    register_third_party_plugins()
    init_logging()

    dataset_repo_id = _get_required("DATASET_REPO_ID")
    dataset_root = Path(_get_required("DATASET_ROOT"))
    task_description = _get_required("TASK_DESCRIPTION")
    num_episodes = int(_get_env("NUM_EPISODES", "1"))
    resume = _get_env("RESUME", "true").lower() == "true"
    push_to_hub = _get_env("PUSH_TO_HUB", "false").lower() == "true"

    fps = int(_get_env("FPS", "30"))

    follower_port = _get_required("FOLLOWER_PORT")
    leader_port = _get_required("LEADER_PORT")
    follower_id = _get_required("FOLLOWER_ID")
    leader_id = _get_required("LEADER_ID")

    front_index = int(_get_env("FRONT_CAM_INDEX", "0"))
    wrist_index = int(_get_env("WRIST_CAM_INDEX", "1"))
    cam_width = int(_get_env("CAM_WIDTH", "640"))
    cam_height = int(_get_env("CAM_HEIGHT", "480"))
    cam_fps = int(_get_env("CAM_FPS", str(fps)))

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=front_index,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_index,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
        ),
    }

    robot_cfg = SO101FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=cameras,
    )
    robot = make_robot_from_config(robot_cfg)

    teleop_cfg = SO101LeaderConfig(
        port=leader_port,
        id=leader_id,
    )
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_obs_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if resume:
        dataset = LeRobotDataset(
            dataset_repo_id,
            root=dataset_root,
        )

        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(cameras),
        )

        sanity_check_dataset_robot_compatibility(
            dataset,
            robot,
            fps,
            dataset_features,
        )
    else:
        dataset = LeRobotDataset.create(
            dataset_repo_id,
            fps,
            root=dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )

    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded = 0
        while recorded < num_episodes and not events["stop_recording"]:
            log_say(
                f"Recording episode {dataset.meta.total_episodes}",
                blocking=False,
            )

            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_obs_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=60,
                single_task=task_description,
                display_data=False,
            )

            dataset.save_episode()
            recorded += 1

    robot.disconnect()
    teleop.disconnect()

    if listener is not None:
        listener.stop()

    if push_to_hub:
        dataset.push_to_hub()

    log_say("Recording finished", blocking=True)


if __name__ == "__main__":
    main()
