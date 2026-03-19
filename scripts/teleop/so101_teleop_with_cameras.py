# Teleoperate an SO101 follower using an SO101 leader with two OpenCV cameras.
# Configure ports/IDs/cameras via environment variables before running.

import os
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def main():
    front_index = int(_get_env("FRONT_CAM_INDEX", "0"))
    wrist_index = int(_get_env("WRIST_CAM_INDEX", "1"))
    cam_width = int(_get_env("CAM_WIDTH", "640"))
    cam_height = int(_get_env("CAM_HEIGHT", "480"))
    cam_fps = int(_get_env("CAM_FPS", "30"))
    cam_warmup = float(_get_env("CAM_WARMUP_S", "2"))

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=front_index,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
            warmup_s=cam_warmup,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=wrist_index,
            width=cam_width,
            height=cam_height,
            fps=cam_fps,
            warmup_s=cam_warmup,
        ),
    }

    robot_config = SO101FollowerConfig(
        port=_get_env("FOLLOWER_PORT", "/dev/tty.usbmodemFOLLOWER"),
        id=_get_env("FOLLOWER_ID", "FOLLOWER"),
        cameras=cameras,
    )
    robot = SO101Follower(robot_config)

    teleop_config = SO101LeaderConfig(
        port=_get_env("LEADER_PORT", "/dev/tty.usbmodemLEADER"),
        id=_get_env("LEADER_ID", "LEADER"),
    )
    teleop = SO101Leader(teleop_config)

    init_rerun(session_name="teleoperation")

    print("Connecting robot and teleoperator...")
    robot.connect()
    teleop.connect()

    print("Teleoperation with cameras started")
    try:
        while True:
            action = teleop.get_action()
            robot.send_action(action)

            obs = robot.get_observation()
            log_rerun_data(observation=obs, action=action)

    except KeyboardInterrupt:
        print("
Stopping teleoperation...")

    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
