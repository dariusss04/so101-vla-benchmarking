# Teleoperate an SO101 follower using an SO100 leader (no cameras).
# Configure ports/IDs via environment variables before running.

import os
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def main():
    robot_config = SO101FollowerConfig(
        port=_get_env("FOLLOWER_PORT", "/dev/tty.usbmodemFOLLOWER"),
        id=_get_env("FOLLOWER_ID", "FOLLOWER"),
    )
    robot = SO101Follower(robot_config)

    teleop_config = SO100LeaderConfig(
        port=_get_env("LEADER_PORT", "/dev/tty.usbmodemLEADER"),
        id=_get_env("LEADER_ID", "LEADER"),
    )
    teleop = SO100Leader(teleop_config)

    print("Connecting robot and teleoperator...")
    robot.connect()
    teleop.connect()

    print("Teleoperation started")
    try:
        while True:
            action = teleop.get_action()
            robot.send_action(action)

    except KeyboardInterrupt:
        print("
Stopping teleoperation...")

    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
