# Calibrate an SO101 follower arm. Configure port/ID via environment variables.

import os
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val


def main():
    follower_config = SO101FollowerConfig(
        port=_get_required("FOLLOWER_PORT"),
        id=_get_required("FOLLOWER_ID"),
    )

    follower = SO101Follower(follower_config)

    print("Connecting to SO101 follower...")
    follower.connect(calibrate=False)

    print("Running calibration for follower...")
    follower.calibrate()

    print("Disconnecting follower.")
    follower.disconnect()


if __name__ == "__main__":
    main()
