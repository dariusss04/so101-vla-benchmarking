# Configure motor IDs for an SO101 leader arm. Configure port/ID via environment variables.

import os
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig


def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val


def main():
    leader = SO101Leader(
        SO101LeaderConfig(
            port=_get_required("LEADER_PORT"),
            id=_get_required("LEADER_ID"),
        )
    )

    print("Connecting to SO101 leader...")
    leader.connect(calibrate=False)

    print("Running motor setup for leader...")
    leader.setup_motors()

    print("Disconnecting leader.")
    leader.disconnect()


if __name__ == "__main__":
    main()
