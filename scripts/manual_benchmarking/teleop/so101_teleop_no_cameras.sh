#!/usr/bin/env bash
# Teleoperate an SO101 follower without cameras using lerobot-teleoperate.
# More flags are available; check with:
#   lerobot-teleoperate --help

set -euo pipefail

TELEOP_PORT=${TELEOP_PORT:-/dev/tty.usbmodemLEADER}
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodemFOLLOWER}
TELEOP_ID=${TELEOP_ID:-LEADER}
ROBOT_ID=${ROBOT_ID:-FOLLOWER}

lerobot-teleoperate \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}"
