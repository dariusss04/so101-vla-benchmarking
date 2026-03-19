#!/usr/bin/env bash
# Configure SO101 follower motor IDs using lerobot-setup-motors.
# More flags are available; check with:
#   lerobot-setup-motors --help

set -euo pipefail

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/tty.usbmodemFOLLOWER}

lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}"
