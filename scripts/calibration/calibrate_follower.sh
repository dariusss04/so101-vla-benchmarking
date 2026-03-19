#!/usr/bin/env bash
# Calibrate an SO101 follower using lerobot-calibrate.
# More flags are available; check with:
#   lerobot-calibrate --help

set -euo pipefail

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/tty.usbmodemFOLLOWER}
FOLLOWER_ID=${FOLLOWER_ID:-FOLLOWER}

lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}"
