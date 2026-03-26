#!/usr/bin/env bash
# Calibrate an SO101 leader using lerobot-calibrate.
# More flags are available; check with:
#   lerobot-calibrate --help

set -euo pipefail

LEADER_PORT=${LEADER_PORT:-/dev/tty.usbmodemLEADER}
LEADER_ID=${LEADER_ID:-LEADER}

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}"
