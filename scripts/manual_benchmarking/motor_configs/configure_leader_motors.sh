#!/usr/bin/env bash
# Configure SO101 leader motor IDs using lerobot-setup-motors.
# More flags are available; check with:
#   lerobot-setup-motors --help

set -euo pipefail

LEADER_PORT=${LEADER_PORT:-/dev/tty.usbmodemLEADER}

lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}"
