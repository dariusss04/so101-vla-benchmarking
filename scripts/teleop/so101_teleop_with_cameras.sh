#!/usr/bin/env bash
# Teleoperate an SO101 follower with cameras using lerobot-teleoperate.
# More flags are available; check with:
#   lerobot-teleoperate --help

set -euo pipefail

TELEOP_PORT=${TELEOP_PORT:-/dev/tty.usbmodemLEADER}
ROBOT_PORT=${ROBOT_PORT:-/dev/tty.usbmodemFOLLOWER}
TELEOP_ID=${TELEOP_ID:-LEADER}
ROBOT_ID=${ROBOT_ID:-FOLLOWER}

FRONT_CAM_INDEX=${FRONT_CAM_INDEX:-0}
WRIST_CAM_INDEX=${WRIST_CAM_INDEX:-1}
CAM_WIDTH=${CAM_WIDTH:-640}
CAM_HEIGHT=${CAM_HEIGHT:-480}
CAM_FPS=${CAM_FPS:-30}
CAM_WARMUP_S=${CAM_WARMUP_S:-2}

lerobot-teleoperate \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{\
    front: {\
      type: opencv,\
      index_or_path: ${FRONT_CAM_INDEX},\
      width: ${CAM_WIDTH},\
      height: ${CAM_HEIGHT},\
      fps: ${CAM_FPS},\
      warmup_s: ${CAM_WARMUP_S}\
    },\
    wrist: {\
      type: opencv,\
      index_or_path: ${WRIST_CAM_INDEX},\
      width: ${CAM_WIDTH},\
      height: ${CAM_HEIGHT},\
      fps: ${CAM_FPS},\
      warmup_s: ${CAM_WARMUP_S}\
    }\
  }" \
  --display_data=true
