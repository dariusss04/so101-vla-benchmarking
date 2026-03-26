#!/usr/bin/env bash
# Record teleoperated episodes using lerobot_record.py.
# More flags are available; check with:
#   python -m lerobot.scripts.lerobot_record --help

set -euo pipefail

RECORD_SCRIPT=${RECORD_SCRIPT:-python -m lerobot.scripts.lerobot_record}

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/tty.usbmodemFOLLOWER}
LEADER_PORT=${LEADER_PORT:-/dev/tty.usbmodemLEADER}
FOLLOWER_ID=${FOLLOWER_ID:-FOLLOWER}
LEADER_ID=${LEADER_ID:-LEADER}

FRONT_CAM_INDEX=${FRONT_CAM_INDEX:-0}
WRIST_CAM_INDEX=${WRIST_CAM_INDEX:-1}
CAM_WIDTH=${CAM_WIDTH:-640}
CAM_HEIGHT=${CAM_HEIGHT:-480}
CAM_FPS=${CAM_FPS:-30}

DATASET_REPO_ID=${DATASET_REPO_ID:-your-username/your-dataset}
DATASET_ROOT=${DATASET_ROOT:-./data/your-dataset}
TASK_DESCRIPTION=${TASK_DESCRIPTION:-"Your task"}
NUM_EPISODES=${NUM_EPISODES:-1}
RESUME=${RESUME:-true}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}

${RECORD_SCRIPT} \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="{\
    front: {type: opencv, index_or_path: ${FRONT_CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}},\
    wrist: {type: opencv, index_or_path: ${WRIST_CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}}\
  }" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${TASK_DESCRIPTION}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --resume="${RESUME}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}"
