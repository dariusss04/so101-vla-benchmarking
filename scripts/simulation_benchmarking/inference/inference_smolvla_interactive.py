# Interactive keyboard-driven rollout for testing LeRobot policies in the CALVIN simulator.

import os
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from pytorch_lightning import seed_everything
import sys

def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)

def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val

DATASET_PATH = _get_required("DATASET_PATH")
TRAIN_FOLDER = _get_required("TRAIN_FOLDER")
DEVICE = _get_env("DEVICE", "cuda")
MAX_STEPS = int(_get_env("MAX_STEPS", "500"))

# Ensure root inference/simulation_benchmarking is in path to find inference_smolvla
sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "inference/simulation_benchmarking"))

try:
    from inference_smolvla import CustomModel, make_env
except ImportError:
    print(f"Error: Could not find inference_smolvla in inference/simulation_benchmarking")
    sys.exit(1)

logger = logging.getLogger(__name__)

def interactive_rollout(model, env, initial_lang_goal, max_steps=500):
    print("\n" + "="*50)
    print("🤖 INTERACTIVE LEROBOT ROLLOUT")
    print(f"Goal: '{initial_lang_goal}'")
    print("="*50)
    print("Controls (Make sure OpenCV window is focused):")
    print(" [t] : Pause and Type a new text instruction")
    print(" [n] : End current rollout")
    print("="*50)

    obs = env.reset()
    lang_goal = initial_lang_goal
    
    cv2.namedWindow("LeRobot Calvin Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LeRobot Calvin Agent", 600, 600)

    for step in range(max_steps):
        # 1. Get Action from LeRobot CustomModel wrapper
        action_tuple = model.step(obs, lang_goal)
        
        # LeRobot wrapper returns (relative_pos, relative_euler, gripper_action)
        action_array = np.concatenate([action_tuple[0], action_tuple[1], [action_tuple[2]]])
        
        # 2. Step physics
        obs, _, _, current_info = env.step(action_array)

        # 3. Visualize & Handle Input
        img = env.render(mode="rgb_array")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.putText(img_bgr, f"Goal: {lang_goal}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("LeRobot Calvin Agent", img_bgr)
        k = cv2.waitKey(1) % 256

        if k == ord('t'):
            print("\n[PAUSED] Enter new command:")
            new_goal = input("> ")
            if new_goal.strip():
                lang_goal = new_goal.strip()
                print(f"Goal updated to: '{lang_goal}'")
                
        elif k == ord('n') or k == 27: # 'n' or ESC
            print("\n[STOPPING ROLLOUT]")
            break

    cv2.destroyAllWindows()


def main():
    seed_everything(0, workers=True)

    # Load Model
    print(f"Loading model from {TRAIN_FOLDER}...")
    model = CustomModel(checkpoint_dir=TRAIN_FOLDER, device=DEVICE)

    # Make Environment
    print(f"Loading environment from {DATASET_PATH}...")
    env = make_env(DATASET_PATH)

    while True:
        lang_goal = input("\nEnter initial instruction (or 'q' to quit): ")
        if lang_goal.lower() == 'q':
            break
            
        model.reset()
        interactive_rollout(model, env, lang_goal, max_steps=MAX_STEPS)

    print("Exiting...")

if __name__ == "__main__":
    main()
