# Single-step isolated task evaluation for CALVIN pipeline using LeRobot models.

from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
import importlib.util

# Ensure root inference/simulation_benchmarking is in path to find inference_smolvla
sys.path.insert(0, str(Path(__file__).absolute().parents[3] / "inference/simulation_benchmarking"))

# Inherit the exact same LeRobot setup logic
try:
    from inference_smolvla import CustomModel, make_env
except ImportError:
    print(f"Error: Could not find inference_smolvla in inference/simulation_benchmarking")
    sys.exit(1)

# Import CALVIN utils
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import join_vis_lang, print_and_save, get_log_dir

logger = logging.getLogger(__name__)

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
DEBUG = _get_env("DEBUG", "false").lower() == "true"
EVAL_LOG_DIR = os.getenv("EVAL_LOG_DIR", None)
TARGET_TASKS_ENV = os.getenv("TARGET_TASKS", None)
EPISODES_PER_TASK = int(_get_env("EPISODES_PER_TASK", "10"))
EP_LEN = int(_get_env("EP_LEN", "240"))
NUM_SEQUENCES = int(_get_env("NUM_SEQUENCES", "5000"))


def evaluate_policy_singlestep(model, env, eval_log_dir=None, debug=False, target_tasks=None, ep_len=240, num_sequences=5000):
    """
    Evaluates the model on exactly one isolated subtask at a time.
    """
    # Try to find calvin_models/conf dynamically
    calvin_conf_env = os.getenv("CALVIN_CONF")
    if calvin_conf_env:
        conf_dir = Path(calvin_conf_env)
    else:
        # Fallback to package location if installed as a package
        spec = importlib.util.find_spec("calvin_models")
        if spec and spec.origin:
            conf_dir = Path(spec.origin).parent / "conf"
        else:
            # Last fallback to the old third_party location relative to root
            conf_dir = Path(__file__).absolute().parents[3] / "third_party/calvin/calvin_models/conf"

    if not conf_dir.exists():
        logger.warning(f"CALVIN conf directory not found at {conf_dir}. Please set CALVIN_CONF env var.")

    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    ALL_SEQS = get_sequences(num_sequences)
    
    # Select sequences that start with our target tasks
    eval_sequences = []
    
    if target_tasks is None:
        target_tasks = {"lift_pink_block_table": 10, "lift_blue_block_table": 10}
        
    counts = {k: 0 for k in target_tasks.keys()}
    
    for init, seq in ALL_SEQS:
        if len(seq) >= 1:
            first_task = seq[0]
            if first_task in target_tasks and counts[first_task] < target_tasks[first_task]:
                eval_sequences.append((init, first_task))
                counts[first_task] += 1
                
        if all(counts[k] >= target_tasks[k] for k in target_tasks):
            break

    print("\n===== ISOLATED SEQUENCES =====")
    for task, count in counts.items():
        print(f" - {task:25s} : {count} / {target_tasks[task]}")
    print("==============================\n")

    results = Counter()

    for init_state, task in eval_sequences:
        # Get language annotation for the task
        lang_annotation = val_annotations[task][0]

        # Reset env to exact initial state
        obs = env.reset(robot_obs=init_state["robot_obs"][0], scene_obs=init_state["scene_obs"][0])
        model.reset()
        start_info = env.get_info()

        success = False
        for step in range(ep_len):
            action = model.step(obs, lang_annotation)
            obs, _, _, current_info = env.step(action)
            
            if debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
            if len(current_task_info) > 0:
                if debug:
                    print(colored("S", "green"), end=" ")
                success = True
                break
                
        if not success and debug:
            print(colored("F", "red"), end=" ")

        if success:
            results[task] += 1

    print("\n\n===== FINAL SINGLE-STEP RESULTS =====")
    for task in target_tasks.keys():
        success_rate = (results[task] / target_tasks[task]) * 100 if target_tasks[task] > 0 else 0
        print(f"{task:25s}: {results[task]}/{target_tasks[task]} ({success_rate:.1f}%)")
        
    print_and_save(results, eval_sequences, {"results": results}, eval_log_dir)
    return results

def main():
    seed_everything(0, workers=True)
    
    target_tasks = None
    if TARGET_TASKS_ENV:
        import json
        tasks_list = json.loads(TARGET_TASKS_ENV)
        target_tasks = {t: EPISODES_PER_TASK for t in tasks_list}
        
    print("Initializing CustomModel for LeRobot...")
    model = CustomModel(checkpoint_dir=TRAIN_FOLDER, device=DEVICE)
    
    print("Loading PyBullet Environment...")
    env = make_env(DATASET_PATH)

    print("Beginning Single-Step evaluation...")
    evaluate_policy_singlestep(
        model, 
        env, 
        eval_log_dir=EVAL_LOG_DIR, 
        debug=DEBUG, 
        target_tasks=target_tasks,
        ep_len=EP_LEN,
        num_sequences=NUM_SEQUENCES
    )

if __name__ == "__main__":
    main()
