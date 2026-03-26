# Multistep CALVIN simulation evaluation pipeline for SmolVLA LeRobot models.

from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import cv2  # Used for local video rendering and logging
import importlib.util

def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)

def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val

DATASET_PATH = _get_required("DATASET_PATH")
TRAIN_FOLDER = _get_required("TRAIN_FOLDER")
CHECKPOINT = os.getenv("CHECKPOINT", None)
DEBUG = _get_env("DEBUG", "false").lower() == "true"
EVAL_LOG_DIR = os.getenv("EVAL_LOG_DIR", None)
DEVICE = _get_env("DEVICE", "cuda")

RECORD_VIDEO_DIR = os.getenv("RECORD_VIDEO_DIR", None)
RECORD_MAX_VIDEOS = int(_get_env("RECORD_MAX_VIDEOS", "10"))

EVAL_MATRIX_ENV = os.getenv("EVAL_MATRIX", None)
EP_LEN = int(_get_env("EP_LEN", "90"))
NUM_SEQUENCES = int(_get_env("NUM_SEQUENCES", "5000"))

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.configs.types import PolicyFeature, FeatureType
import json
from safetensors.torch import load_file
# --- EVALUATION MATRIX ---
logger = logging.getLogger(__name__)

def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=True)
    return env

def normalize_features(features: dict | None):
    if features is None:
        return None

    out = {}
    for name, ft in features.items():
        if isinstance(ft, PolicyFeature):
            out[name] = ft
        elif isinstance(ft, dict):
            out[name] = PolicyFeature(
                type=FeatureType(ft["type"]),
                shape=tuple(ft.get("shape", ())),
            )
        else:
            raise TypeError(f"Unsupported feature type for {name}: {type(ft)}")
    return out


class CustomModel(CalvinBaseModel):
    def __init__(self, checkpoint_dir, device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)

        # -------------------------
        # 1. Load preprocessors
        # -------------------------
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(checkpoint_dir / "preprocessor"),
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(checkpoint_dir / "postprocessor"),
            config_filename="policy_postprocessor.json",
            overrides={"device_processor": {"device": "cpu"}},
        )

        # -------------------------
        # 2. Load and clean config
        # -------------------------
        with open(checkpoint_dir / "config.json", "r") as f:
            cfg_dict = json.load(f)

        import inspect
        valid_args = inspect.signature(SmolVLAConfig.__init__).parameters.keys()
        cleaned_cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_args}

        config = SmolVLAConfig(**cleaned_cfg_dict)
        config.input_features = normalize_features(config.input_features)
        config.output_features = normalize_features(config.output_features)

        # -------------------------
        # 3. Instantiate policy and load weights
        # -------------------------
        self.policy = SmolVLAPolicy(config)
        self.policy.to(self.device)

        state_dict = load_file(
            str(checkpoint_dir / "model.safetensors"),
            device=str(self.device),
        )
        self.policy.load_state_dict(state_dict, strict=False)

        self.policy.eval()
        self.reset()

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def step(self, obs, goal):
        # Image scaling: LeRobot expects [0, 1] for VLMs
        img_front = torch.from_numpy(obs["rgb_obs"]["rgb_static"]).permute(2, 0, 1).float() / 255.0
        img_wrist = torch.from_numpy(obs["rgb_obs"]["rgb_gripper"]).permute(2, 0, 1).float() / 255.0

        batch = {
            "observation.images.front": img_front,
            "observation.images.wrist": img_wrist,
            "observation.state": torch.from_numpy(obs["robot_obs"]).float(),
            "task": goal,
        }

        model_dtype = next(self.policy.parameters()).dtype

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(self.device, dtype=model_dtype)
            else:
                batch[k] = v

        batch = self.preprocessor(batch)
        
        # The model's state_proj might only expect 6-dim (pos + euler)
        # Slicing after normalization to ensure dimension match
        if batch["observation.state"].shape[-1] > 6:
            batch["observation.state"] = batch["observation.state"][:, :6]

        # Query policy for action
        action = self.policy.select_action(batch)
        
        # Post-process model output back into robot action space
        action = self.postprocessor({"action": action})["action"]
        action = action.squeeze(0).cpu().numpy()
        
        # SmolVLA models operating in Action Chunking mode (n_action_steps > 1) output a sequence of actions. 
        # For strict reactive evaluation in simulators like PyBullet, we select only the very first action (t=0) and discard the rest. 
        # To generalize to other environments, adapt the slice index based on your action execution frequency.
        if action.ndim == 2:
            action = action[0]

        return (action[:3], action[3:6], 1 if action[6] > 0 else -1)



def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False, record_obj=None, eval_matrix=None, ep_len=90, num_sequences=5000):
    """
    Run this function to evaluate a model on the CALVIN challenge.
    record_obj is a dict containing {"video_dir": path, "max_videos": int, "num_recorded": 0}
    """
    if eval_matrix is None:
        eval_matrix = {
            ("lift_pink_block_table", "stack_block"): 5, 
            ("lift_blue_block_table", "stack_block"): 5, 
            ("lift_pink_block_table", "*"): 5,
            ("lift_blue_block_table", "*"): 5,
        }

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
            conf_dir = Path(__file__).absolute().parents[2] / "third_party/calvin/calvin_models/conf"

    if not conf_dir.exists():
        logger.warning(f"CALVIN conf directory not found at {conf_dir}. Please set CALVIN_CONF env var.")

    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    ALL_SEQS = get_sequences(num_sequences)
    
    eval_sequences = []
    counts = {k: 0 for k in eval_matrix.keys()}
    
    for init, seq in ALL_SEQS:
        if len(seq) >= 2:
            # Attempt exact pair match or fallback to wildcard
            current_pair = (seq[0], seq[1])
            wildcard_pair = (seq[0], "*")
            
            # Priority 1: Exact match
            matched_key = None
            if current_pair in eval_matrix and counts[current_pair] < eval_matrix[current_pair]:
                matched_key = current_pair
            # Priority 2: Wildcard match (e.g., if any second task satisfies the matrix requirement)
            elif wildcard_pair in eval_matrix and counts[wildcard_pair] < eval_matrix[wildcard_pair] and current_pair not in eval_matrix:
                matched_key = wildcard_pair
                
            if matched_key is not None:
                # Cache sequence and the matched tracking key for JSON reporting
                eval_sequences.append((init, [seq[0], seq[1]], matched_key))
                counts[matched_key] += 1
                
        # Terminate search when evaluation matrix quotas are fulfilled
        if all(counts[k] >= eval_matrix[k] for k in eval_matrix):
            break

    print("\n===== EVALUATION MATRIX SEQUENCES =====")
    for pair, count in counts.items():
        print(f" - {pair[0]} -> {pair[1]:15s} : {count} / {eval_matrix[pair]}")
    print("==============================\n")

    results = []
    plans = defaultdict(list)
    per_task_results = defaultdict(list)
    
    # Track results per specific task pair to output detailed structured JSON metrics
    matrix_results = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence, matched_key) in enumerate(eval_sequences):
        task_name = eval_sequence[0]
        # We use matched_key (e.g., ("lift_pink_block_table", "*")) instead of the exact pair
        pair_key = matched_key

        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence,
            val_annotations, plans, debug, record_obj, ep_len=ep_len
        )

        results.append(result)
        matrix_results[pair_key].append(result)

        # success = 1 if solved, 0 if failed
        success = 1 if result >= 1 else 0
        per_task_results[task_name].append(success)

        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    print("\n===== PER-TASK SUCCESS RATES =====")
    for task, successes in per_task_results.items():
        rate = 100 * np.mean(successes) if len(successes) > 0 else 0.0
        print(f"{task:30s} : {rate:5.1f}%  ({len(successes)} episodes)")
    print("=================================\n")

    # Calculate statistics and save structured JSON readout
    final_json_data = {}
    for pair, res_list in matrix_results.items():
        n_total = len(res_list)
        sr_task1 = sum(1 for r in res_list if r >= 1) / n_total if n_total > 0 else 0.0
        sr_task2 = sum(1 for r in res_list if r >= 2) / n_total if n_total > 0 else 0.0
        
        # JSON requires string keys: format as "task1, task2"
        key_str = f"{pair[0]}, {pair[1]}"
        final_json_data[key_str] = [n_total, sr_task1, sr_task2]

    # Save results
    if eval_log_dir is not None:
        with open(Path(eval_log_dir) / "per_task_results.json", "w") as f:
            json.dump(final_json_data, f, indent=2)

    legacy_eval_sequences = [(init, seq) for init, seq, _ in eval_sequences]
    print_and_save(results, legacy_eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, record_obj=None, ep_len=90):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
        
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, record_obj, ep_len=ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, record_obj=None, ep_len=90):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    # --- VIDEO RECORDING SETUP ---
    video_writer = None
    if record_obj is not None and record_obj["num_recorded"] < record_obj["max_videos"]:
        video_dir = Path(record_obj["video_dir"])
        video_dir.mkdir(parents=True, exist_ok=True)
        video_filename = video_dir / f"eval_{record_obj['num_recorded']:03d}_{subtask.replace(' ', '_')}.mp4"
        
        # Render a throwaway frame strictly to acquire dynamic HxWxC dimensions for cv2.VideoWriter. 
        # This natively generalizes video recording to any environment resolution.
        img_np = env.render(mode="rgb_array")
        h, w, c = img_np.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        video_writer = cv2.VideoWriter(str(video_filename), fourcc, 30.0, (w, h))

    for step in range(ep_len):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        
        # Render visualization and append to video stream if recording is enabled
        if debug or video_writer is not None:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            
            # cv2 expects BGR format cleanly converted from RGB
            if video_writer is not None:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)
                
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            if video_writer is not None:
                video_writer.release()
                record_obj["num_recorded"] += 1
            return True

    if debug:
        print(colored("fail", "red"), end=" ")
    if video_writer is not None:
        video_writer.release()
        record_obj["num_recorded"] += 1
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    
    eval_matrix = None
    if EVAL_MATRIX_ENV:
        matrix_list = json.loads(EVAL_MATRIX_ENV)
        eval_matrix = {}
        for item in matrix_list:
            eval_matrix[(item[0], item[1])] = int(item[2])
    
    record_obj = None
    if RECORD_VIDEO_DIR:
        record_obj = {
            "video_dir": RECORD_VIDEO_DIR,
            "max_videos": RECORD_MAX_VIDEOS,
            "num_recorded": 0
        }

    # evaluate a custom model (we assume True for Custom LeRobot models)
    model = CustomModel(
        checkpoint_dir=CHECKPOINT if CHECKPOINT else TRAIN_FOLDER,
        device=DEVICE,
    )
    env = make_env(DATASET_PATH)
    evaluate_policy(
        model,
        env,
        epoch="custom",
        eval_log_dir=EVAL_LOG_DIR,
        debug=DEBUG,
        record_obj=record_obj,
        eval_matrix=eval_matrix,
        ep_len=EP_LEN,
        num_sequences=NUM_SEQUENCES
    )


if __name__ == "__main__":
    main()
