# SmolVLA training on converted CALVIN datasets with true-resume and optional Hub push.

import os
import logging
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins

def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)

def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val

DATASET_PATH = _get_required("DATASET_PATH")
OUTPUT_DIR = Path(_get_env("OUTPUT_DIR", "outputs/train/smolvla_model"))
PRETRAINED_MODEL = os.getenv("PRETRAINED_MODEL", None)
BATCH_SIZE = int(_get_env("BATCH_SIZE", "8"))
STEPS = int(_get_env("STEPS", "120000"))
LR = float(_get_env("LR", "1e-4"))
LOG_EVERY = int(_get_env("LOG_EVERY", "50"))
SAVE_EVERY = int(_get_env("SAVE_EVERY", "2000"))
KEEP_CHECKPOINTS = int(_get_env("KEEP_CHECKPOINTS", "6"))
SEED = int(_get_env("SEED", "42"))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cleanup_old_checkpoints(root: Path, keep: int):
    ckpts = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()], 
                   key=lambda x: int(x.name))
    if len(ckpts) > keep:
        for folder in ckpts[:-keep]:
            logging.info(f"Cleanup : Deleting {folder.name}")
            shutil.rmtree(folder)

def find_latest_checkpoint_dir(root: Path):
    if not root.exists(): return None, 0
    ckpts = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
    if not ckpts: return None, 0
    latest = max(ckpts, key=lambda p: int(p.name))
    return latest, int(latest.name)

def save_all(policy, pre, post, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    pre.save_pretrained(out_dir / "preprocessor")
    post.save_pretrained(out_dir / "postprocessor")

def save_training_state(out_dir: Path, step: int, optimizer: torch.optim.Optimizer):
    state = {"step": step, "optimizer": optimizer.state_dict(), "rng_torch": torch.get_rng_state()}
    torch.save(state, out_dir / "training_state.pt")

def load_training_state(ckpt_dir: Path, optimizer: torch.optim.Optimizer):
    p = ckpt_dir / "training_state.pt"
    if not p.exists(): return 0
    state = torch.load(p, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_torch"])
    return int(state.get("step", 0))

def main():
    register_third_party_plugins()
    init_logging()
    set_seed(SEED)

    device = get_device()
    logging.info(f"Using device : {device}")

    # Initialisation Dataset
    is_local_dataset = Path(DATASET_PATH).is_dir()

    if is_local_dataset:
        logging.info(f"Loading local dataset from: {DATASET_PATH}")
        ds0 = LeRobotDataset("local", root=DATASET_PATH)
    else:
        logging.info(f"Loading HF Hub dataset from: {DATASET_PATH}")
        ds0 = LeRobotDataset(repo_id=DATASET_PATH, download_videos=True)

    features = dataset_to_policy_features(ds0.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, device=str(device))
    cfg.use_amp = True

    delta_timestamps = {"action": [i / ds0.meta.fps for i in range(int(cfg.n_action_steps))]}
    
    if is_local_dataset:
        dataset = LeRobotDataset("local", root=DATASET_PATH, delta_timestamps=delta_timestamps)
    else:
        dataset = LeRobotDataset(repo_id=DATASET_PATH, delta_timestamps=delta_timestamps)

    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume Logic
    resume_ckpt_dir, resume_step = find_latest_checkpoint_dir(output_dir)
    
    if resume_ckpt_dir:
        logging.info(f"Found existing checkpoint! Resuming at step {resume_step} from {resume_ckpt_dir}")
        model_path = resume_ckpt_dir 
        if not (model_path / "config.json").exists():
            logging.error(f"ERROR: config.json not found in {model_path}")
            return
        
        policy = SmolVLAPolicy.from_pretrained(str(model_path), config=cfg).to(device).train()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
        
        # Load optimizer state if available
        state_path = resume_ckpt_dir / "training_state.pt"
        if state_path.exists():
            logging.info(f"Loading optimizer state from {state_path}")
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng_torch"])
        else:
            logging.warning("training_state.pt not found. Optimizer starts from scratch.")
    else:
        if PRETRAINED_MODEL is not None:
            logging.info(f"Initializing from pretrained model: {PRETRAINED_MODEL}")
            policy = SmolVLAPolicy.from_pretrained(PRETRAINED_MODEL, config=cfg).to(device).train()
        else:
            logging.info("Initializing model from scratch.")
            policy = SmolVLAPolicy(cfg).to(device).train()
            
        optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
        resume_step = 0

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_iter = iter(dataloader)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)

    step = resume_step
    logging.info(f"Starting training from step {step} to {STEPS}...")

    while step < STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = pre(batch)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            loss, _ = policy.forward(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            logging.info(f"Step {step}/{STEPS} - Loss: {loss.item():.4f}")

    if step % SAVE_EVERY == 0 or step == STEPS:
        ckpt_dir = output_dir / f"{step:06d}"
        save_all(policy, pre, post, ckpt_dir)
        save_training_state(ckpt_dir, step, optimizer)
        cleanup_old_checkpoints(output_dir, KEEP_CHECKPOINTS)
        
        # Optional: Push to Hub
        HUB_REPO_ID = os.getenv("HUB_REPO_ID", None)
        PUSH_TO_HUB = os.getenv("PUSH_TO_HUB", "false").lower() == "true"
        if HUB_REPO_ID and PUSH_TO_HUB:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=HUB_REPO_ID,
                repo_type="model",
                path_in_repo=f"checkpoints/{step:06d}",
                commit_message=f"Checkpoint {step:06d}",
            )
            logging.info(f"Pushed checkpoint {step} to HF")

    logging.info("Training finished.")

if __name__ == "__main__":
    main()
