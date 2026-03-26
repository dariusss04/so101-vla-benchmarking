# ACT training with checkpoint push and true-resume support (optimizer/scheduler state).
# Configure dataset/model/output via environment variables before running.

import os
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.train_utils import (
    save_checkpoint,
    load_training_state,
    get_step_identifier,
)


def _get_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val


# ==========================
# CONFIG (env-driven)
# ==========================
RESUME = _get_env("RESUME", "true").lower() == "true"
HF_REPO_ID = _get_required("HF_REPO_ID")
DATASET_REPO_ID = _get_required("DATASET_REPO_ID")
DATASET_LOCAL_ROOT = Path(_get_required("DATASET_LOCAL_ROOT"))
OUTPUT_DIR = Path(_get_required("OUTPUT_DIR"))

BATCH_SIZE = int(_get_env("BATCH_SIZE", "16"))
STEPS = int(_get_env("STEPS", "100000"))
SAVE_EVERY = int(_get_env("SAVE_EVERY", "10000"))
LOG_EVERY = int(_get_env("LOG_EVERY", "50"))
SEED = int(_get_env("SEED", "42"))
NUM_WORKERS = int(_get_env("NUM_WORKERS", "4"))
USE_AMP = _get_env("USE_AMP", "true").lower() == "true"
VIDEO_BACKEND = _get_env("VIDEO_BACKEND", "pyav")
RESUME_CACHE_DIR = Path(_get_env("RESUME_CACHE_DIR", "./resume_ckpt"))


# ==========================
# HELPERS
# ==========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_delta_timestamps(cfg, fps):
    return {"action": [i / fps for i in cfg.action_delta_indices]}


def find_latest_checkpoint_step(api: HfApi, repo_id: str) -> int | None:
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    steps = set()
    for f in files:
        if f.startswith("checkpoints/") and "/pretrained_model/" in f:
            parts = f.split("/")
            if len(parts) >= 3 and parts[1].isdigit():
                steps.add(int(parts[1]))
    return max(steps) if steps else None


def download_checkpoint(repo_id: str, step: int, local_dir: Path) -> Path:
    step_id = f"{step:06d}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=[f"checkpoints/{step_id}/**"],
        local_dir_use_symlinks=False,
    )
    return local_dir / "checkpoints" / step_id


def push_checkpoint(api: HfApi, checkpoint_dir: Path, step: int):
    step_id = f"{step:06d}"
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        path_in_repo=f"checkpoints/{step_id}",
        commit_message=f"Checkpoint {step_id}",
    )


# ==========================
# MAIN
# ==========================

def main():
    register_third_party_plugins()
    init_logging()
    set_seed(SEED)

    device = get_device()
    logging.info(f"Using device: {device}")

    api = HfApi()

    ds0 = LeRobotDataset(DATASET_REPO_ID, root=DATASET_LOCAL_ROOT, video_backend=VIDEO_BACKEND)
    fps = ds0.meta.fps

    features = dataset_to_policy_features(ds0.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    start_step = 0
    if RESUME:
        latest_step = find_latest_checkpoint_step(api, HF_REPO_ID)
        if latest_step is None:
            raise RuntimeError("No checkpoints found on HF repo to resume from.")
        logging.info(f"Resuming from HF checkpoint step {latest_step}")
        checkpoint_dir = download_checkpoint(HF_REPO_ID, latest_step, RESUME_CACHE_DIR)
        pretrained_dir = checkpoint_dir / "pretrained_model"

        policy = ACTPolicy.from_pretrained(str(pretrained_dir)).to(device).train()
        policy.config.device = device.type
        policy.config.use_amp = USE_AMP

        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained_dir),
            preprocessor_overrides={"device_processor": {"device": device.type}},
            postprocessor_overrides={"device_processor": {"device": device.type}},
        )

        optimizer_cfg = policy.config.get_optimizer_preset()
        scheduler_cfg = policy.config.get_scheduler_preset()

        class DummyCfg:
            use_policy_training_preset = True
            optimizer = optimizer_cfg
            scheduler = scheduler_cfg
            steps = STEPS

        optimizer, lr_scheduler = make_optimizer_and_scheduler(DummyCfg, policy)
        start_step, optimizer, lr_scheduler = load_training_state(checkpoint_dir, optimizer, lr_scheduler)

    else:
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            use_vae=True,
        )
        cfg.device = device.type
        cfg.use_amp = USE_AMP

        pre, post = make_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)
        policy = ACTPolicy(cfg).to(device).train()

        optimizer_cfg = policy.config.get_optimizer_preset()
        scheduler_cfg = policy.config.get_scheduler_preset()

        class DummyCfg:
            use_policy_training_preset = True
            optimizer = optimizer_cfg
            scheduler = scheduler_cfg
            steps = STEPS

        optimizer, lr_scheduler = make_optimizer_and_scheduler(DummyCfg, policy)
        start_step = 0

    delta_timestamps = make_delta_timestamps(policy.config, fps)
    ds = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_LOCAL_ROOT,
        delta_timestamps=delta_timestamps,
        video_backend=VIDEO_BACKEND,
    )

    dataloader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting training at step {start_step}")

    use_amp = device.type == "cuda" and policy.config.use_amp
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    step = start_step
    dl_iter = iter(dataloader)
    start_time = time.perf_counter()

    while step < STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = pre(batch)
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            loss, _ = policy(batch)

        if not torch.isfinite(loss):
            logging.error(f"NaN/Inf loss at step {step}")
            break

        loss.backward()
        if optimizer_cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), optimizer_cfg.grad_clip_norm)

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        step += 1

        if step % LOG_EVERY == 0:
            logging.info(
                f"step={step}/{STEPS} loss={loss.item():.4f} time_s={time.perf_counter() - start_time:.1f}"
            )

        if step % SAVE_EVERY == 0 or step == STEPS:
            checkpoint_dir = OUTPUT_DIR / "checkpoints" / get_step_identifier(step, STEPS)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=type("Cfg", (), {
                    "save_pretrained": lambda self, d: policy.config.save_pretrained(d),
                    "peft": None,
                    "save_checkpoint": True
                })(),
                policy=policy,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                preprocessor=pre,
                postprocessor=post,
            )
            logging.info(f"Saved checkpoint at step {step}: {checkpoint_dir}")
            push_checkpoint(api, checkpoint_dir, step)
            logging.info(f"Pushed checkpoint {step} to HF")

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    pre.save_pretrained(final_dir / "preprocessor")
    post.save_pretrained(final_dir / "postprocessor")
    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Final model",
    )
    logging.info("Training finished + final model pushed.")


if __name__ == "__main__":
    main()
