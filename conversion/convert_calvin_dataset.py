# Convert raw CALVIN dataset folders into unified LeRobot format with video and parquet output.

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import imageio
from tqdm import tqdm

from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.datasets.utils import (
    create_empty_dataset_info,
    write_info,
    write_tasks,
    write_stats,
)

def _get_required(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required env var: {name}")
    return val

# ============================================================
# Video stats helper (GLOBAL, MULTI-EPISODE)
# ============================================================
def compute_video_stats_from_mp4s(
    mp4_paths: list[Path],
    *,
    sample_every: int = 30,
    max_total_samples: int = 5000,
) -> Dict[str, np.ndarray]:
    """
    Compute global RGB stats over multiple videos by sampling frames.
    Output format matches LeRobot expectations.
    """
    samples = []

    for mp4_path in mp4_paths:
        reader = imageio.get_reader(str(mp4_path))
        try:
            for i, frame in enumerate(reader):
                if i % sample_every != 0:
                    continue
                frame = np.asarray(frame, dtype=np.float32) / 255.0
                samples.append(frame.reshape(-1, 3).mean(axis=0))
                if len(samples) >= max_total_samples:
                    break
        finally:
            reader.close()

        if len(samples) >= max_total_samples:
            break

    if not samples:
        arr = np.zeros((1, 3), dtype=np.float32)
    else:
        arr = np.stack(samples, axis=0)

    def ch(x):
        return x.astype(np.float32).reshape(3, 1, 1)

    return {
        "min": ch(arr.min(axis=0)),
        "max": ch(arr.max(axis=0)),
        "mean": ch(arr.mean(axis=0)),
        "std": ch(arr.std(axis=0)),
        "count": np.array([arr.shape[0]], dtype=np.int64),
        "q01": ch(np.quantile(arr, 0.01, axis=0)),
        "q10": ch(np.quantile(arr, 0.10, axis=0)),
        "q50": ch(np.quantile(arr, 0.50, axis=0)),
        "q90": ch(np.quantile(arr, 0.90, axis=0)),
        "q99": ch(np.quantile(arr, 0.99, axis=0)),
    }


# ============================================================
# MAIN CONVERSION
# ============================================================
def convert_calvin_dataset_to_lerobot(
    train_dir: str,
    val_dir: str,
    out_dir: str,
    selected_tasks: list[str],
    *,
    fps: int = 30,
    use_rel_actions: bool = False,
    static_key: str = "rgb_static",
    gripper_key: str = "rgb_gripper",
    video_codec: str = "libx264",
    video_crf: int = 18,
    video_preset: str = "slow",
    parquet_compression: str = "snappy",
    mapping: dict[str, str] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Feature names
    # Define these lists based on the specific robot kinematics of your target environment.
    # CALVIN uses 15 dims for the state (EE pos(3), Euler(3), Width(1), Joints(7), Gripper(1)).
    # If your dataset uses a different setup (e.g., quaternions or an ALOHA dual-arm robot), update these strings to match the names required by LeRobot.
    # --------------------------------------------------------
    state_names = [
        "ee_position_x", "ee_position_y", "ee_position_z",
        "ee_orientation_rx", "ee_orientation_ry", "ee_orientation_rz",
        "gripper_width",
        "joint_position_00", "joint_position_01", "joint_position_02",
        "joint_position_03", "joint_position_04", "joint_position_05",
        "joint_position_06",
        "gripper_action",
    ]

    action_names = [
        "ee_position_x", "ee_position_y", "ee_position_z",
        "ee_orientation_rx", "ee_orientation_ry", "ee_orientation_rz",
        "gripper_action",
    ]

    # --------------------------------------------------------
    # Write tasks.parquet
    # --------------------------------------------------------
    mapped_tasks = [mapping.get(t, t) for t in selected_tasks] if mapping else selected_tasks
    
    write_tasks(
        pd.DataFrame({"task_index": range(len(mapped_tasks))},
                     index=pd.Index(mapped_tasks)),
        out_dir,
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------
    (out_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos/observation.images.front/chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos/observation.images.wrist/chunk-000").mkdir(parents=True, exist_ok=True)

    episode_rows = []
    all_episode_stats = []

    episode_index = 0
    global_index = 0
    split_episode_start = {}

    # Default observation shapes (will strictly dynamically update based on the first raw sample read)
    # Adjust `robot_dim` and `action_dim` defaults if you are porting an environment where the first frame is absent or malformed.
    robot_dim = 15
    action_dim = 7
    Hs, Ws = 200, 200
    Hg, Wg = 84, 84
    has_seen_sample = False

    # --------------------------------------------------------
    # TRAIN then VALIDATION
    # --------------------------------------------------------
    for split_name, split_dir in [("train", train_dir), ("validation", val_dir)]:
        split_episode_start[split_name] = episode_index
        split_dir = Path(split_dir)

        if not split_dir.exists():
            continue

        ann_path = split_dir / "lang_annotations/auto_lang_ann.npy"
        if not ann_path.exists():
            print(f"Cannot find annotations at {ann_path}")
            continue

        ann = np.load(
            ann_path,
            allow_pickle=True,
        ).item()

        if "task" in ann["language"]:
            texts = ann["language"]["task"]
        else:
            texts = ann["language"]["ann"]

        ranges = ann["info"]["indx"]

        # Flatten if it's an array of arrays
        if isinstance(texts, np.ndarray) and texts.ndim > 1:
            texts = texts.flatten().tolist()
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], list):
            texts = [item for sub in texts for item in sub]

        ep_ids_path = split_dir / "ep_start_end_ids.npy"
        if not ep_ids_path.exists():
            print(f"Skipping {split_name}, ep_start_end_ids.npy missing")
            continue
            
        ep_ids = np.load(ep_ids_path)
        first_step = int(ep_ids[0, 0])
        
        sample_path = split_dir / f"episode_{first_step:07d}.npz"
        if sample_path.exists() and not has_seen_sample:
            sample = np.load(sample_path)
            Hs, Ws, _ = sample[static_key].shape
            Hg, Wg, _ = sample[gripper_key].shape
            robot_dim = sample["robot_obs"].shape[0]
            action_key = "rel_actions" if (use_rel_actions and "rel_actions" in sample.files) else "actions"
            action_dim = sample[action_key].shape[0]
            has_seen_sample = True
        else:
            action_key = "rel_actions" if use_rel_actions else "actions"

        # ----------------------------------------------------
        # Iterate episodes
        # ----------------------------------------------------
        for txt, (s_id, e_id) in tqdm(
            zip(texts, ranges),
            total=len(texts),
            desc=f"Processing {split_name}",
        ):
            if txt not in selected_tasks:
                continue

            task_index = selected_tasks.index(txt)
            length = int(e_id - s_id + 1)
            file_index = episode_index

            # -------------------------------
            # Video writers (PER EPISODE)
            # We construct MP4 streams iteratively using `imageio` with `ffmpeg` bindings.
            # To optimize storage footprint vs. CPU decode overhead for massive datasets, tweak `video_crf` (compression quality) and `video_preset`.
            # -------------------------------
            front_path = out_dir / "videos/observation.images.front/chunk-000" / f"file-{file_index:03d}.mp4"
            wrist_path = out_dir / "videos/observation.images.wrist/chunk-000" / f"file-{file_index:03d}.mp4"

            wf = imageio.get_writer(
                front_path, fps=fps, codec=video_codec,
                ffmpeg_params=["-crf", str(video_crf), "-preset", video_preset],
                macro_block_size=1,
            )
            ww = imageio.get_writer(
                wrist_path, fps=fps, codec=video_codec,
                ffmpeg_params=["-crf", str(video_crf), "-preset", video_preset],
                macro_block_size=1,
            )

            states, actions = [], []

            for step in range(int(s_id), int(e_id) + 1):
                npz = np.load(split_dir / f"episode_{step:07d}.npz")
                wf.append_data(npz[static_key])
                ww.append_data(npz[gripper_key])
                states.append(npz["robot_obs"].astype(np.float32))
                actions.append(npz[action_key].astype(np.float32))

            wf.close()
            ww.close()

            # -------------------------------
            # Parquet data (PER EPISODE)
            # -------------------------------
            table = pa.Table.from_pydict({
                "observation.state": states,
                "action": actions,
                "timestamp": [i / fps for i in range(length)],
                "frame_index": list(range(length)),
                "episode_index": [episode_index] * length,
                "index": list(range(global_index, global_index + length)),
                "task_index": [task_index] * length,
            })

            pq.write_table(
                table,
                out_dir / "data/chunk-000" / f"file-{file_index:03d}.parquet",
                compression=parquet_compression,
            )

            # -------------------------------
            # Stats
            # -------------------------------
            stats = compute_episode_stats(
                {
                    "observation.state": np.asarray(states),
                    "action": np.asarray(actions),
                    "timestamp": np.asarray([i / fps for i in range(length)]),
                    "frame_index": np.asarray(list(range(length))),
                    "episode_index": np.asarray([episode_index] * length),
                    "index": np.asarray(list(range(global_index, global_index + length))),
                    "task_index": np.asarray([task_index] * length),
                },
                features={
                    "observation.state": {"dtype": "float32", "shape": (robot_dim,), "names": state_names},
                    "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
                    "timestamp": {"dtype": "float32", "shape": (1,)},
                    "frame_index": {"dtype": "int64", "shape": (1,)},
                    "episode_index": {"dtype": "int64", "shape": (1,)},
                    "index": {"dtype": "int64", "shape": (1,)},
                    "task_index": {"dtype": "int64", "shape": (1,)},
                },
            )
            all_episode_stats.append(stats)

            # -------------------------------
            # meta/episodes row
            # -------------------------------
            episode_rows.append({
                "episode_index": episode_index,
                "length": length,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": file_index,
                "videos/observation.images.front/from_timestamp": 0.0,
                "videos/observation.images.front/to_timestamp": length / fps,
                "videos/observation.images.wrist/chunk_index": 0,
                "videos/observation.images.wrist/file_index": file_index,
                "videos/observation.images.wrist/from_timestamp": 0.0,
                "videos/observation.images.wrist/to_timestamp": length / fps,
                "data/chunk_index": 0,
                "data/file_index": file_index,
            })

            global_index += length
            episode_index += 1

    # --------------------------------------------------------
    # Finalize metadata
    # --------------------------------------------------------
    if episode_index == 0:
        print("No valid episodes generated. Aborting metadata creation.")
        return

    pd.DataFrame(episode_rows).to_parquet(
        out_dir / "meta/episodes/chunk-000/file-000.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate stats + robust VIDEO stats
    # LeRobot requires normalization factors (mean, std, min, max, quantiles) over the entire dataset split.
    # If your custom dataset is massive (e.g. > 100GB of video), limit `max_total_samples` here to bound RAM usage.
    # --------------------------------------------------------
    aggregated_stats = aggregate_stats(all_episode_stats)

    # Collect video paths from multiple episodes
    front_videos = sorted(
        (out_dir / "videos/observation.images.front/chunk-000").glob("file-*.mp4")
    )
    wrist_videos = sorted(
        (out_dir / "videos/observation.images.wrist/chunk-000").glob("file-*.mp4")
    )

    aggregated_stats["observation.images.front"] = compute_video_stats_from_mp4s(
        front_videos,
        sample_every=30,
        max_total_samples=5000,
    )

    aggregated_stats["observation.images.wrist"] = compute_video_stats_from_mp4s(
        wrist_videos,
        sample_every=30,
        max_total_samples=5000,
    )

    write_stats(aggregated_stats, out_dir)

    info = create_empty_dataset_info(
        codebase_version="v3.0",
        fps=fps,
        features={
            "observation.images.front": {"dtype": "video", "shape": (Hs, Ws, 3), "names": ["h", "w", "c"]},
            "observation.images.wrist": {"dtype": "video", "shape": (Hg, Wg, 3), "names": ["h", "w", "c"]},
            "observation.state": {"dtype": "float32", "shape": (robot_dim,), "names": state_names},
            "action": {"dtype": "float32", "shape": (action_dim,), "names": action_names},
            "timestamp": {"dtype": "float32", "shape": (1,)},
            "frame_index": {"dtype": "int64", "shape": (1,)},
            "episode_index": {"dtype": "int64", "shape": (1,)},
            "index": {"dtype": "int64", "shape": (1,)},
            "task_index": {"dtype": "int64", "shape": (1,)},
        },
        use_videos=True,
        robot_type="panda",   
    )

    info["total_episodes"] = episode_index
    info["total_frames"] = global_index
    info["total_tasks"] = len(mapped_tasks)

    # Note: validation split boundary might be wrong if there are missing episodes
    if "validation" not in split_episode_start:
        info["splits"] = {
            "train": f"0:{episode_index}"
        }
    else:
        info["splits"] = {
            "train": f"{split_episode_start['train']}:{split_episode_start['validation']}",
            "validation": f"{split_episode_start['validation']}:{episode_index}",
        }

    write_info(info, out_dir)

    (out_dir / "README.md").write_text(
        f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

Converted from CALVIN dataset to LeRobot framework format.

- **Tasks Count:** {len(mapped_tasks)}
- **Total Episodes:** {episode_index}

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{json.dumps(info, indent=2)}
``` 
"""
    )

    print(f"✅ DATASET CONVERSION FINISHED. Found at {out_dir}")

def main():
    DATASET_PATH = _get_required("DATASET_PATH")
    # Default to data/calvin_lerobot in the root if OUT_DIR is not provided
    DEFAULT_OUT = str(Path(__file__).absolute().parents[1] / "data" / "calvin_lerobot")
    OUT_DIR = os.getenv("OUT_DIR", DEFAULT_OUT)
    TASKS_ENV = _get_required("TASKS")

    tasks = json.loads(TASKS_ENV)
    
    HUB_REPO_ID = os.getenv("HUB_REPO_ID", None)
    PUSH_TO_HUB = os.getenv("PUSH_TO_HUB", "false").lower() == "true"
    
    MAPPING_JSON = os.getenv("MAPPING_JSON", None)
    mapping = None
    if MAPPING_JSON:
        with open(MAPPING_JSON, "r") as f:
            mapping = json.load(f)

    train_dir = str(Path(DATASET_PATH) / "training")
    val_dir = str(Path(DATASET_PATH) / "validation")
    
    convert_calvin_dataset_to_lerobot(
        train_dir=train_dir,
        val_dir=val_dir,
        out_dir=OUT_DIR,
        selected_tasks=tasks,
        mapping=mapping
    )

    if HUB_REPO_ID and PUSH_TO_HUB:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"Uploading dataset to {HUB_REPO_ID}...")
        api.create_repo(repo_id=HUB_REPO_ID, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=OUT_DIR,
            repo_id=HUB_REPO_ID,
            repo_type="dataset",
        )
        print(f"✅ Dataset successfully pushed to https://huggingface.co/datasets/{HUB_REPO_ID}")

if __name__ == "__main__":
    main()
