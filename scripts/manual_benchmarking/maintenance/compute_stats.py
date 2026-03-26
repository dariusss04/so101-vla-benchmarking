# Compute dataset-wide statistics and write meta/stats.json for a local LeRobot dataset.

import os
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env var: {name}")
    return value


# --- Configuration ---
HF_REPO_ID = _get_env("DATASET_REPO_ID", "your-username/your-dataset")
DATASET_ROOT = Path(_get_env("DATASET_ROOT", "./data"))
LOCAL_NAME = _get_env("DATASET_NAME", "your-dataset")
STATS_PATH = Path(_get_env("STATS_PATH", str(DATASET_ROOT / LOCAL_NAME / "meta" / "stats.json")))


def format_image_stat(vals):
    return [[[float(v)]] for v in vals]


def compute_precise_stats(dataset):
    features = dataset.features
    stats = {}
    key_order = [
        "action",
        "observation.state",
        "observation.images.front",
        "observation.images.wrist",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]

    num_frames = len(dataset)
    print(f"Processing {num_frames} samples from {HF_REPO_ID}...")

    for key in key_order:
        if key not in features:
            continue
        print(f" -> Computing: {key}")
        is_video = features[key]["dtype"] == "video"

        if is_video:
            sample_data = dataset[0][key]
            if sample_data.ndim == 4:
                num_channels = sample_data.shape[1]
                chan_dim = 1
            else:
                num_channels = sample_data.shape[0]
                chan_dim = 0

            c_min = torch.full((num_channels,), float("inf"))
            c_max = torch.full((num_channels,), float("-inf"))
            c_sum = torch.zeros((num_channels,), dtype=torch.float64)
            c_sum_sq = torch.zeros((num_channels,), dtype=torch.float64)
            hists = torch.zeros((num_channels, 256), dtype=torch.int64)

            for i in tqdm(range(num_frames), desc=f"Scanning {key}", leave=False):
                img = dataset[i][key]
                img_f = img.float()
                img_b = (img_f * 255).clamp(0, 255).to(torch.uint8)

                for c in range(num_channels):
                    if chan_dim == 1:
                        chan_f = img_f[:, c].flatten()
                        chan_b = img_b[:, c].flatten()
                    else:
                        chan_f = img_f[c].flatten()
                        chan_b = img_b[c].flatten()
                    c_min[c] = min(c_min[c], chan_f.min())
                    c_max[c] = max(c_max[c], chan_f.max())
                    c_sum[c] += chan_f.sum()
                    c_sum_sq[c] += (chan_f ** 2).sum()
                    hists[c] += torch.bincount(chan_b, minlength=256)

            N = float(hists[0].sum())
            mean = c_sum / N
            std = torch.sqrt((c_sum_sq / N) - (mean ** 2)).clamp(min=1e-8)

            def get_q(h, q_lvl):
                target = q_lvl * h.sum()
                cum = torch.cumsum(h, dim=0)
                return torch.searchsorted(cum, target).float() / 255.0

            q_lvls = [0.01, 0.10, 0.50, 0.90, 0.99]
            qs = [[get_q(hists[c], q) for c in range(num_channels)] for q in q_lvls]

            stats[key] = {
                "min": format_image_stat(c_min),
                "max": format_image_stat(c_max),
                "mean": format_image_stat(mean),
                "std": format_image_stat(std),
                "count": [num_frames],
                "q01": format_image_stat(qs[0]),
                "q10": format_image_stat(qs[1]),
                "q50": format_image_stat(qs[2]),
                "q90": format_image_stat(qs[3]),
                "q99": format_image_stat(qs[4]),
            }
        else:
            data = torch.stack([dataset[i][key] for i in range(num_frames)]).float()
            if data.ndim == 1:
                data = data.unsqueeze(1)

            d_min = data.min(0)[0].tolist()
            d_max = data.max(0)[0].tolist()
            d_mean = data.mean(0).tolist()
            d_std = data.std(0).tolist()
            d_qs = torch.quantile(data, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]), dim=0)

            def fmt_q(q_row):
                val = q_row.tolist()
                return val[0] if len(val) == 1 else val

            stats[key] = {
                "min": d_min,
                "max": d_max,
                "mean": d_mean,
                "std": d_std,
                "count": [num_frames],
                "q01": fmt_q(d_qs[0]),
                "q10": fmt_q(d_qs[1]),
                "q50": fmt_q(d_qs[2]),
                "q90": fmt_q(d_qs[3]),
                "q99": fmt_q(d_qs[4]),
            }

        gc.collect()

    return stats


def main():
    dataset = LeRobotDataset(HF_REPO_ID, root=DATASET_ROOT)
    final_stats = compute_precise_stats(dataset)

    STATS_PATH = Path(STATS_PATH)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(final_stats, f, indent=4)

    print(f"\n Success! Stats file written to: {STATS_PATH}")


if __name__ == "__main__":
    main()
