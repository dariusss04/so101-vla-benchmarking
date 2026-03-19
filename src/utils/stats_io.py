# Read/write helpers for dataset stats.json files.

import json
from pathlib import Path
from typing import Any


def load_stats(stats_path: Path) -> dict[str, Any]:
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")
    return json.loads(stats_path.read_text())


def write_stats(stats_path: Path, stats: dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=4))
