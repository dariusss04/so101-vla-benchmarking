# Inference (with stats)

These scripts run physical inference **and** log per‑trial metrics into CSV files. Some metrics are computed automatically, while others are entered manually after each run.

- `inference_act_with_stats.py` — ACT inference with CSV logging.
- `inference_smolvla_with_stas.py` — SmolVLA inference with CSV logging.

The CSV column order is defined in `configs/csv_schema.yaml`. All paths, IDs, and camera settings are provided via environment variables.
