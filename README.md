# Benchmarking VLA Models on a Physical SO101 Robot and in Simulation

This repository contains our end-to-end benchmarking framework for **Vision-Language-Action (VLA)** models in two complementary settings:

- **Manual benchmarking on a physical SO101 robot**
- **Simulation benchmarking based on CALVIN, translated into LeRobot-compatible datasets and evaluation pipelines**

The project is motivated by a simple problem: **VLA evaluation is hard to standardize**. Purely manual real-world benchmarking captures the deployment setting, but it is slow, noisy, sensitive to environmental changes, and difficult to reproduce. Simulation benchmarking is more scalable and reproducible, but only useful if the data, task structure, and evaluation protocol are integrated cleanly into the same VLA workflow.

Our goal is therefore twofold:

1. **Quantify the challenges of manual real-world benchmarking**
2. **Build the infrastructure for scalable simulation benchmarking using public long-horizon benchmarks**

## Project Scope

This repository is organized around two benchmarking tracks.

### 1. Manual Benchmarking

The manual track covers:
- teleoperation and dataset recording on the physical robot
- training ACT and SmolVLA models on LeRobot-formatted datasets
- running real-world inference benchmarks
- logging per-trial metrics such as success, completion time, retries, and failure modes

This part of the project uses:
- **Robot:** SO101 leader-follower setup
- **Sensors:** wrist RGB camera + top/front RGB camera
- **Framework:** [LeRobot](https://github.com/huggingface/lerobot)

### 2. Simulation Benchmarking

The simulation track covers:
- inspection of CALVIN task distributions
- conversion of CALVIN episodes into LeRobot-compatible datasets
- training and inference scripts for simulation-based benchmarking
- long-horizon evaluation inside the CALVIN environment

This part of the project uses:
- **Benchmark:** [CALVIN](https://github.com/mees/calvin)
- **Simulation environment:** CALVIN simulator and task suites
- **Bridge layer:** conversion and evaluation tooling that connects CALVIN-style data and evaluation to LeRobot-style training/inference workflows

## Why Both Tracks Matter

The central argument of the project is that **manual and simulation benchmarking serve different purposes**:

- **Manual benchmarking** gives the most faithful picture of real deployment, including camera noise, grasp instability, lighting changes, hardware failures, and reset variability.
- **Simulation benchmarking** enables scale, determinism, repeatability, and systematic variation of conditions that would be prohibitively expensive to evaluate on hardware.

The long-term aim is not to replace one with the other, but to use both to study the **real-to-sim gap** in a structured way.

## Repository Structure

```text
benchmarkingVlas/
  configs/                  # YAML configs for tasks, camera profiles, defaults, CSV schema
  conversion/               # CALVIN inspection + conversion tools into LeRobot format
  data/                     # Dataset metadata, summaries, and download instructions
  docs/                     # Project documentation (currently minimal)
  inference/
    manual_benchmarking/    # Physical inference with metrics/stat logging
    simulation_benchmarking/# Simulation evaluation entry points
  scripts/
    manual_benchmarking/    # Teleop, recording, calibration, training, inference, maintenance
    simulation_benchmarking/# Simulation training and inference scripts
  src/                      # Shared helpers for env handling, metadata, dataset/stats I/O
  requirements.txt          # Manual benchmarking environment
  requirements_simulation.txt # Simulation benchmarking environment
```

## Manual Benchmarking

### What It Includes

The manual benchmarking pipeline covers the full real-world flow:
- hardware calibration
- teleoperation-based demonstration collection
- dataset organization in LeRobot format
- training ACT and SmolVLA policies
- physical inference runs with per-trial CSV logging

### Hardware Setup

The real-world setup is based on:
- **SO101 leader-follower manipulators**
- **Parallel-jaw gripper**
- **Two RGB views**: one wrist camera and one top/front camera
- **Manual teleoperation** for demonstration recording

### Manual Environment Setup

We recommend using a dedicated Python environment for the physical robot setup.

```bash
python -m venv .venv-manual
source .venv-manual/bin/activate
pip install -r requirements.txt
```

This environment is centered around **LeRobot v0.4.4**. If you want to install the upstream framework directly as well:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.4.4
pip install -e .
```

### Manual Benchmarking Entry Points

The manual workflow lives primarily in:
- `scripts/manual_benchmarking/`
- `inference/manual_benchmarking/`
- `data/`
- `configs/`

Key components include:
- teleoperation scripts
- recording scripts
- training scripts for ACT and SmolVLA
- inference scripts with statistics collection

### Real-World Limitations

Manual benchmarking is important, but it is also where most practical problems appear:
- bandwidth and hardware instability
- camera timeouts and control loop timing issues
- lighting sensitivity
- object placement variability
- reset inconsistency across trials
- substantial manual effort for evaluation and annotation

These limitations are a major reason why simulation benchmarking is part of the project.

## Simulation Benchmarking

### What It Includes

The simulation side extends the project from physical benchmarking to **scalable and reproducible evaluation**.

It includes:
- CALVIN task inspection and filtering
- dataset conversion into LeRobot-compatible format
- simulation-side training/inference scripts
- structured evaluation inside the CALVIN environment

### Benchmark Used

We use **CALVIN**, a long-horizon language-conditioned robot manipulation benchmark.

Links:
- CALVIN code: [github.com/mees/calvin](https://github.com/mees/calvin)
- CALVIN project/paper page: [oiermees.com/publication/calvin](https://www.oiermees.com/publication/calvin/)

CALVIN is relevant here because it provides:
- long-horizon task sequences
- language instructions
- multi-view observations
- a standardized simulator evaluation protocol

### Simulation Environment Setup

The simulation environment depends on **CALVIN itself** as well as the additional packages used by the bridging scripts in this repository.

Recommended setup:

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
conda create -n calvin_venv python=3.8
conda activate calvin_venv
sh install.sh
```

Then install the simulation-side Python requirements for this repository:

```bash
pip install -r /path/to/benchmarkingVlas/requirements_simulation.txt
```

### Simulation Pipeline in This Repository

The simulation flow is organized into three layers:

1. **Inspect CALVIN tasks**
   - determine task frequencies and filter usable subsets
2. **Convert CALVIN into LeRobot-compatible datasets**
   - reconstruct episodes from annotated segments
   - map observations/actions/metadata into the LeRobot schema
3. **Train and evaluate VLA models in simulation**
   - use simulation-specific training and inference entry points

Relevant folders:
- `conversion/`
- `scripts/simulation_benchmarking/`
- `inference/simulation_benchmarking/`

### Current Limitation of the Simulation Track

At the moment, the simulation pipeline and dataset conversion infrastructure are in place, but **the currently converted CALVIN subset is too small for a fair task-level model comparison**. In particular, the available per-task episode count is not yet sufficient to train strong policies for a rigorous simulation-vs-real comparison.

This means the simulation part of the repository currently contributes primarily as:
- an evaluation infrastructure layer
- a dataset translation pipeline
- a reproducible foundation for future large-scale benchmarking

## Datasets

The manual datasets are stored in LeRobot format and documented in `data/`.

You can preview LeRobot datasets visually with the official viewer:
- [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)

The simulation data flow is based on raw CALVIN data plus our conversion pipeline in `conversion/`.

## Configs and Shared Utilities

- `configs/` centralizes tasks, camera profiles, defaults, and CSV schema definitions
- `src/` contains shared helpers for environment handling, metadata, and dataset/statistics I/O

These files are intended to keep the scripts portable and to avoid hard-coded local paths wherever possible.

## Installation Summary

### Manual Benchmarking

```bash
python -m venv .venv-manual
source .venv-manual/bin/activate
pip install -r requirements.txt
```

### Simulation Benchmarking

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin
conda create -n calvin_venv python=3.8
conda activate calvin_venv
sh install.sh
pip install -r /path/to/benchmarkingVlas/requirements_simulation.txt
```

## Notes on Paths and Environment Variables

Scripts in this repository are designed to avoid hard-coded machine-specific settings where possible. Ports, dataset locations, repository IDs, and output folders are generally passed via environment variables and documented in the folder-specific READMEs.

## License

MIT License — see [LICENSE](LICENSE).
