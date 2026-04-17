# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end autonomous vehicle planning system built on the NuPlan/OpenScene dataset. Implements multiple planning agents (TransFuser, Diffusion Planner, GTRS) using deep learning for trajectory prediction.

## Dependency Policy

Do not install any dependency without explicit user permission. This includes pip, uv add, conda, etc. Use `uv` to run Python scripts (e.g., `uv run python script.py`).

## Commands

```bash
# Training
uv run python navsim/planning/script/run_training_dense.py

# Evaluation (PDM Score)
uv run python navsim/planning/script/run_pdm_score.py

# Metric caching
uv run python navsim/planning/script/run_metric_caching.py
```

No formal test suite exists. Testing is done via Jupyter notebooks in each phase directory.

## Required Environment Variables

```
NUPLAN_MAPS_ROOT        # NuPlan maps path
OPENSCENE_DATA_ROOT     # OpenScene dataset path
NAVSIM_EXP_ROOT         # Experiment output directory
```

## Architecture

The project is organized into 4 phases plus a core framework:

- **phase-1 dataset/** — Dataset loading and validation
- **phase-2 model/** — Model implementations and experimentation (most active)
- **phase-3 eval/** — Evaluation notebooks and utilities
- **phase-4 training/** — Training scripts
- **navsim/** — Core framework (agents, planning pipeline, data utilities, evaluation)
- **nuplan-devkit/** — External NuPlan toolkit (git submodule, do not modify)
- **model_weights/** — Pre-trained checkpoints

### Agent System

All agents extend `navsim/agents/abstract_agent.py` (`AbstractAgent`, a PyTorch Module) with methods: `forward()`, `compute_trajectory()`, `get_feature_builders()`, `get_target_builders()`, `compute_loss()`, `get_optimizers()`.

Key agent implementations:
- **TransFuser** (`navsim/agents/transfuser/`) — Multi-modal camera+LiDAR fusion with GPT-based cross-attention and ResNet34 backbones. Primary baseline with pre-trained weights.
- **Diffusion Planner** (`phase-2 model/diffusion_planner.py`) — Diffusion model for trajectory generation using DPM Solver.
- **GTRS** (`navsim/agents/gtrs_dense/`, `gtrs_aug/`) — Graph-based trajectory reasoning.

### Data Flow

```
Sensor Data (Camera/LiDAR) → Feature Builders → Agent Model → Trajectory Prediction → PDM Score Evaluation
```

Core data structures in `navsim/common/dataclasses.py`: `AgentInput`, `Scene`, `EgoStatus`, `Trajectory`, `Camera`, `Lidar`.

### Training Infrastructure

- PyTorch Lightning via `navsim/planning/training/agent_lightning_module.py`
- Hydra config system — configs in `navsim/planning/script/config/`
- Scene loading via `navsim/common/dataloader.py` with filtering support

### Development Workflow

Phase directories contain Jupyter notebooks for interactive experimentation. New models are prototyped in phase-2, evaluated in phase-3, and training scripts go in phase-4.
