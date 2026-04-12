# Task 1: Skeleton — Plan

## Goal
Set up full project structure and get a dummy strategy running end-to-end through the training pipeline with MLflow logging.

## Design choices
- Strategy pattern with ABC (`GenerativeStrategy`) as the central abstraction
- Trainer is the context class — never knows which strategy is running
- OmegaConf for config with base + strategy YAML merging
- Thin MLflow wrapper (`MLflowLogger`) keeps mlflow out of Trainer
- DummyStrategy validates the full pipeline before real models arrive

## Runnable outcome
- `uv run python scripts/smoke_test.py` — passes for dummy strategy
- `uv run python scripts/train.py --config configs/vae.yaml --strategy-override dummy` — completes 100 epochs, logs metrics + sample grids to MLflow
