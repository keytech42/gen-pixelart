# Task 1: Skeleton — Context

## Key files
- `src/strategies/base.py` — GenerativeStrategy ABC
- `src/strategies/dummy.py` — DummyStrategy + DummyModel
- `src/trainer.py` — Trainer context class, training loop, sample grid saving
- `src/mlflow_logger.py` — MLflow wrapper
- `src/palette.py` — snap_to_palette, generate_default_palette
- `src/config.py` — OmegaConf config loader with defaults merging
- `scripts/train.py` — entry point with strategy registry
- `scripts/smoke_test.py` — strategy contract validation

## Decisions made
- Switched from MLX to PyTorch after weighing tradeoffs (see decisions.md)
- Strategy `train_step` receives `(model, optimizer, batch)` — strategy owns the backward pass
- Uses `uv` for dependency management (`uv add` / `uv run`)

## Notes
- MLflow warns about deprecated filesystem backend — can switch to sqlite later if needed
- DummyModel is a single linear layer that memorizes the training set (loss → 0.0) — expected behavior, confirms pipeline works
