# AGENTS.md

## Project
Pixel art generative model. Comparing VAE / VQ-VAE / Diffusion via strategy pattern.
Framework: PyTorch (MPS backend on Apple Silicon). No JAX. No MLX.

## Architecture Invariant
Trainer is the context. Strategies implement GenerativeStrategy ABC (`src/strategies/base.py`).
Adding a new strategy must NOT touch Trainer, logging, or data pipeline.
Model building blocks (ResBlock, attention, encoder, decoder, U-Net) live in `src/models/` — strategies compose them.

## PyTorch Conventions
- Standard training loop: `optimizer.zero_grad()` → forward → `loss.backward()` → `optimizer.step()`.
- Use `torch.no_grad()` for inference/sampling.
- Device management: Trainer detects MPS/CUDA/CPU and moves model + data.

## MLflow
- Experiment per strategy. Run per config.
- Log palette-snapped sample grids, not raw output.
- Wrap MLflow in `src/mlflow_logger.py` — Trainer never imports mlflow directly.

## Commands
```bash
uv run python scripts/train.py --config configs/vae.yaml
uv run python scripts/smoke_test.py
mlflow ui
```

## Code Style
- Type hints. No `Any`.
- Config via OmegaConf.
- `logging` module, no `print()`.

## After Model Code Changes
Run `uv run python scripts/smoke_test.py` to verify strategy contracts.

## Dev Docs
- `.agents/dev_docs/active/<task>/` — plan, context, tasks files per task
- `.agents/dev_docs/archive/` — completed tasks
- `.agents/dev_docs/decisions.md` — cumulative cross-task design decisions

## Tutorial Docs
This is a learning project. Every meaningful engineering decision gets a tutorial-style explanation in `docs/learn/`.
- One doc per major concept, numbered in build order
- Structure: **What** → **Why here** (pixel art specific) → **How** (annotated code) → **Gotchas**
- When adding a new strategy or making a non-trivial design choice, update or create the relevant doc
- These are the project's primary learning artifact — keep them honest and detailed
