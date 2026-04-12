# Pixel Art Generative Model — Handover Document

## 1. Project Overview

A **learning-oriented** project to train small generative models that produce aesthetic pixel art sprites (Aseprite-style). The core engineering goal is comparing multiple generative architectures through a strategy pattern, with experiment tracking via MLflow — all running natively on Apple Silicon via MLX.

This is a study toy project. Prioritize clarity and learning value over production polish.

### Why pixel art is tractable

Pixel art sprites are typically 16×16 to 64×64 with palettes of 4–32 colors. A 32×32 sprite with 16 colors is a categorical grid — orders of magnitude simpler than natural image generation. Models in the low millions of parameters can produce genuinely aesthetic results. A 5M-param diffusion model on an M3 Max should converge in hours.

The hard aesthetic challenge: pixel artists are deliberate about every single pixel. The model should produce sprites with clean outlines, readable silhouettes, and coherent shading — not plausible-looking blobs.

---

## 2. Architecture Design

### 2.1 Strategy Pattern

The central design decision. Three generative architectures share identical training infrastructure with only model logic swapped.

**Strategy contract (ABC):**

```
GenerativeStrategy
├── build_model(config) → model params
├── train_step(params, batch, rng) → (loss_dict, updated_params)
├── sample(params, n_samples, rng) → images
└── get_metrics(params, batch) → dict
```

**Context class:** `Trainer` owns the training loop, calls strategy methods, never knows which generative approach is running. Adding a new strategy must NOT require changes to Trainer, logging, or data pipeline.

**Why `loss_dict`:** Each strategy returns differently-named losses:
- VAE → `recon_loss`, `kl_loss`
- VQ-VAE → `recon_loss`, `codebook_loss`, `commitment_loss`
- Diffusion → `noise_prediction_mse`

MLflow logs them all without the Trainer needing to interpret them.

**Model composition:** Strategies *compose* model components — a diffusion strategy uses a U-Net, a VQ-VAE uses encoder + codebook + decoder. Model building blocks (ResBlock, attention, encoder, decoder, U-Net) live separately from strategies to enable reuse without duplication.

### 2.2 Three strategies

**VAE (implement first):**
Simplest training loop. Reparameterization trick + KL divergence. Gets the full pipeline (data → train → sample → MLflow) working end-to-end fastest. Output will be blurry but recognizable.

**VQ-VAE + autoregressive prior:**
Learns a codebook of visual motifs (256–512 entries). More natural for pixel art — mirrors how artists think in repeated patterns and palette-constrained blocks. Two-stage: train VQ-VAE, then train a small transformer prior over codebook indices.

**Diffusion (DDPM):**
Lightweight U-Net (~4–8M params). Best output quality. Noise scheduling + iterative denoising. Key trick: discretize output by snapping to nearest palette color after denoising.

### 2.3 MLX conventions

- Train step uses `mx.value_and_grad` — strategies return pure functions.
- Use `mx.compile` on hot paths. Never mutate arrays in compiled functions.
- Model params are nested dicts (MLX's `nn.Module` pattern), not PyTorch state_dicts.
- No PyTorch. No JAX. MLX only.

---

## 3. MLflow Integration

### Experiment structure

| Level | Maps to | Purpose |
|---|---|---|
| Experiment | Strategy name (`vae` / `vqvae` / `diffusion`) | Compare runs within an architecture |
| Run | Specific hyperparameter config | One training session |
| Artifact | Sample grids, model checkpoints | Visual + reproducibility tracking |

### What to log

- **Metrics:** All loss components from `loss_dict`, logged per step or epoch.
- **Params:** Full config (learning rate, latent dim, palette size, image resolution, model size).
- **Artifacts at intervals:** Sample image grids — **palette-snapped**, not raw model output. The raw output may look like mush while the discretized version already looks like passable pixel art. That gap is itself informative.
- **Final artifacts:** Trained model params, final sample grid, config YAML.

### MLflow logger

Wrap MLflow in a thin `MLflowLogger` class so the Trainer doesn't import mlflow directly. This keeps the strategy pattern clean and makes the logging backend swappable (if ever needed).

No `print()` for training progress — use Python's `logging` module, optionally forwarded to MLflow.

---

## 4. Data

### Sources

- OpenGameArt, itch.io sprite sheets, Aseprite community assets
- Scope narrowly at first: e.g., "16×16 RPG character sprites" — a few hundred to low thousands of examples is enough for small models

### Pipeline

- Sprite loading with palette extraction (auto-detect or specify palette size)
- Palette quantization utilities (snap continuous RGB → nearest palette color)
- Augmentation: palette swaps, horizontal flips. No random crops or color jitter (these break pixel art semantics).

### Dataset class

Should yield `(image_tensor, palette, label)` where label is optional (for future conditioning work).

---

## 5. Project Structure

```
pixel-gen/
├── AGENTS.md                   # Project context for agent sessions (CLAUDE.md is a symlink to this)
├── CLAUDE.md -> AGENTS.md      # Symlink
├── configs/
│   ├── base.yaml               # Shared: dataset path, image_size, palette_size, epochs, log interval
│   ├── vae.yaml
│   ├── vqvae.yaml
│   └── diffusion.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py          # Sprite loading, palette extraction
│   │   └── augmentation.py     # Palette swap, horizontal flip
│   ├── strategies/
│   │   ├── base.py             # GenerativeStrategy ABC
│   │   ├── vae.py
│   │   ├── vqvae.py
│   │   └── diffusion.py
│   ├── models/
│   │   ├── unet.py             # Shared U-Net backbone
│   │   ├── encoder.py
│   │   ├── codebook.py         # Vector quantization layer
│   │   └── blocks.py           # ResBlock, attention, conv blocks
│   ├── trainer.py              # Context class — training loop
│   ├── mlflow_logger.py        # MLflow wrapper
│   └── palette.py              # Color quantization utilities
├── scripts/
│   ├── train.py                # Entry point: load config → pick strategy → train
│   ├── sample.py               # Generate from a logged model run
│   ├── smoke_test.py           # Fast validation (see §7)
│   └── compare_runs.py         # Pull MLflow data, generate comparison grids
├── notebooks/
│   └── exploration.ipynb
└── .agents/
    └── dev_docs/
        ├── active/             # In-progress task docs
        │   └── <task>/
        │       ├── <task>-plan.md
        │       ├── <task>-context.md
        │       └── <task>-tasks.md
        ├── archive/            # Completed task docs
        └── decisions.md        # Cross-task cumulative design decisions
```

---

## 6. AGENTS.md / CLAUDE.md

`CLAUDE.md` is a **symlink** to `AGENTS.md`:

```bash
ln -s AGENTS.md CLAUDE.md
```

Content lives in `AGENTS.md`. Claude Code reads it via the symlink.

### Content guidance

The example below is a **starting-point sketch, not a prescription**. Claude Code should refine this toward simplicity and robustness as the project takes shape — restructure sections, adjust conventions, add or remove rules based on what actually helps during implementation. The goal is a file that makes each session productive, not a comprehensive specification.

```markdown
# AGENTS.md (example — refine for simplicity and robustness)

## Project
Pixel art generative model. Comparing VAE / VQ-VAE / Diffusion via strategy pattern.
Framework: MLX. No PyTorch.

## Architecture Invariant
Trainer is the context. Strategies implement GenerativeStrategy ABC (src/strategies/base.py).
Adding a new strategy must NOT touch Trainer, logging, or data pipeline.

## MLX
- mx.value_and_grad for train steps. Pure functions.
- mx.compile on hot paths. No array mutation in compiled functions.

## MLflow
- Experiment per strategy. Run per config.
- Log palette-snapped sample grids, not raw output.

## Commands
- python scripts/train.py --config configs/vae.yaml
- python scripts/smoke_test.py
- mlflow ui

## Code style
- Type hints. No Any.
- Config via OmegaConf.
- logging module, no print().
```

---

## 7. Smoke Test

`scripts/smoke_test.py` — instantiates each implemented strategy, runs one `train_step` on a random batch, calls `sample`, and verifies output shapes. Fast feedback loop (~seconds) that catches breakages without a full training run.

**Mention this file in AGENTS.md.** Claude Code should run it after any model code change.

---

## 8. Dev Docs & decisions.md

### .agents/dev_docs/ (three-file context)

Each task gets a directory under `active/`:
- `<task>-plan.md` — the accepted plan
- `<task>-context.md` — key files, decisions, blockers
- `<task>-tasks.md` — checklist of work

Completed tasks move to `archive/`.

### decisions.md (cumulative, cross-task)

ML projects accumulate design decisions that aren't task-scoped. Examples:
- "Switched from MSE to L1 loss for diffusion because of palette snapping artifacts"
- "Codebook size 256 is sufficient for 16-color sprites; 512 needed for 32-color"
- "mx.compile breaks on the VQ straight-through estimator — left that path uncompiled"

This file is the project's institutional memory across sessions and branches. Update it whenever a non-obvious decision is made.

---

## 9. Task Decomposition

Each task should produce a **runnable state** — something that executes and can be verified.

| Task | Scope | Runnable outcome |
|---|---|---|
| 1. skeleton | Project structure, ABC, config loading, dummy strategy, Trainer, MLflow wiring | `train.py` runs with dummy strategy, logs to MLflow |
| 2. data | Sprite dataset loader, palette extraction, augmentation | Loads real data, logs sample batch to MLflow |
| 3. vae | First real strategy end-to-end | Generates blurry but recognizable sprites |
| 4. vqvae | Second strategy, codebook mechanics | Two-strategy comparison on MLflow dashboard |
| 5. diffusion | Noise scheduler, U-Net, sampling loop | Three-way comparison on MLflow |
| 6. conditioning | Class labels or palette conditioning across strategies | Conditional generation |

Each task maps to roughly one Claude Code session. Dev docs carry context between sessions.

---

## 10. Claude Code Hooks (optional, recommended)

**Pre-commit:** If any file in `src/strategies/` is modified, run `smoke_test.py`. Prevents a strategy from silently breaking the ABC contract.

**Post-task:** Prompt to update `decisions.md` and record any MLflow experiment/run IDs created during the session.

---

## 11. Key Risks & Pitfalls

- **PyTorch-isms creeping in.** MLX's functional style is different. Watch for `.backward()`, `optimizer.step()`, `torch.no_grad()` patterns — none of these exist in MLX.
- **Strategy boundary violations.** If the Trainer starts checking `isinstance(strategy, DiffusionStrategy)`, the pattern is broken. Refactor the ABC instead.
- **Raw vs. palette-snapped evaluation.** Always evaluate and log the snapped output. Raw floating-point images are misleading for pixel art quality.
- **Data quality > model size.** A curated dataset of 500 high-quality sprites will outperform 5000 scraped low-quality ones. Invest time in curation.
- **Aesthetic gap.** "Generates something" is easy. "Generates something intentional-looking" is hard. Clean outlines, readable silhouettes, and coherent shading clusters are the real benchmarks — track these visually in MLflow, not just loss curves.
