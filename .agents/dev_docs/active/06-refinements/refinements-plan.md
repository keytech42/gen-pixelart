# Task 6: Refinements — Plan

## Goal
Fix what's broken, complete what's half-done, optimize what works. Four improvements in priority order, each producing a measurable result.

## Subtasks (in order)

### 6a. BCE loss for VAE
- **Why**: Binary sprites (0 or 1 pixels). MSE gives weak gradients near extremes; BCE gives strong signal for crisp decisions.
- **Scope**: One-line loss change in VAEStrategy, retrain, compare sample grids.
- **Done when**: Side-by-side comparison shows whether BCE improves output crispness.

### 6b. VQ-VAE autoregressive prior
- **Why**: VQ-VAE reconstructions are excellent but sampling is broken (random codebook indices → checkerboard). A prior over codebook index sequences makes it a real generative model.
- **Scope**: Small transformer/GPT trained on the 4x4 index grids from VQ-VAE encoder. Two-stage: freeze VQ-VAE, train prior on its codebook indices.
- **Done when**: VQ-VAE samples from learned prior look coherent (comparable to diffusion quality).

### 6c. DDIM sampling for diffusion
- **Why**: 1000-step sampling takes ~12s for 16 images. DDIM reduces to ~50 steps with near-identical quality.
- **Scope**: Add DDIM sampler to NoiseScheduler, configurable step count.
- **Done when**: 50-step DDIM produces samples visually equivalent to 1000-step DDPM.

### 6d. Colored sprites
- **Why**: Monochrome is limiting. Colored palette (8-16 colors) produces more visually interesting output.
- **Scope**: Switch to Kenney colored tilesheet, update palette extraction, retune configs.
- **Done when**: All three strategies train on colored sprites with coherent palette use.

## Design principle
Each subtask is independently valuable and produces a runnable, verifiable result. If we stop after any one, the project is in a better state than before.
