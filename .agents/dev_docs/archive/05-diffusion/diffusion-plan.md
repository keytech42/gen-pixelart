# Task 5: Diffusion (DDPM) — Plan

## Goal
Third and final generative strategy. A lightweight DDPM that iteratively denoises random noise into pixel art sprites. Completes the three-way comparison.

## What we're building
1. **Noise scheduler** — linear beta schedule, precomputed alpha/alpha_bar tables
2. **U-Net** — small U-Net with residual blocks, time embedding, skip connections
3. **DiffusionStrategy** — implements GenerativeStrategy ABC
   - `train_step`: sample timestep, add noise, predict noise, MSE loss
   - `sample`: iterative denoising loop (T steps from pure noise)
   - `get_metrics`: noise prediction MSE at random timesteps

## Key concepts
- **Forward process**: gradually add Gaussian noise over T timesteps until image → pure noise
- **Reverse process**: learn to denoise one step at a time, starting from pure noise
- **The model predicts noise, not images**: given noisy image x_t and timestep t, predict the noise ε that was added
- **Loss**: MSE between predicted noise and actual noise

## Architecture
```
U-Net (input: noisy image + time embedding)
  Down: Conv(3→64) → ResBlock(64) → ↓ → ResBlock(128) → ↓ → ResBlock(256)
  Mid:  ResBlock(256) → ResBlock(256)
  Up:   ResBlock(256) → ↑ → ResBlock(128) → ↑ → ResBlock(64)
  Out:  Conv(64→3)

Time embedding: sinusoidal positional encoding → MLP → added to each ResBlock
```

For 16x16 images this U-Net will be small (~2-4M params). No attention (overkill at this resolution).

## Design choices
- Linear beta schedule (simple, well-understood)
- T=1000 timesteps for training, can use fewer for fast sampling later
- No attention layers — 16x16 is too small to benefit
- Predict noise (ε-prediction), not x_0 or v

## Done when
- Smoke test passes with all three strategies
- Three-way comparison on MLflow dashboard
- Diffusion samples should be the sharpest of the three
