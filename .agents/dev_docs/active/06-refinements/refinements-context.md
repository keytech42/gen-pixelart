# Task 6: Refinements — Context

## Key files
- `src/strategies/vae.py` — BCE loss change (6a)
- `src/models/codebook.py` — index extraction for prior (6b)
- `src/strategies/vqvae.py` — prior integration (6b)
- `src/strategies/diffusion.py` — DDIM sampler (6c)
- `scripts/download_data.py` — colored tilesheet (6d)
- `configs/base.yaml` — palette_size updates (6d)

## Decisions
_(updated during implementation)_

## Current state
- VAE uses MSE loss, produces blurry output on binary sprites
- VQ-VAE codebook utilization ~15%, reconstructions good, sampling broken
- Diffusion is best sampler but slow (1000 steps)
- All trained on 873 focused monochrome 16x16 sprites
