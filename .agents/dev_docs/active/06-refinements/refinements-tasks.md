# Task 6: Refinements — Tasks

## 6a. BCE loss for VAE
- [x] Switch VAEStrategy loss from MSE to BCE
- [x] Retrain 500 epochs on focused dataset
- [x] Compare BCE vs MSE sample grids — BCE produces sharper edges, better silhouettes, more defined shapes
- [x] Update 02-vae.md tutorial with findings

## 6b. VQ-VAE autoregressive prior
- [x] Encode full dataset through frozen VQ-VAE → collect 4x4 index grids (873 × 16 sequences)
- [x] Build IndexPrior (4-layer GPT, 828K params) over 16-token index sequences
- [x] Train prior — CE loss 4.03 → 0.45 in 300 epochs (~40 seconds)
- [x] Update VQVAEStrategy.sample() to use learned prior when available
- [x] Comparison: prior samples are coherent sprites; random indices are checkerboard noise
- [x] Update 03-vqvae.md tutorial

## 6c. DDIM sampling
- [x] Implement ddim_sample in NoiseScheduler (eta parameter, timestep subsequence)
- [x] Add sampling_method + sampling_steps to config and DiffusionStrategy
- [x] Compare: DDIM 50 steps (0.83s) vs DDPM 1000 steps (13.6s) — 16.4x speedup, comparable quality
- [x] Update 04-diffusion.md tutorial with DDIM section and speed/quality table
- [x] Smoke test passes with DDIM path

## 6d. Colored sprites
- [x] Slice Kenney colored tilesheet (colored-transparent_packed.png → 1077 sprites, 8-color palette)
- [x] Filter to 873 focused sprites (same density filter)
- [x] VAE loss made configurable (MSE for colored, BCE for binary) via config
- [x] Update configs (palette_size=8, data path)
- [x] Updated download_data.py to support both mono and colored variants
- [x] Retrain all three strategies at 1000 epochs — diffusion produces coherent colored sprites
- [ ] Update 01-data-pipeline.md tutorial
- [ ] Update README with colored samples
