# Task 6: Refinements — Tasks

## 6a. BCE loss for VAE
- [x] Switch VAEStrategy loss from MSE to BCE
- [x] Retrain 500 epochs on focused dataset
- [x] Compare BCE vs MSE sample grids — BCE produces sharper edges, better silhouettes, more defined shapes
- [ ] Update 02-vae.md tutorial with findings

## 6b. VQ-VAE autoregressive prior
- [x] Encode full dataset through frozen VQ-VAE → collect 4x4 index grids (873 × 16 sequences)
- [x] Build IndexPrior (4-layer GPT, 828K params) over 16-token index sequences
- [x] Train prior — CE loss 4.03 → 0.45 in 300 epochs (~40 seconds)
- [x] Update VQVAEStrategy.sample() to use learned prior when available
- [x] Comparison: prior samples are coherent sprites; random indices are checkerboard noise
- [ ] Update 03-vqvae.md tutorial

## 6c. DDIM sampling
- [ ] Implement DDIM sampler in NoiseScheduler
- [ ] Add sampling_steps config option
- [ ] Compare 50-step DDIM vs 1000-step DDPM visually
- [ ] Update 04-diffusion.md tutorial

## 6d. Colored sprites
- [ ] Slice Kenney colored tilesheet
- [ ] Update palette extraction for multi-color
- [ ] Retune configs (palette_size, potentially model capacity)
- [ ] Retrain all three strategies
- [ ] Update 01-data-pipeline.md tutorial
