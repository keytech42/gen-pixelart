# Task 5: Diffusion — Tasks

- [x] Noise scheduler (beta schedule, alpha_bar precomputation, q_sample, p_sample)
- [x] Sinusoidal time embedding
- [x] ResBlock with time conditioning (GroupNorm, SiLU, additive time injection)
- [x] U-Net model (down/mid/up with skip connections, 7.4M params)
- [x] DiffusionStrategy implementing GenerativeStrategy ABC
- [x] Register in train.py + smoke_test.py
- [x] Smoke test passes (all four strategies: dummy, vae, vqvae, diffusion)
- [x] Train on real sprites — noise_pred_mse 1.1→0.02, sharpest samples of all three strategies
- [x] Write 04-diffusion.md tutorial
