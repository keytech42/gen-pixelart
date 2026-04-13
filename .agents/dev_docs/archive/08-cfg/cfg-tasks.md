# Task 8: CFG — Tasks

- [x] Add null class token to U-Net (num_classes+1 embedding)
- [x] Training: random label dropout with p_uncond=0.1
- [x] Sampling: dual forward pass + guidance blending via _guided_noise_pred
- [x] Add guidance_scale, p_uncond to config
- [x] Smoke test passes (all strategies, conditional + unconditional)
- [x] Compared scales 1/3/5/7 — scale 3 is sweet spot for this dataset size
- [x] Write 07-classifier-free-guidance.md tutorial
