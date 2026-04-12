# Task 5: Diffusion — Context

## Key files
- `src/models/unet.py` — U-Net with time conditioning (to build)
- `src/models/blocks.py` — ResBlock with time embedding (to build)
- `src/strategies/diffusion.py` — DiffusionStrategy (to build)
- `configs/diffusion.yaml` — timesteps, beta schedule, U-Net channels

## Decisions
_(updated during implementation)_

## Reference
- DDPM: Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
- Forward: q(x_t|x_0) = N(sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
- Reverse: model predicts ε, then x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ) + σ_t * z
