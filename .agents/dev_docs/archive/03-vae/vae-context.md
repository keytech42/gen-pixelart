# Task 3: VAE — Context

## Key files
- `src/models/encoder.py` — Encoder (to build)
- `src/models/blocks.py` — shared building blocks if needed
- `src/strategies/vae.py` — VAEStrategy (to build)
- `configs/vae.yaml` — latent_dim, encoder/decoder channels, kl_weight

## Decisions
_(updated during implementation)_

## Reference
- VAE paper: Kingma & Welling 2013, "Auto-Encoding Variational Bayes"
- Key formula: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
- Reparameterization: z = mu + sigma * epsilon, epsilon ~ N(0,1)
