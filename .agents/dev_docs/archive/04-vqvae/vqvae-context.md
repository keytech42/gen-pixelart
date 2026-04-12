# Task 4: VQ-VAE — Context

## Key files
- `src/models/codebook.py` — VectorQuantizer (to build)
- `src/models/encoder.py` — reuse ConvEncoder, ConvDecoder from Task 3
- `src/strategies/vqvae.py` — VQVAEStrategy (to build)
- `configs/vqvae.yaml` — codebook_size, codebook_dim, commitment_weight

## Decisions
_(updated during implementation)_

## Reference
- VQ-VAE paper: van den Oord et al. 2017, "Neural Discrete Representation Learning"
- Straight-through estimator: gradients of decoder input copied to encoder output
- Loss: recon + ||sg[z_e] - e||^2 + beta * ||z_e - sg[e]||^2
