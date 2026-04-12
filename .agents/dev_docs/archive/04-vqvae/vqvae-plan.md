# Task 4: VQ-VAE — Plan

## Goal
Second generative strategy. VQ-VAE replaces the continuous latent with a discrete codebook — more natural for pixel art where artists think in repeated patterns and palette-constrained blocks.

## What we're building
1. **VectorQuantizer** — codebook of K embeddings, straight-through estimator for gradients
2. **VQVAEModel** — encoder → quantize → decoder (reuses ConvEncoder/ConvDecoder from Task 3)
3. **VQVAEStrategy** — implements GenerativeStrategy ABC
   - `train_step`: recon loss + codebook loss + commitment loss
   - `sample`: sample random codebook indices, decode
   - `get_metrics`: recon MSE + codebook utilization

## Key concepts
- **Vector quantization**: encoder output is mapped to nearest codebook entry
- **Straight-through estimator**: gradients flow through the quantization step by copying decoder gradients to encoder
- **Three loss terms**:
  - Reconstruction: MSE(decoded, input)
  - Codebook: MSE(codebook entries, encoder output).detach() — moves codebook toward encoder
  - Commitment: MSE(encoder output, codebook entries.detach()) — prevents encoder from drifting

## Architecture
```
Input (3, 16, 16)
  → ConvEncoder [32, 64] → (N, 64, 4, 4)
  → VectorQuantizer(K=256, D=64) → quantized (N, 64, 4, 4) + indices
  → ConvDecoder [64, 32] → (N, 3, 16, 16)
```

## Design choices
- Codebook size 256, dim 64 (matching encoder output channels)
- No flattening to latent vector — quantize at the spatial feature map level
- EMA codebook updates vs gradient-based: start with gradient-based (simpler)
- Sampling: random codebook indices → decode (naive but functional for now)

## Done when
- Smoke test passes with VQ-VAE strategy
- Two-strategy comparison visible on MLflow dashboard (VAE vs VQ-VAE)
- VQ-VAE samples should look at least as good as VAE (likely sharper)
