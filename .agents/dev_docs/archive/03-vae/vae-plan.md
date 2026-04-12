# Task 3: VAE — Plan

## Goal
First real generative strategy end-to-end. A convolutional VAE that generates recognizable (if blurry) pixel art sprites.

## What we're building
1. **Encoder** — Conv layers → (mu, log_var) latent vectors
2. **Decoder** — Latent vector → ConvTranspose layers → reconstructed image
3. **VAEStrategy** — implements GenerativeStrategy ABC
   - `train_step`: reconstruction loss + KL divergence (with configurable kl_weight)
   - `sample`: decode random latent vectors from N(0,1)
   - `get_metrics`: reconstruction MSE + KL

## Design choices
- Small model (~200-500K params) — 16x16 monochrome sprites don't need more
- Encoder: Conv2d layers with stride 2 for downsampling (no pooling)
- Decoder: ConvTranspose2d for upsampling
- Loss: MSE reconstruction + beta-weighted KL (beta=kl_weight from config)
- Latent dim from config (default 64, likely overkill for 2-color — we can tune)

## Architecture sketch
```
Input (3, 16, 16)
  → Conv(3→32, k3, s2, p1)  → (32, 8, 8)
  → Conv(32→64, k3, s2, p1) → (64, 4, 4)
  → Flatten → Linear → mu, log_var (latent_dim)
  → Reparameterize
  → Linear → Unflatten (64, 4, 4)
  → ConvT(64→32, k3, s2, p1, op1) → (32, 8, 8)
  → ConvT(32→3, k3, s2, p1, op1)  → (3, 16, 16)
  → Sigmoid
```

## Done when
- `smoke_test.py` passes with VAE strategy
- Training on real sprites produces recognizable output in MLflow sample grids
- Palette-snapped samples show clean black/white pixel art shapes
