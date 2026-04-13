# Vector Quantized VAE (VQ-VAE)

## What

VQ-VAE replaces the VAE's continuous latent space with a **discrete codebook** — a fixed-size dictionary of learned embedding vectors. The encoder outputs a continuous feature map, but each spatial position gets snapped to its nearest codebook entry before decoding. The result is a compressed, discrete representation of the image.

Think of it as a vocabulary of visual building blocks. The encoder says "this 4x4 region of the image is best described by codebook entry #47," and the decoder knows how to turn entry #47 back into pixels.

## Why here

VQ-VAE is a natural fit for pixel art because pixel artists already think in discrete terms:

- Limited color palettes (4-32 colors)
- Repeated tile patterns
- Deliberate placement of individual pixels

A continuous latent space (like VAE) struggles with this — it wants to smoothly interpolate between states, producing blurry averages. A discrete codebook forces crisp decisions: this region uses pattern A or pattern B, with nothing in between.

The tradeoff: sampling is harder. With a VAE, you sample from N(0,1). With VQ-VAE, you need a **prior** over codebook indices — which indices tend to appear together, and in what spatial arrangement? Without a prior, sampling means picking random indices, which produces incoherent images.

## How

### Vector Quantization (`src/models/codebook.py`)

The core operation: given a continuous vector from the encoder, find the nearest vector in the codebook.

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, z_e):
        # z_e: (N, D, H, W) from encoder
        # Find nearest codebook entry for each spatial position
        dists = ||z_e - codebook||²  # (N*H*W, K) distance matrix
        indices = dists.argmin(dim=-1)  # Nearest codebook entry
        z_q = self.embedding(indices)   # Look up the actual vectors
```

This is just a nearest-neighbor lookup. The hard part is training it.

### The straight-through estimator (`src/models/codebook.py:63-64`)

The problem: `argmin` is not differentiable. Gradients can't flow through "pick the nearest entry." So how does the encoder learn?

```python
# Straight-through: copy gradients from z_q to z_e
z_q = z_e + (z_q - z_e).detach()
```

This single line is the key insight. In the **forward pass**, `z_q` has the codebook values (what we want for decoding). In the **backward pass**, `.detach()` makes `(z_q - z_e)` a constant, so the gradient of `z_q` equals the gradient of `z_e`. The decoder's gradients pass straight through the quantization step to the encoder.

It's a hack. An elegant, mathematically-motivated hack. The encoder doesn't get the "right" gradient (the true gradient is zero almost everywhere), but it gets a useful one: "if the decoder wanted this quantized vector to be slightly different, move the encoder output in that direction."

### Three loss terms

VQ-VAE has three loss components, not two:

```python
# 1. Reconstruction: how well does the decoded image match the input?
recon_loss = F.mse_loss(recon, batch)

# 2. Codebook loss: move codebook entries toward encoder outputs
codebook_loss = F.mse_loss(z_q, z_e.detach())  # Note: z_e is detached

# 3. Commitment loss: move encoder outputs toward codebook entries
commitment_loss = F.mse_loss(z_e, z_q.detach())  # Note: z_q is detached
```

**Why two separate losses for the codebook?** Because the encoder and codebook are pulling toward each other, and we want to control the balance:

- **Codebook loss** (`sg[z_e] - e`): "codebook entries, move toward where the encoder is pointing." The encoder is detached (`sg` = stop gradient) so this only updates the codebook.
- **Commitment loss** (`z_e - sg[e]`): "encoder, commit to your nearest codebook entry — don't drift around." The codebook is detached so this only updates the encoder.

Without the commitment loss, the encoder's output can grow unboundedly (there's no KL penalty like in VAE). The commitment loss keeps it anchored near the codebook.

In our implementation, `VectorQuantizer.forward()` returns them separately, and the strategy applies the correct weighting from the paper:

```python
# Paper: L = recon + ||sg[z_e] - e||² + β * ||z_e - sg[e]||²
loss = recon_loss + codebook_loss + self.commitment_weight * commitment_loss
```

Note: codebook_loss always has weight 1.0. Only commitment_loss gets the β weight. An earlier version of this code incorrectly applied β to both — this underweighted the codebook loss by 4x and contributed to poor codebook utilization.

### Architecture (`src/strategies/vqvae.py`)

```
Input (3, 16, 16)
  → ConvEncoder [32, 64]         → (64, 4, 4)  continuous features
  → VectorQuantizer(K=128, D=64) → (64, 4, 4)  discrete (quantized)
  → ConvDecoder [64, 32]         → (3, 16, 16) reconstructed image
```

Key difference from VAE: **no flattening to a vector.** The encoder outputs a 4x4 spatial feature map with 64 channels. Each of the 16 spatial positions gets independently quantized to a codebook entry. The latent representation is a 4x4 grid of codebook indices — 16 integers from {0, ..., 127}.

This spatial structure is important: nearby positions in the grid correspond to nearby regions of the image. The codebook entries learn to represent local visual patterns, not global image properties.

### Sampling: random indices vs. learned prior

**The naive approach** — random codebook indices — produces incoherent checkerboard noise:

```python
# Random: ignores spatial structure entirely
indices = torch.randint(0, K, (n_samples, 4, 4))
```

The model learned that certain index patterns go together (e.g., entry #12 at position (0,0) usually appears next to entry #47 at position (0,1)), but random sampling doesn't respect this.

**The fix: an autoregressive prior** (`src/models/prior.py`). A small GPT-style transformer (4 layers, 828K params) trained on the VQ-VAE's codebook index sequences. The 4x4 grid is flattened to 16 tokens and modeled autoregressively with causal masking:

```python
class IndexPrior(nn.Module):
    def sample(self, n_samples, device, temperature=1.0):
        indices = torch.randint(0, K, (n_samples, 1))  # Random first token
        for i in range(1, 16):
            logits = self.forward(indices)        # Predict next token
            next_token = sample(logits[:, -1, :])  # From learned distribution
            indices = cat([indices, next_token])
        return indices
```

**Two-stage training** (`scripts/train_vqvae_prior.py`):
1. Train VQ-VAE normally (500 epochs)
2. Freeze VQ-VAE, encode full dataset → 873 sequences of 16 indices each
3. Train prior on these sequences (300 epochs, CE loss 4.03 → 0.45, ~40 seconds)

The result is dramatic: prior-sampled sprites look coherent (chests, buildings, icons), while random-index sprites are noise. The prior learned which codebook entries belong together and in what spatial arrangement.

**Temperature controls diversity.** `temperature=1.0` matches the training distribution. Lower values (0.7-0.8) produce more typical, "safer" sprites. Higher values produce more varied but potentially less coherent output.

## Gotchas

**Codebook collapse.** Only 13% of our 128-entry codebook is used at convergence. The other 87% of entries are "dead" — no encoder output is close to them. This happens because:
- Early training: a few entries get lucky, attract nearby encoder outputs, and get reinforced
- Dead entries: never get selected, never get gradients, drift further away

Solutions (not yet implemented): EMA updates (exponential moving average of encoder outputs, more stable than gradient-based), codebook reset (periodically reinitialize dead entries to random encoder outputs), or lower codebook size (if 17 entries suffice, use 32 instead of 128).

**commitment_weight = 0.25 is a sensitive hyperparameter.** Too low: encoder output drifts far from codebook, straight-through gradient becomes inaccurate. Too high: encoder is over-constrained, can't learn useful features. The original VQ-VAE paper uses 0.25 as default. We kept it.

**VQ-VAE reconstructions are sharper than VAE.** Even though recon_loss is similar (~0.08 vs ~0.01 for VAE), the discrete bottleneck forces the model to make hard decisions at each spatial position. No averaging, no blurring. The reconstructed images will have more structure (but also more wrong pixels when the wrong codebook entry is chosen).

**The codebook dimension must match (or be projected to) the encoder output channels.** Our encoder outputs 64-channel feature maps, and codebook_dim=64. If they differed, we'd need a 1x1 conv to project between them. The model handles this with `pre_quant` and `post_quant` layers, but matching dimensions avoids the extra parameters.
