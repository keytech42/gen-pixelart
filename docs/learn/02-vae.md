# Variational Autoencoder (VAE)

## What

A VAE learns to compress images into a small latent vector and decompress them back. Unlike a regular autoencoder, the VAE imposes structure on the latent space — specifically, it forces it to look like a standard normal distribution. This means you can *sample* new images by decoding random points from that distribution.

The core idea: learn an encoder q(z|x) that maps images to latent distributions, and a decoder p(x|z) that maps latent vectors back to images. Training maximizes the Evidence Lower Bound (ELBO):

```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
       ↑ reconstruction    ↑ regularization
```

In practice:
- **Reconstruction term** → MSE between input and output (how well can we reconstruct?)
- **KL term** → how close is the learned latent distribution to N(0,1)? (how "regular" is the latent space?)

## Why here

VAE is the simplest generative architecture to get working end-to-end. It validated the full pipeline — data loading, training loop, strategy pattern, MLflow logging, palette snapping — before we built more complex models.

For pixel art specifically, VAEs have a known weakness: they produce blurry output. MSE loss averages over possible outputs, and the average of a pixel being black-or-white is gray. For our 2-color sprites, palette snapping partially rescues this — even if the raw output is grayish, snapping to black/white recovers sharp edges. This is why we always log palette-snapped samples, not raw output.

## How

### The reparameterization trick (`src/models/encoder.py:103-106`)

The encoder outputs two vectors: `mu` (mean) and `log_var` (log variance). To sample z from this distribution:

```python
def reparameterize(self, mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)       # Random noise
    return mu + std * eps             # Differentiable sampling
```

**Why not just sample directly from N(mu, sigma)?** Because sampling is not differentiable — you can't backpropagate through a random number generator. The reparameterization trick separates the randomness (`eps`) from the parameters (`mu`, `std`). The gradient flows through `mu + std * eps` because both `mu` and `std` are deterministic functions of the encoder output.

This is the single most important idea in VAEs. Without it, you can't train with standard backpropagation.

### Architecture (`src/models/encoder.py`)

```
Input (3, 16, 16)
  → Conv(3→32, k3, s2, p1) + BN + ReLU  → (32, 8, 8)
  → Conv(32→64, k3, s2, p1) + BN + ReLU → (64, 4, 4)
  → Flatten → (1024,)
  → Linear → mu (32,)     ← mean of latent distribution
  → Linear → log_var (32,) ← log variance of latent distribution
  → Reparameterize → z (32,)
  → Linear → (1024,) → Reshape (64, 4, 4)
  → ConvTranspose(64→32, k3, s2, p1, op1) + BN + ReLU → (32, 8, 8)
  → ConvTranspose(32→3, k3, s2, p1, op1) + Sigmoid    → (3, 16, 16)
```

**Stride-2 convolutions for downsampling** instead of pooling. Strided convs are learnable — the model decides how to downsample, not a fixed pooling operation.

**Sigmoid on the output.** Our images are in [0, 1], and Sigmoid naturally maps to that range. This pairs with MSE loss. If we used BCE loss instead (which we considered — see Gotchas), we'd also want Sigmoid output.

**32-dim latent space.** A 16x16 monochrome sprite has 16*16 = 256 binary decisions. 32 continuous dimensions is enough to capture the variation in 1,077 sprites. Higher dims give more capacity but make the latent space harder to regularize.

### Loss (`src/strategies/vae.py:36-42`)

```python
recon_loss = F.mse_loss(recon, batch, reduction="mean")
kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
loss = recon_loss + self.kl_weight * kl_loss
```

**The KL formula explained:**

For a single latent dimension with learned mean μ and variance σ²:

```
KL(N(μ, σ²) || N(0, 1)) = -0.5 * (1 + log(σ²) - μ² - σ²)
```

We work with `log_var = log(σ²)` for numerical stability (variance is always positive, but log variance can be any real number). The formula becomes:

```
KL = -0.5 * (1 + log_var - mu² - exp(log_var))
```

Averaged over all latent dimensions and the batch.

**kl_weight = 0.0005 (beta-VAE weighting).** Without this, the KL term dominates early training and the model learns to ignore the latent space (KL → 0, recon stays bad). A small weight lets reconstruction drive learning first, then KL gradually regularizes. This is the "beta" in beta-VAE. In our training: recon_loss dropped 0.27 → 0.01 while KL settled around 3.5 — a healthy balance.

### Sampling (`src/strategies/vae.py:53-59`)

```python
def sample(self, model, n_samples, device):
    z = torch.randn(n_samples, model.latent_dim, device=device)
    return model.decode(z)
```

Sample from N(0,1) — the distribution we pushed the latent space toward via KL regularization — and decode. If training worked, this should produce images that look like they came from the training distribution.

## Gotchas

**BCE over MSE for binary sprites — confirmed experimentally.** Our sprites are essentially binary (pixel is black or white). We switched from MSE to BCE (`F.binary_cross_entropy`) and compared 500-epoch training runs. BCE produces sharper edges, better silhouettes, and more defined shapes. Why: BCE gives stronger gradients when the prediction is near 0 or 1, pushing the model toward crisp black/white decisions. MSE treats a 0.1 error at 0.5 the same as at 0.99 — it doesn't penalize "almost right" enough for binary data. This is now the default loss.

**BatchNorm during sampling.** The decoder uses BatchNorm, which behaves differently in `train` vs. `eval` mode. During training, BN uses per-batch statistics; during eval (sampling), it uses running averages accumulated during training. If the running averages are poorly calibrated (early in training, or if the training distribution is very different from N(0,1) decoded), sampling quality suffers. An alternative is to use LayerNorm or no normalization in the decoder.

**The blurry output problem.** VAEs minimize expected reconstruction error. When a pixel could plausibly be black or white, the optimal prediction hedges — even with BCE, the model may produce probabilities near 0.5 for ambiguous pixels. Palette snapping rescues this by forcing discrete decisions. VQ-VAE (next doc) addresses the root cause by forcing discrete representations.

**KL collapse.** If kl_weight is too high, the encoder learns to output mu=0, log_var=0 for all inputs — the KL term is minimized but the latent space carries no information. The decoder then ignores z and learns an average image. Watch for KL approaching 0 with high recon_loss — that's collapse. Our kl_weight=0.0005 avoids this.
