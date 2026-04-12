# Denoising Diffusion Probabilistic Model (DDPM)

## What

Diffusion models generate images by learning to reverse a noise-adding process. The idea:

1. **Forward process**: Start with a real image. Gradually add Gaussian noise over T=1000 steps until it becomes pure static.
2. **Reverse process**: Learn a neural network that can undo one step of noise at a time. Starting from pure noise, apply this denoiser 1000 times to get a clean image.

The model never sees a "clean → image" mapping directly. Instead, it learns: "given this noisy image and the fact that it's at noise level t, what noise was added?" This is the **epsilon prediction** formulation.

## Why here

Diffusion models produce the highest-quality output of our three strategies. The results show this clearly — diffusion samples have the sharpest edges, most coherent silhouettes, and most sprite-like structure.

For pixel art specifically, diffusion has an advantage: it makes decisions iteratively. At high noise levels, the model decides the rough shape. At low noise levels, it refines individual pixels. This mirrors how pixel artists work — sketch the silhouette first, then place individual pixels. VAEs make all decisions at once (decode the entire image from one vector), which is why they tend toward blurry averages.

The tradeoff: sampling is slow. Each image requires T forward passes through the U-Net. Our 1000-step sampler takes ~12 seconds for 16 images. VAE and VQ-VAE sample in a single forward pass.

## How

### The noise schedule (`src/strategies/diffusion.py: NoiseScheduler`)

A **linear beta schedule** defines how much noise to add at each timestep:

```python
betas = torch.linspace(beta_start, beta_end, timesteps)  # 0.0001 → 0.02
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)  # Running product
```

`alpha_bar_t` is the key quantity: it tells you how much of the original signal remains at timestep t. At t=0, alpha_bar ≈ 1 (almost no noise). At t=999, alpha_bar ≈ 0 (almost pure noise).

The forward process can jump directly to any timestep without iterating:

```python
def q_sample(self, x_0, t, noise=None):
    # q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    return x_t, noise
```

This is a weighted mix of the original image and random noise. At early timesteps the image dominates; at late timesteps the noise dominates.

### The U-Net (`src/models/unet.py`)

The denoising model is a U-Net: encoder-decoder with skip connections. It takes a noisy image (3, 16, 16) and a timestep, and predicts the noise that was added.

```
Input: noisy image (3, 16, 16) + timestep t
  ↓ input conv → (64, 16, 16)
  ↓ ResBlock×2 → skip₁(64, 16, 16) → Downsample → (64, 8, 8)
  ↓ ResBlock×2 → skip₂(128, 8, 8)  → Downsample → (128, 4, 4)
  ↓ ResBlock×2 → skip₃(256, 4, 4)
  ↓ Mid: ResBlock → ResBlock → (256, 4, 4)
  ↑ cat(skip₃) → ResBlock×2 → Upsample → (128, 8, 8)
  ↑ cat(skip₂) → ResBlock×2 → Upsample → (64, 16, 16)
  ↑ cat(skip₁) → ResBlock×2 → (64, 16, 16)
  ↓ GroupNorm → SiLU → Conv → predicted noise (3, 16, 16)
```

**Skip connections** are crucial. The encoder captures "what's in the image" at different scales. The decoder uses these skip features to place details precisely. Without them, the model would struggle to reconstruct fine pixel-level details.

**No attention layers.** At 16x16, the spatial dimensions are small enough that convolutions already have a large effective receptive field. Attention would add parameters and computation for negligible benefit. (At 64x64+, attention becomes important for capturing long-range dependencies.)

### Time conditioning (`src/models/blocks.py`)

The model needs to know *how noisy* the input is. Timestep t is injected via:

1. **Sinusoidal embedding**: integer t → high-dimensional vector (same idea as positional encoding in Transformers)
2. **MLP projection**: embedding → time_emb_dim features
3. **Injection into ResBlocks**: time embedding is projected and added to the feature map between convolutions

```python
class ResBlock(nn.Module):
    def forward(self, x, t_emb=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if self.time_proj is not None and t_emb is not None:
            t = self.act(self.time_proj(t_emb))
            h = h + t[:, :, None, None]  # Broadcast over spatial dims
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)  # Residual connection
```

This additive injection means: "shift all feature activations based on the current noise level." At high noise, the model might activate broad shape detectors. At low noise, it might activate edge refiners.

### Training (`src/strategies/diffusion.py: train_step`)

Each training step:

```python
# 1. Sample random timesteps for each image in the batch
t = torch.randint(0, T, (batch_size,))

# 2. Add noise at those timesteps (forward process)
x_t, noise = scheduler.q_sample(x_0, t)

# 3. Predict the noise
pred_noise = model(x_t, t)

# 4. Loss: how close was the prediction?
loss = F.mse_loss(pred_noise, noise)
```

That's it. The model never sees the denoising *process* during training — it just learns to predict noise at random timesteps. The iterative denoising only happens at inference time.

**Why predict noise instead of the clean image?** Both work mathematically, but noise prediction gives more uniform gradient signal across timesteps. Predicting x_0 directly gives very small gradients at low noise levels (t near 0) where x_t ≈ x_0 already.

### Sampling (`src/strategies/diffusion.py: sample`)

The reverse process, starting from pure noise:

```python
x = torch.randn(n_samples, 3, 16, 16)  # Pure noise

for t in reversed(range(T)):  # 999, 998, ..., 1, 0
    pred_noise = model(x, t)
    # Compute x_{t-1} from x_t and predicted noise
    mean = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * pred_noise)
    if t > 0:
        x = mean + σ_t * random_noise  # Add a bit of stochasticity
    else:
        x = mean  # Final step: no noise added
```

Each step partially denoises the image. The formula comes from Bayes' rule applied to the Gaussian forward process — we're computing the posterior p(x_{t-1} | x_t, x_0), where x_0 is estimated from the predicted noise.

The `σ_t` (posterior variance) term adds controlled randomness. This is what makes the sampler *generative* rather than deterministic — different random seeds produce different images.

## Gotchas

**Sampling speed.** 1000 forward passes through a 7.4M parameter U-Net. On MPS, sampling 16 images takes ~12 seconds. This is the main practical drawback of DDPM. Solutions exist (DDIM sampling reduces to ~50 steps, consistency distillation reduces to 1-2 steps) but aren't implemented here.

**The loss doesn't directly measure image quality.** `noise_prediction_mse` tells you how well the model predicts noise, not how good the generated images look. A model can have low noise prediction error but still produce blurry or incoherent samples — especially early in training when it's only good at predicting noise at high timesteps. Always check the sample grids, not just the loss curve.

**Linear vs cosine schedule.** We use a linear beta schedule (simple, from the original DDPM paper). The cosine schedule (from Improved DDPM) allocates more timesteps to low noise levels, which can improve fine detail quality. For 16x16 monochrome sprites the difference is small, but switching is a one-line change.

**GroupNorm, not BatchNorm.** The U-Net uses GroupNorm (groups=8) instead of BatchNorm. Reason: during sampling, each denoising step processes the same batch of images but at different noise levels than training. BatchNorm's running statistics would be miscalibrated. GroupNorm normalizes within each sample independently, so it works identically in training and inference.

**7.4M parameters for 16x16 images.** This is overkill — the model has more parameters than pixels in the entire training set (1077 * 16 * 16 * 3 = 828K values). But diffusion models are surprisingly resistant to overfitting because of the noise injection in training — each image is seen at 1000 different noise levels, creating massive data augmentation. The model generalizes because it has to denoise, not memorize.

**Output clamping.** After the final denoising step, we clamp to [0, 1] before palette snapping. The reverse process can produce values slightly outside this range because of the accumulated stochasticity. Clamping is standard practice.
