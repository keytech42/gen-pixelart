# Design Decisions

Cumulative log of non-obvious design decisions made across tasks.

---

## Task 1: Skeleton

- **PyTorch over MLX**: Chose PyTorch for access to reference implementations, better debugging ecosystem, and first-class MLflow integration. MLX's functional elegance wasn't worth the friction for a learning project focused on generative architectures. MPS backend is sufficient for this model scale.
- **Config merging via OmegaConf defaults key**: Strategy configs list `defaults: [base]` and get merged with `base.yaml`. Simple and avoids Hydra complexity for a toy project.
- **Dummy strategy for pipeline validation**: Included a `DummyStrategy` that exercises the full Trainer → Strategy → MLflow path with a trivial linear model. Ensures infrastructure works before real model code arrives.
- **Strategy train_step receives optimizer**: Strategies get `(model, optimizer, batch)` and own the backward pass. This gives each strategy full control over gradient computation (important for VQ straight-through, diffusion noise prediction, etc.).

## Task 2: Data

- **Kenney 1-Bit Pack as starter dataset**: 1,077 monochrome 16x16 sprites, CC0 license. Binary palette simplifies early training — good difficulty ramp before colored sprites.
- **Nearest-neighbor resize only**: Bilinear creates non-palette colors. All resizing in the pipeline uses `Image.NEAREST`.
- **K-means palette extraction over hardcoded palettes**: Works for any sprite set. Subsample to 100k pixels for speed on large datasets.
- **base.yaml defaults to image_size=16, palette_size=2**: Matches Kenney 1-Bit Pack. Override per-strategy config for larger/colored datasets.
- **Trainer takes a Dataset, not a Tensor**: Initial version stacked all sprites into a tensor with augmentation baked in once. Switched to passing a `Dataset` with a `DataLoader` so augmentations (flip, palette swap) are applied randomly per access each epoch — standard PyTorch pattern.

## Task 3: VAE

- **Two conv layers, not three**: Config had [32, 64, 128] but 128 channels at 2x2 spatial is wasteful for 16x16 monochrome. Reduced to [32, 64] → 4x4 spatial bottleneck. ~70K params total.
- **latent_dim=32, kl_weight=0.0005**: 64-dim latent was overkill for 2-color sprites. Low KL weight lets the model focus on reconstruction first — KL settled around 3.5 which is fine.
- **Strategy stores kl_weight, not model**: Strategy owns hyperparameters that affect the loss, model owns architecture. Clean separation — the model's forward() is a pure encoder-decoder, the strategy decides how to weight loss terms.

## Task 4: VQ-VAE

- **Codebook size 128, dim 64**: 256 entries with only ~13% utilization at convergence suggests 128 is sufficient. Dim matches encoder output channels to avoid extra projection layers.
- **Gradient-based codebook updates over EMA**: Simpler to implement and debug. EMA would be the next step if codebook collapse becomes an issue.
- **Random-index sampling is naive**: VQ-VAE samples by decoding random codebook indices, which ignores spatial structure. Samples are blocky. A proper autoregressive prior over indices is needed for good generation (future work).
- **VQ loss: codebook and commitment must be weighted separately**: The original VQ-VAE paper applies commitment_weight (β) only to the commitment loss, not the codebook loss. `L = recon + codebook_loss + β * commitment_loss`. An early implementation incorrectly bundled both as `β * (codebook + commitment)`, underweighting the codebook signal by 4x. Fixed by returning them separately from VectorQuantizer.

## Task 5: Diffusion

- **No attention in U-Net**: 16x16 spatial resolution is too small to benefit. Convolutions already have near-global receptive field. Saves ~40% parameters.
- **GroupNorm over BatchNorm**: BatchNorm running statistics are miscalibrated during the 1000-step sampling loop (different noise distribution than training). GroupNorm normalizes per-sample, so works identically in train and eval modes.
- **Linear beta schedule, not cosine**: Simpler, well-understood. Cosine schedule would improve fine detail quality but the difference is negligible at 16x16 monochrome.
- **Smoke test uses timesteps=10**: Full 1000-step sampling in the smoke test would take too long. 10 steps validates the contract (shapes, types, gradients) without the wait.
- **7.4M params is intentionally large**: Diffusion resists overfitting via noise injection (each image seen at 1000 noise levels). Larger capacity helps capture the full denoising mapping.

## Task 6a: BCE loss for VAE

- **BCE over MSE for binary sprites**: Confirmed experimentally. BCE gives stronger gradients near 0 and 1, pushing the model toward crisp black/white decisions. 500-epoch comparison showed BCE produces sharper edges, better silhouettes, and more black space between features. MSE produces softer blobs with white bleeding. Kept as the default loss.

## Task 6b: VQ-VAE Prior

- **Separate prior model, not integrated into strategy ABC**: The prior is a second model trained on VQ-VAE's output. Embedding it into the GenerativeStrategy ABC would complicate the interface for no benefit. Instead, VQVAEStrategy has an optional `prior` field — when set, `sample()` uses it; otherwise falls back to random indices.
- **4-layer GPT over 16-token sequences**: 4x4 spatial grid = 16 codebook indices. Small transformer (828K params) with causal masking. CE loss dropped 4.03 → 0.45 in 300 epochs (~40 seconds). Completely fixed VQ-VAE sampling — output went from checkerboard noise to coherent sprites.
- **Two-stage training script**: `scripts/train_vqvae_prior.py` handles both stages sequentially. Stage 1 trains VQ-VAE (500 epochs), stage 2 encodes dataset and trains prior (300 epochs). Clean separation — prior never modifies VQ-VAE weights.

## Task 6c: DDIM Sampling

- **DDIM 50 steps as default**: 16.4x speedup (0.83s vs 13.6s for 16 images) with comparable visual quality. Config defaults to `sampling_method: ddim, sampling_steps: 50`.
- **eta=0 (deterministic)**: Same noise → same image. Useful for reproducibility. DDPM equivalence at eta=1 preserved as a fallback.
- **Backward compatible**: `sampling_method` defaults to `"ddpm"` if not in config, so old configs and smoke test still work. Smoke test uses `sampling_method: ddim` with 5 steps to validate the DDIM code path quickly.
