# Class-Conditional Generation

## What

Unconditional models generate "a sprite" — you get whatever the model decides. Conditional models generate "a sprite of type X" — you specify what you want. The conditioning signal can be a class label, text description, or any other metadata. In our case, it's a cluster ID: "generate something from cluster 6" means "generate a tree."

## Why here

A model trained on 873 mixed sprites (trees, characters, buildings, cards, terrain) has to learn all of these distributions simultaneously. When you sample unconditionally, you get a random mix. Conditioning lets you steer the output, which is both more useful and more interpretable — you can verify the model learned each category separately.

For pixel art game development, this is the difference between "generate random sprites" and "I need 5 more tree variations for this forest tileset."

## How: Clustering for labels

### The problem: unlabeled data

Our sprites came from a tilesheet with no metadata. There are no filenames like "tree_01.png" or "character_running.png" — just `sprite_003_012.png` (row 3, column 12). We need to create labels before we can do conditional generation.

### Why k-means on raw pixels

We used k-means clustering on flattened pixel features — each 16x16x3 sprite becomes a 768-dimensional vector, and k-means groups vectors by Euclidean distance.

This sounds naive. Why not use a pretrained feature extractor? Or a VAE's latent space? Three reasons:

**1. The images are tiny.** At 16x16, there's no "high-level" vs "low-level" feature distinction. A CNN's first conv layer sees the entire image. Raw pixels capture everything there is to capture.

**2. Visual similarity ≈ semantic similarity at this scale.** Two trees look alike in pixel space (green mass, centered, roughly triangular). Two characters look alike (tan/gray figure, centered, roughly humanoid). At 256x256, raw pixel similarity would be useless — two photos of dogs in different poses have very different pixel values. At 16x16 with 8 colors, similar-looking sprites really are similar.

**3. Simplicity.** K-means is deterministic, fast (~1 second for 873 sprites), requires no training, and produces results you can visually verify. The goal is labels for conditioning, not a state-of-the-art taxonomy.

### Alternatives considered

**Learned clustering (VAE latent space).** Encode sprites through the VAE, cluster in latent space. Pro: captures structural similarity better than raw pixels. Con: circular dependency — we'd need to train the VAE first, and the VAE is one of the models we want to condition. Also, our VAE's latent space is 32-dimensional, which is actually *lower* resolution than raw pixels (768-dim). For tiny sprites, this loses information.

**Manual labeling.** Look at each sprite, assign a category. Pro: highest quality labels. Con: 873 sprites × ~5 seconds each = over an hour of tedious work. And manual labels introduce human bias — is a house with a chimney a "building" or a "structure"? Clustering is objective.

**Self-supervised contrastive learning.** Train a small encoder with augmentation-based contrastive loss, then cluster in the learned feature space. Pro: captures augmentation-invariant structure. Con: massive overkill for 16x16 sprites, and we'd need to choose augmentations carefully (pixel art augmentations are limited).

**Density-based clustering (DBSCAN, HDBSCAN).** Doesn't require specifying k. Pro: finds natural cluster boundaries. Con: requires tuning eps/min_samples, and the "natural" clusters might not correspond to useful categories (could split characters into 20 micro-clusters by pose).

### Interpreting the results

Our 8 clusters (`scripts/cluster_sprites.py`):

```
Cluster 0 (52):  Colorful objects — mixed-color icons, potions, large items
Cluster 1 (63):  Small characters — humanoid figures, gray/tan
Cluster 2 (56):  Cards/UI — playing cards, red-accented panels
Cluster 3 (75):  Wide structures — fences, walls, shelves
Cluster 4 (79):  Panels/windows — brown tiles, checkered patterns
Cluster 5 (125): Faces/heads — round characters, helmets, robots
Cluster 6 (220): Nature — trees, plants, cacti (green dominant)
Cluster 7 (203): Terrain/debris — brown scattered tiles, particles
```

**What makes these clusters "good"?**
- Each cluster is visually coherent — you can look at 8 random samples and guess the category
- Cluster sizes are roughly balanced (52-220 range, no cluster has 2 sprites)
- The distinguishing features are meaningful: color dominance (green → nature), shape (round → faces, wide → structures), density (sparse → terrain)

**What's imperfect?**
- Clusters 3 and 4 are similar (both are brown rectangular structures). K-means with k=8 splits them, but k=7 might merge them.
- Cluster 0 is a "miscellaneous" catch-all for sprites that don't fit elsewhere. This is common with k-means — outliers get grouped together.
- The terrain cluster (7) contains both intentional terrain tiles and sprite debris. A human would separate these.

**Does k matter?** We chose k=8 empirically. Too few (k=3): clusters are too broad ("everything green" vs "everything brown" vs "other"). Too many (k=16): clusters fragment into micro-categories that aren't useful for conditioning (characters split by left-facing vs right-facing). k=8 is a sweet spot for this dataset size and diversity.

## How: Conditioning the model

### The key insight: conditioning = biased embedding

In an unconditional diffusion model, the U-Net receives `(noisy_image, timestep)` and predicts the noise. Adding conditioning means giving it one more input: the class label. The model learns to denoise differently depending on the label — "if this is a tree, the green parts are signal, not noise."

### Implementation: class embedding added to time embedding

```python
class UNet(nn.Module):
    def __init__(self, ..., num_classes=None):
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)

    def forward(self, x, t, class_label=None):
        t_emb = self.time_embed(t)
        if class_label is not None and self.class_embed is not None:
            t_emb = t_emb + self.class_embed(class_label)
        # ... rest of U-Net uses t_emb in every ResBlock
```

**Why add to time embedding?** Both time and class are *global* conditioning signals — they don't vary spatially across the image. They answer the same type of question for the ResBlocks: "what context should I denoise in?" Time says "how noisy is this?" Class says "what category is this?" Adding them together lets the ResBlocks see both signals through a single injection path. No architectural changes needed below this point.

**Why `nn.Embedding`?** Our class labels are discrete integers (0-7). An embedding table maps each integer to a learned vector of the same dimension as the time embedding. During training, the model learns that class 6's embedding should bias denoising toward green, tree-like patterns.

### The strategy pattern still holds

The ABC adds optional `labels` and `class_label` parameters:

```python
def train_step(self, model, optimizer, batch, labels=None) -> dict:
    ...
def sample(self, model, n_samples, device, class_label=None) -> Tensor:
    ...
```

Strategies that don't support conditioning (VAE, VQ-VAE, dummy) accept and ignore these parameters. Only DiffusionStrategy forwards them to the model. The Trainer extracts labels from DataLoader batches and passes them through — it doesn't know or care whether the strategy uses them.

### Training

During each training step, the DataLoader yields `(image, label)` tuples. The strategy passes the label to the model alongside the noisy image and timestep:

```python
pred_noise = model(x_t, t, class_label=labels)
loss = F.mse_loss(pred_noise, noise)
```

The model learns to predict noise conditional on the class. There's no separate "conditional loss" — it's the same MSE loss, but the model gets extra information to help it denoise.

### Sampling

Specify a class label at sampling time:

```python
# "Generate 8 trees"
samples = strategy.sample(model, n_samples=8, device=device, class_label=6)
```

The label is broadcast to all samples in the batch and passed to the model at every denoising step.

## Gotchas

**No classifier-free guidance (yet).** Our conditioning is simple: always provide the label during training. A more powerful technique called classifier-free guidance (CFG) randomly drops labels during training (replacing them with a null token), then at sampling time computes:

```
eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

This amplifies the effect of the conditioning signal. With high guidance_scale, "tree" becomes "very tree-like." CFG is a natural follow-up but adds complexity we don't need for a first pass.

**Cluster quality limits conditioning quality.** If cluster 0 is a heterogeneous "miscellaneous" bin, conditional samples for class 0 will also be heterogeneous. The model faithfully learns the cluster distribution — garbage labels in, garbage conditioning out.

**Small clusters may underfit.** Cluster 0 has only 52 sprites. With 1000 training epochs, the model sees each class-0 sprite ~1000 times but each class-6 sprite ~1000 times too — and there are 4x more class-6 sprites. The model may learn class 6 (nature) better than class 0 (miscellaneous) simply due to data volume. Oversampling small clusters or weighting their loss higher could help.

**Conditioning doesn't mean control.** Specifying class 6 means "generate something from the cluster-6 distribution," not "generate a specific tree." You can't say "generate a pine tree facing left." For that level of control, you'd need much more detailed conditioning (text, reference images, sketches) and much more data.
