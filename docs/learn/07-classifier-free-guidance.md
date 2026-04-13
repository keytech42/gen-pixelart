# Classifier-Free Guidance (CFG)

## What

Classifier-free guidance is a technique that amplifies the effect of conditioning during sampling, making the model "try harder" to produce outputs that match the requested class. It requires no additional classifier — just a small change to training and a dual-pass sampling procedure.

The core formula:

```
ε_guided = ε_uncond + s * (ε_cond - ε_uncond)
```

Where `s` is the guidance scale. At s=1, you get standard conditional sampling. At s>1, the model overshoots in the direction of the conditioning — producing more class-specific output at the cost of some diversity.

## Why here

Our conditional diffusion model showed uneven quality across classes (see `06-conditioning-analysis.md`). Classes with high intra-class variance (nature, terrain) produced less coherent samples than classes with low variance (cards, characters). CFG addresses this by amplifying whatever class signal the model learned, pushing even broad distributions toward more typical examples.

For pixel art specifically, CFG has a bonus effect: it tends to produce cleaner, more decisive output. The guidance pushes the model away from "uncertain" noise predictions and toward confident ones, which translates to crisper pixel placement.

## How: Training with label dropout

### The problem with standard conditioning

A standard conditional model always sees labels during training. At sampling time, when you provide a label, the model uses it. When you don't, there's no fallback — the model has never seen an input without a label. It can't distinguish "I know the class and it's X" from "I don't know the class."

CFG fixes this by training the model to handle both cases.

### Random label dropout (`src/strategies/diffusion.py`)

During training, we randomly replace class labels with a special "null" token with probability `p_uncond` (default: 10%):

```python
# During train_step:
if labels is not None and self.p_uncond > 0:
    drop_mask = torch.rand(batch_size) < self.p_uncond  # 10% of the time
    labels = labels.clone()
    labels[drop_mask] = self.num_classes  # null token (index 8 for 8 classes)
```

**The null token** is an extra entry in the class embedding table. The U-Net has `nn.Embedding(num_classes + 1, time_emb_dim)` — indices 0-7 are the real classes, index 8 is "unconditional." When the model sees the null token, it learns to denoise without class information, producing the unconditional noise prediction ε_uncond.

**Why 10% dropout?** It's a tradeoff:
- Too low (1%): the model rarely sees unconditional inputs, so ε_uncond is poorly calibrated. Guidance is noisy.
- Too high (50%): the model spends half its capacity on unconditional denoising, weakening the conditional predictions.
- 10% is the standard default from the original CFG paper (Ho & Salimans, 2022). It gives enough unconditional training signal without significantly hurting conditional quality.

### Why this works: learning two distributions simultaneously

After training with dropout, the model implicitly represents two distributions:
- **p(noise | x_t, t, class)** — conditional, from the 90% of steps with real labels
- **p(noise | x_t, t)** — unconditional, from the 10% of steps with null labels

The same model weights represent both, shared through the same U-Net. The only difference is what goes into the class embedding. This is remarkably efficient — you're learning unconditional generation "for free" as a byproduct of conditional training.

## How: Guided sampling

### The dual forward pass

At each denoising step during sampling:

```python
def _guided_noise_pred(self, model, x, t, labels):
    if labels is None or self.guidance_scale <= 1.0:
        return model(x, t, class_label=labels)  # Standard prediction

    # Two forward passes
    eps_cond = model(x, t, class_label=labels)           # "what noise, given class?"
    eps_uncond = model(x, t, class_label=null_labels)     # "what noise, unconditionally?"

    # Guided blend
    return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
```

**Cost**: 2x the compute per denoising step. With DDIM 50 steps, that's 100 model evaluations instead of 50. Still fast (~1.5s for 16 images on MPS).

### Interpreting the guidance formula

```
ε_guided = ε_uncond + s * (ε_cond - ε_uncond)
```

Think of `(ε_cond - ε_uncond)` as the "class direction" — the component of the noise prediction that is specifically about this class. The guidance scale `s` controls how far you go in that direction:

- **s = 0**: Pure unconditional. Ignores the class entirely.
- **s = 1**: Standard conditional. Same as no guidance.
- **s = 3**: Moderate guidance. Class features are amplified — trees are greener, cards are more card-shaped.
- **s = 5-7**: Strong guidance. Very class-specific but may lose diversity and produce artifacts.
- **s > 10**: Over-guidance. Saturated colors, repetitive patterns, visual artifacts. The model is extrapolating too far.

### The quality-diversity tradeoff

Higher guidance produces samples that are more recognizably "class X" but more similar to each other. The model converges toward the mode of the conditional distribution — the single most typical example of each class.

This is a fundamental tradeoff, not a bug:
- **Low guidance**: diverse but some samples may not clearly match the class
- **High guidance**: clear class membership but less variation between samples

For pixel art, moderate guidance (s=3-5) tends to work well. The palette constraint already limits diversity, so pushing toward class modes produces clean, recognizable sprites without excessive repetition.

### Why not train a separate classifier?

An older approach (Dhariwal & Nichol, 2021) uses a separate classifier p(y|x_t) to guide sampling. You train a noise-aware classifier, compute its gradient with respect to x_t, and use that gradient to steer the denoising process.

CFG replaces this with a simpler approach: instead of an external classifier, the model itself provides the class signal through the contrast between conditional and unconditional predictions. Advantages:

1. **No extra model**: One model does both conditional and unconditional prediction
2. **No gradient computation**: CFG operates on noise predictions, not gradients through a classifier
3. **Better quality**: Empirically, CFG produces better samples than classifier guidance at equivalent compute

The intuition: a classifier trained on noisy images is hard to get right (what does a class-6 image look like at noise level t=900?). The diffusion model already knows how to handle all noise levels — using its own conditional/unconditional contrast is more reliable than an external classifier's gradient.

## Implementation details

### The null class token in the embedding table

```python
# In UNet.__init__:
self.class_embed = nn.Embedding(num_classes + 1, time_emb_dim)
#                                           ^^^ extra slot for null token
```

The null token (index `num_classes`) starts with random weights like any other embedding. During training, the model learns what "no class" looks like — the null embedding converges to a "generic sprite" representation.

### Guidance scale as a sampling-time knob

The guidance scale is not trained — it's a hyperparameter you set at sampling time. The same trained model can be sampled with scale=1 (no guidance), scale=3 (moderate), or scale=7 (strong) without retraining. This makes it cheap to experiment.

```python
# Same model, different guidance:
samples_weak = strategy.sample(model, 16, device, class_label=2, guidance_scale=1.0)
samples_strong = strategy.sample(model, 16, device, class_label=2, guidance_scale=5.0)
```

### Interaction with DDIM

CFG composes cleanly with DDIM. The guided noise prediction is computed first, then fed into the DDIM update rule. DDIM doesn't care how the noise prediction was computed — it just needs ε. This means you get both the speed of DDIM (50 steps) and the quality of CFG (amplified conditioning) simultaneously.

## Gotchas

**Double compute cost.** Each guided denoising step requires two model forward passes. For DDIM with 50 steps, that's 100 evaluations. Mitigations: batch the conditional and unconditional inputs together (one forward pass with 2N batch size), or cache the unconditional prediction if it's the same across a batch.

**Over-guidance artifacts.** At very high scales (>7), the model extrapolates beyond the training distribution. For pixel art, this manifests as saturated colors, repeated patterns, and loss of fine detail. The sweet spot depends on the class — tight classes (cards) tolerate higher guidance than broad classes (nature).

**p_uncond must be set before training.** The model needs to see unconditional inputs during training to learn ε_uncond. You can't add CFG to a model trained without dropout — the null embedding is untrained and ε_uncond is random noise. If you want CFG, set p_uncond > 0 from the start.
