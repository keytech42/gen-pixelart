# Why Some Classes Condition Better Than Others

## The phenomenon

We trained a class-conditional diffusion model on 8 auto-clustered sprite categories. The results were uneven:

| Class | Quality | Intra-class variance |
|-------|---------|---------------------|
| 2: Cards | Excellent — learned exact card template | Very low (all cards share border + number + suit layout) |
| 1: Characters | Strong — clear humanoid shapes | Low (centered figure, head + body + legs pattern) |
| 3: Structures | Good — rectangular with detail | Medium (all rectangular but varied internal patterns) |
| 6: Nature | Moderate — some green, less structured | High (trees, cacti, bushes, grass — many distinct shapes) |
| 7: Terrain | Moderate — correct density, less coherent | Very high (scattered particles, tiles, debris — no template) |

Class 2 (cards, 56 sprites) outperforms class 6 (nature, 220 sprites) despite having **4x less training data**. This is counterintuitive — more data usually helps. Understanding why requires thinking about what the model is actually learning.

## The theory: conditional entropy

When you condition a generative model on class label y, the model learns the **conditional distribution** p(x|y). The difficulty of learning this distribution depends on its **conditional entropy** H(X|Y=y) — how much uncertainty remains about the image after you know the class.

**Low conditional entropy** (cards): Knowing "this is a card" tells you almost everything. It will be a rectangle with a colored border, a number in the upper-left, and a suit symbol. The only variation is which number and which suit. The conditional distribution is tight and unimodal — easy to learn.

**High conditional entropy** (nature): Knowing "this is nature" leaves enormous uncertainty. Is it a triangular pine tree? A round bush? A cactus? Falling leaves? The conditional distribution is broad and multimodal — the model must learn many distinct sub-patterns, each with limited data.

Formally: the model's per-class loss is bounded below by the conditional entropy. A class with low conditional entropy has a lower achievable loss floor, so the model can produce sharper, more accurate samples with the same capacity and training time.

## Why more data doesn't always help

Class 6 has 220 sprites. But those 220 sprites cover many visual sub-types:

```
Nature (220 total):
  ├── Triangular pine trees (~40)
  ├── Round deciduous trees (~35)
  ├── Cacti (~20)
  ├── Bushes (~30)
  ├── Leaf particles (~25)
  ├── Grass/ground cover (~30)
  └── Flowers/mushrooms (~40)
```

Each sub-type has ~30 examples. Meanwhile, class 2 has 56 cards that all share the same template. The **effective sample density** per mode matters more than the total count:

- Cards: 56 examples / 1 visual template = 56 samples per mode
- Nature: 220 examples / ~7 visual modes = ~31 samples per mode

Cards have nearly 2x the sample density per mode. The model sees more repetitions of the same pattern, enabling stronger learning.

This is a fundamental tradeoff in conditional generation: **classes with consistent visual templates condition better than classes with diverse visual content**, regardless of total sample count.

## The role of the clustering method

K-means on raw pixels groups sprites by **pixel-level similarity**, which creates clusters of varying semantic coherence:

**High coherence** (cards, characters): Sprites that look similar in pixel space also share semantic structure. All cards have the same layout. All characters have the same body plan. K-means captures both visual and structural similarity.

**Lower coherence** (nature, terrain): Sprites are grouped because they share color statistics (green, brown) but have diverse structures. A pine tree and a bush are both "green centered mass" in pixel space but have very different shapes. K-means captures color similarity but misses structural differences.

This means the clustering method directly impacts conditioning quality. A clustering approach that captures structural similarity (e.g., using shape descriptors, or clustering in a learned feature space) would produce tighter classes and better conditioning — especially for visually diverse categories.

## Practical implications

**1. Cluster quality > cluster quantity.** When designing conditional generation, invest in label quality. Tight, visually coherent classes condition well even with little data. Broad, heterogeneous classes condition poorly even with lots of data.

**2. Consider sub-clustering.** If a class has high intra-class variance (like nature), split it. "Pine trees" and "cacti" as separate classes would each condition better than "nature" as a single class. The cost: more classes means less data per class, and the model needs more capacity.

**3. Classifier-free guidance amplifies what's already learned.** CFG works by contrasting conditional and unconditional predictions. If the conditional distribution is already tight (cards), CFG sharpens it further. If the conditional distribution is broad (nature), CFG amplifies the average of all nature sub-types — which may just produce a generic green blob. CFG improves cards more than it improves nature.

**4. The conditioning ceiling is set by H(X|Y).** No amount of training, model capacity, or guidance scale can make a conditional model produce tight samples from a broad conditional distribution. If "nature" inherently contains 7 distinct visual patterns, the model will always produce varied output for that class. This is correct behavior — the class genuinely is diverse.

## Connection to the broader theory

This phenomenon — conditioning quality varying by class — appears throughout generative modeling:

- **Text-to-image models** generate "a red car" well (tight visual concept) but "an abstract painting" poorly (unbounded visual space)
- **Language models** complete "The capital of France is" confidently (one correct answer) but "Write a poem about" with high variance (many valid completions)
- **Class-conditional ImageNet models** generate "goldfish" crisply (distinctive, consistent shape) but "pizza" with more variation (many toppings, angles, lighting)

The unifying principle: **conditioning works best when the condition reduces uncertainty.** A useful class label is one where p(x|y) has significantly lower entropy than p(x). If knowing the class doesn't narrow down what the image looks like, the label isn't doing useful work.

This is also why the original VQ-VAE paper frames the prior as modeling a distribution over discrete codes — the codebook forces low entropy at each spatial position, making the prior's job easier. The diffusion model doesn't have this advantage; it must learn to denoise continuous pixel values conditional on a class signal, which is harder when the class spans many visual modes.
