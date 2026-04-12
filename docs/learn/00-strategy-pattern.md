# The Strategy Pattern for Generative Models

## What

The strategy pattern separates *what varies* (the generative model) from *what stays the same* (training loop, logging, data loading). In our case:

- **Context**: `Trainer` — owns the training loop, optimizer, device management, MLflow logging
- **Strategy**: `GenerativeStrategy` ABC — defines four methods that each generative architecture must implement
- **Concrete strategies**: `VAEStrategy`, `VQVAEStrategy`, `DiffusionStrategy` (future)

The key rule: **adding a new strategy must not touch the Trainer, logging, or data pipeline**. If you find yourself writing `if isinstance(strategy, SomeStrategy)` in the Trainer, the pattern is broken.

## Why here

We're comparing three fundamentally different generative architectures. Without this pattern, you'd end up with either:

1. **Three separate training scripts** — duplicated boilerplate, bugs fixed in one but not the others
2. **One monolithic script with conditionals** — `if strategy == "vae": ...` scattered everywhere, impossible to reason about

The strategy pattern gives us a third option: shared infrastructure with swappable model logic. This is especially valuable for a learning project because you can see exactly what's different between VAE, VQ-VAE, and Diffusion — it's everything inside the strategy, and nothing outside it.

## How

### The contract (`src/strategies/base.py`)

```python
class GenerativeStrategy(ABC):
    @abstractmethod
    def build_model(self, config: DictConfig) -> nn.Module: ...

    @abstractmethod
    def train_step(self, model, optimizer, batch) -> dict[str, float]: ...

    @abstractmethod
    def sample(self, model, n_samples, device) -> torch.Tensor: ...

    @abstractmethod
    def get_metrics(self, model, batch) -> dict[str, float]: ...
```

Four methods. That's the entire interface.

**`build_model`** receives the full config and returns an `nn.Module`. The strategy decides what model to build — the Trainer just calls `.to(device)` on whatever it gets back.

**`train_step`** receives the model, optimizer, and a batch. The strategy owns the entire forward-backward pass. This is critical — each architecture computes loss differently:

| Strategy | Loss components |
|----------|----------------|
| VAE | `recon_loss + kl_weight * kl_loss` |
| VQ-VAE | `recon_loss + commitment_weight * vq_loss` |
| Diffusion | `noise_prediction_mse` (future) |

The Trainer doesn't interpret these — it just logs whatever keys the dict contains.

**`sample`** generates new images. Each architecture samples differently:
- VAE: decode random vectors from N(0,1)
- VQ-VAE: decode random codebook indices (naive) or sample from a learned prior
- Diffusion: iterative denoising from pure noise

**`get_metrics`** computes evaluation metrics without gradient tracking. Used for validation.

### The Trainer (`src/trainer.py`)

The Trainer's training loop is intentionally generic:

```python
for epoch in range(cfg.epochs):
    for batch in loader:
        loss_dict = self.strategy.train_step(self.model, self.optimizer, batch)
        # ... log loss_dict to MLflow
    if (epoch + 1) % cfg.sample_interval == 0:
        self._log_samples(epoch)
```

It doesn't know what `loss_dict` contains. It doesn't know how sampling works. It just calls the strategy methods and logs the results. This is the strategy pattern doing its job.

### The registry (`scripts/train.py`)

```python
STRATEGY_REGISTRY = {
    "dummy": DummyStrategy,
    "vae": VAEStrategy,
    "vqvae": VQVAEStrategy,
}
```

Config files specify `strategy: vae` and the registry maps it to the right class. Adding a new strategy means: implement the ABC, add one line to the registry. Done.

### Why `loss_dict` and not a scalar

Each strategy returns differently-named losses. The VAE returns `recon_loss` and `kl_loss`. The VQ-VAE returns `recon_loss`, `vq_loss`, and `codebook_utilization`. If the Trainer expected a single scalar loss, we'd lose this granularity. By returning a dict, MLflow logs all components separately — you can plot `recon_loss` across strategies to compare reconstruction quality, even though the total loss formulas are completely different.

## Gotchas

**Strategy owns the backward pass.** The Trainer doesn't call `loss.backward()` — the strategy does, inside `train_step`. This gives each strategy full control over gradient computation, which matters for:
- VQ-VAE's straight-through estimator (gradients bypass the quantization step)
- Diffusion's noise prediction (loss is computed on predicted vs. actual noise, not on the image directly)

**Strategies can hold state.** `VAEStrategy` stores `kl_weight`, `VQVAEStrategy` stores `commitment_weight`. These are set in `build_model` from the config. The model itself stays a pure `nn.Module` — it doesn't know about loss weighting.

**The smoke test enforces the contract.** `scripts/smoke_test.py` instantiates every registered strategy, runs `train_step`, `sample`, and `get_metrics`, and checks return types and shapes. Run it after any model code change. If a strategy breaks the ABC contract, you'll know in seconds, not after a 30-minute training run.
