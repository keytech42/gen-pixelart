"""Smoke test: instantiate each strategy, run one train_step, call sample, verify shapes."""

import logging
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from src.strategies.base import GenerativeStrategy
from src.strategies.dummy import DummyStrategy
from src.strategies.vae import VAEStrategy
from src.strategies.vqvae import VQVAEStrategy
from src.strategies.diffusion import DiffusionStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES: dict[str, type[GenerativeStrategy]] = {
    "dummy": DummyStrategy,
    "vae": VAEStrategy,
    "vqvae": VQVAEStrategy,
    "diffusion": DiffusionStrategy,
}

TEST_CONFIG = OmegaConf.create({
    "dataset": {"path": "data/sprites", "image_size": 16, "palette_size": 8},
    "model": {
        "latent_dim": 32,
        "encoder_channels": [16, 32],
        "decoder_channels": [32, 16],
        "kl_weight": 0.001,
        "codebook_size": 64,
        "codebook_dim": 32,
        "commitment_weight": 0.25,
        "channels": [32, 64],
        "num_res_blocks": 1,
        "timesteps": 10,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "time_emb_dim": 64,
        "sampling_method": "ddim",
        "sampling_steps": 5,
    },
})

IMAGE_SIZE = 16
BATCH_SIZE = 4
CHANNELS = 3
N_SAMPLES = 4
DEVICE = torch.device("cpu")


def test_strategy(name: str, strategy: GenerativeStrategy) -> bool:
    logger.info("Testing %s ...", name)
    try:
        model = strategy.build_model(TEST_CONFIG).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch = torch.rand(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

        # train_step
        loss_dict = strategy.train_step(model, optimizer, batch)
        assert isinstance(loss_dict, dict), f"train_step should return dict, got {type(loss_dict)}"
        for k, v in loss_dict.items():
            assert isinstance(k, str), f"loss key should be str, got {type(k)}"
            assert isinstance(v, (int, float)), f"loss value for '{k}' should be numeric, got {type(v)}"
        logger.info("  train_step OK: %s", loss_dict)

        # sample
        samples = strategy.sample(model, N_SAMPLES, DEVICE)
        assert samples.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES} samples, got {samples.shape[0]}"
        logger.info("  sample OK: shape=%s", samples.shape)

        # get_metrics
        metrics = strategy.get_metrics(model, batch)
        assert isinstance(metrics, dict), f"get_metrics should return dict, got {type(metrics)}"
        logger.info("  get_metrics OK: %s", metrics)

        logger.info("  %s PASSED", name)
        return True

    except Exception:
        logger.exception("  %s FAILED", name)
        return False


def main() -> None:
    results = {}
    for name, cls in STRATEGIES.items():
        results[name] = test_strategy(name, cls())

    logger.info("--- Results ---")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %s: %s", name, status)
        if not passed:
            all_passed = False

    if not all_passed:
        logger.error("Some strategies failed!")
        sys.exit(1)
    logger.info("All strategies passed.")


if __name__ == "__main__":
    main()
