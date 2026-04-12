"""Config loading with OmegaConf — merges base + strategy configs."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """Load a strategy config, merging with base.yaml if 'defaults' key is present."""
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)

    if "defaults" in cfg:
        base_names = cfg.pop("defaults")
        config_dir = Path(config_path).parent
        bases = []
        for name in base_names:
            base_path = config_dir / f"{name}.yaml"
            bases.append(OmegaConf.load(str(base_path)))
        merged = OmegaConf.merge(*bases, cfg)
        assert isinstance(merged, DictConfig)
        return merged

    return cfg
