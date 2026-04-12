"""Thin MLflow wrapper so the Trainer never imports mlflow directly."""

import logging
from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Wraps MLflow tracking calls for the training loop."""

    def __init__(self, config: DictConfig, strategy_name: str) -> None:
        tracking_uri = config.mlflow.tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(strategy_name)
        self.run = mlflow.start_run()
        mlflow.log_params(
            OmegaConf.to_container(config, resolve=True, throw_on_missing=True)  # type: ignore[arg-type]
        )
        logger.info(
            "MLflow run started: experiment=%s run_id=%s",
            strategy_name,
            self.run.info.run_id,
        )

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path) -> None:
        mlflow.log_artifact(str(local_path))

    def end_run(self) -> None:
        mlflow.end_run()
        logger.info("MLflow run ended.")
