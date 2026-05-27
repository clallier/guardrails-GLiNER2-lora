"""Observability tracker implementing granular logging and metric streaming."""

import logging
import time
from typing import Any, Dict, List

import wandb
from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LOG_FORMAT, LogType


class TrainingTracker:
    """Handles structured application logs and metric reporting.

    This class encapsulates standard terminal logging, in-memory logs for
    Streamlit interface streaming, and integration with Weights & Biases (wandb).

    Attributes:
        config (GlobalConfig): Configuration parameters for the run.
        logs (List[Dict[str, Any]]): Chronological in-memory log entries.
        logger (logging.Logger): Python logger instance.
    """

    def __init__(self, config: GlobalConfig) -> None:
        """Initializes the training tracker with configuration.

        Args:
            config (GlobalConfig): Run hyperparameter configuration.
        """
        self.config: GlobalConfig = config
        self.logs: List[Dict[str, Any]] = []

        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        self.logger: logging.Logger = logging.getLogger("FinetunerTracker")

        self._init_wandb()

    def add_log(self, category: LogType, message: str) -> None:
        """Logs an action or status update into memory and standard streams.

        Args:
            category (LogType): The log type (e.g., LogType.REQUEST, LogType.RESPONSE, ...).
            message (str): Plain-text details of the logged event.

        Examples:
            >>> tracker = TrainingTracker(GlobalConfig())
            >>> tracker.add_log(LogType.TRAIN, "Epoch 1 Completed")
        """
        entry = {
            "timestamp": time.time(),
            "category": category,
            "message": message,
        }
        self.logs.append(entry)
        self.logger.info("[%s] %s", category, message)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """Streams quantitative training/evaluation metrics to logs and wandb.

        High level role: Metrics streaming dispatcher. Logs metrics to standard
        tracker logs and dispatch to WandB if an active run exists.

        Args:
            step (int): The current training step iteration.
            metrics (Dict[str, Any]): Dictionary of metric name to scalar value.
        """
        self.add_log(LogType.METRIC, f"Step {step} - Metrics: {metrics}")
        if self.config.use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)

    def get_logs(self) -> List[Dict[str, Any]]:
        """Retrieves history of structured logs.

        Returns:
            List[Dict[str, Any]]: List of log entries.
        """
        return self.logs

    def _init_wandb(self) -> None:
        """Initializes Weights & Biases connection if enabled by configuration.

        High level role: WandB session initiator. Gracefully configures and initializes
        the Weights & Biases tracker, falling back to disabled mode on connection/auth errors.
        """
        if not self.config.use_wandb:
            return

        try:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
            self.add_log(LogType.SYSTEM, "Successfully initialized Weights & Biases session.")
        except Exception as err:
            self.add_log(LogType.ERROR, f"WandB init failed: {str(err)}. Disabling wandb logging.")
            self.config.use_wandb = False
