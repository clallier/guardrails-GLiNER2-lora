"""Finetuning hyperparameter configuration state container."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.finetuner.constants import BASE_MODEL_ID, DEFAULT_WANDB_PROJECT


@dataclass
class GlobalConfig:
    """Hyperparameter and hardware state configuration container for GLIGuard.

    High level role: Encapsulates all global configuration parameters, hyperparameter variables,
    platform preferences (e.g. MLX vs PyTorch), and logging integrations for training runs.
    It acts as the single source of truth for execution parameters.

    Attributes:
        model_id (str): The Hugging Face model identifier to finetune. Defaults to BASE_MODEL_ID.
        learning_rate (float): The optimizer learning rate. Defaults to 1e-5.
        batch_size (int): Training batch size per device. Defaults to 4.
        epochs (int): Number of training epochs. Defaults to 2.
        test_split (float): Ratio of dataset partitioned for testing (0.0 to 1.0). Defaults to 0.1
        seed (int): Random seed for reproducibility. Defaults to 0.
        use_wandb (bool): Flag to enable tracking via Weights & Biases. Defaults to True.
        wandb_project (str): Wandb project grouping. Defaults to DEFAULT_WANDB_PROJECT.
        wandb_run_name (Optional[str]): Optional custom WandB run name. Defaults to None.
        steps_per_eval (int): Frequency of validation evaluation in steps. Defaults to 100.
        steps_per_report (int): Frequency of logging report printouts in steps. Defaults to 10.
        adapter_path (str): File system directory to save model adapters or weights.
            Defaults to "adapters".

    Examples:
        >>> config = GlobalConfig(batch_size=8)
        >>> print(config.batch_size)
        8
    """

    model_id: str = BASE_MODEL_ID
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 2
    test_split: float = 0.1
    seed: int = 0
    use_wandb: bool = True
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_run_name: Optional[str] = None
    steps_per_eval: int = 100
    steps_per_report: int = 10
    adapter_path: str = "adapters"

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration state into a dictionary representation.

        High level role: Serializer method. Converts the dataclass properties into
        a flat python dictionary for integration with external APIs such as Weights & Biases.

        Args:
            None.

        Returns:
            Dict[str, Any]: Hyperparameter settings dictionary mapping attribute names to values.

        Raises:
            None.

        Examples:
            >>> config = GlobalConfig(batch_size=8)
            >>> data = config.to_dict()
            >>> data["batch_size"]
            8
        """
        return {
            "model_id": self.model_id,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "test_split": self.test_split,
            "seed": self.seed,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "wandb_run_name": self.wandb_run_name,
            "steps_per_eval": self.steps_per_eval,
            "steps_per_report": self.steps_per_report,
            "adapter_path": self.adapter_path,
        }
