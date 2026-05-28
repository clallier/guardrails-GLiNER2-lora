"""CLI entry point for launching a finetuning training run with optional WandB logging."""

import argparse
import os

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LogType
from src.finetuner.core.model import ModelFactory
from src.finetuner.core.tracker import TrainingTracker
from src.finetuner.core.trainer import PyTorchTrainer
from src.finetuner.data.dataset import DatasetPipeline


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the training run.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(description="GLIGuard LLM Guardrails Finetuning CLI")
    parser.add_argument("--model-id", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--test-split", type=float, default=None, help="Test split ratio")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb-run", type=str, default=None, help="WandB run name")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> GlobalConfig:
    """Constructs a GlobalConfig from CLI args, falling back to defaults.

    High level role: CLI parser adapter. Integrates parsed CLI argument namespaces
    into the robust GlobalConfig dataclass structure.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        GlobalConfig: Training configuration.

    Raises:
        None.

    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(
            no_wandb=False,
            model_id=None,
            epochs=None,
            batch_size=None,
            lr=None,
            test_split=None,
            wandb_project=None,
            wandb_run=None)
        >>> config = _build_config(args)
    """
    kwargs = {"use_wandb": not args.no_wandb}
    if args.model_id:
        kwargs["model_id"] = args.model_id
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
    if args.lr:
        kwargs["learning_rate"] = args.lr
    if args.test_split:
        kwargs["test_split"] = args.test_split
    if args.wandb_project:
        kwargs["wandb_project"] = args.wandb_project
    if args.wandb_run:
        kwargs["wandb_run_name"] = args.wandb_run
    return GlobalConfig(**kwargs)


_TRAIN_CACHE = "data/train.jsonl"
_VAL_CACHE = "data/valid.jsonl"


def _data_cache_exists() -> bool:
    """Returns True when both local JSONL splits are already on disk.

    Returns:
        bool: True if data/train.jsonl and data/valid.jsonl both exist.
    """
    return os.path.exists(_TRAIN_CACHE) and os.path.exists(_VAL_CACHE)


def main() -> None:
    """Runs the full dataset preparation and finetuning pipeline from the CLI.

    Automatically reuses cached data/train.jsonl and data/valid.jsonl when they
    already exist on disk, skipping the HuggingFace download and dedup pipeline.
    """
    args = _parse_args()
    config = _build_config(args)
    tracker = TrainingTracker(config)

    tracker.add_log(LogType.SYSTEM, f"Starting CLI training run. WandB={config.use_wandb}")

    pipeline = DatasetPipeline(config, tracker)
    if _data_cache_exists():
        train_ds, test_ds, val_ds = pipeline.load_from_cache(_TRAIN_CACHE, _VAL_CACHE)
    else:
        train_ds, test_ds, val_ds = pipeline.prepare_dataset()

    factory = ModelFactory(config, tracker)
    trainer = PyTorchTrainer(config, tracker, factory)

    metrics = trainer.train(train_ds, test_ds)
    tracker.add_log(LogType.SYSTEM, f"Training complete. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
