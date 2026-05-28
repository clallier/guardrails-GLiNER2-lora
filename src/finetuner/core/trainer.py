from typing import Any, Dict, List, cast

import torch
from datasets import Dataset
from gliner2.training.data import Classification, InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import (
    CLASSIFICATION_LABELS,
    CLASSIFICATION_TASK,
    LABEL_NAMES,
    LABEL_SAFE,
    LABEL_UNSAFE,
    LogType,
)
from src.finetuner.core.model import ModelFactory
from src.finetuner.core.tracker import TrainingTracker


class PyTorchTrainer:
    """Finetuning execution engine utilizing GLiNER2Trainer and PyTorch.

    This class prepares a standard safety classification dataset into a format
    compatible with GLiNER2 schema-conditioned inputs and executes parameter-efficient
    LoRA adapter fine-tuning.

    Attributes:
        config (GlobalConfig): Target training parameters.
        tracker (TrainingTracker): Activity tracker and metrics logger.
        model_factory (ModelFactory): Loaded model factory interface.
    """

    def __init__(
        self,
        config: GlobalConfig,
        tracker: TrainingTracker,
        model_factory: ModelFactory,
    ) -> None:
        """Initializes the PyTorch trainer with dependency injection.

        High level role: Dependency injector and trainer state initializer.
        Description: This constructor injects the configuration preferences,
        observability tracker, and dynamic model factory interface, storing them
        locally as attributes for the training execution lifecycle.

        Args:
            config (GlobalConfig): Hyperparameter and run preferences.
            tracker (TrainingTracker): Tracker instance for granular metrics and log streaming.
            model_factory (ModelFactory): Dynamic factory loaded with target weights.
        """
        self.config: GlobalConfig = config
        self.tracker: TrainingTracker = tracker
        self.model_factory: ModelFactory = model_factory

    def train(self, train_ds: Dataset, test_ds: Dataset) -> Dict[str, Any]:
        """Finetunes the GLiNER2 guardrail model using LoRA adapters.

        This method converts the input datasets to schema-driven InputExample instances,
        configures a LoRA-enabled TrainingConfig, instantiates a GLiNER2Trainer, and
        launches the training run.

        Args:
            train_ds (Dataset): Standardized HuggingFace training set
            containing 'text' and 'label' columns.
            val_ds (Dataset): Standardized HuggingFace validation set.

        Returns:
            Dict[str, Any]: Performance and optimization metrics dictionary.

        Raises:
            ImportError: If gliner2 is not installed correctly.
            Exception: If the underlying training execution fails.

        Examples:
            >>> trainer = PyTorchTrainer(config, tracker, factory)
            >>> results = trainer.train(train_dataset, val_dataset)
            >>> print(results["best_metric"])
        """
        self.tracker.add_log(LogType.REQUEST, "Initializing PyTorch GLiNER2 LoRA training pipeline")
        model = self.model_factory.load_model()

        # 1. Structure data as InputExample instances
        train_examples = self._to_input_examples(train_ds)
        val_examples = self._to_input_examples(test_ds)

        # 2. Configure Training & LoRA
        config = self._create_training_config()

        # 3. Instantiate Trainer & Run
        self.tracker.add_log(LogType.TRAIN, "Starting gliner2.GLiNER2Trainer execution loop.")
        trainer = GLiNER2Trainer(model=model, config=config)

        results = trainer.train(
            train_data=train_examples,
            eval_data=val_examples,
        )

        self.tracker.log_metrics(step=self.config.epochs, metrics=results)
        self.tracker.add_log(LogType.RESPONSE, "PyTorch native GLiNER2 LoRA training run complete.")
        return results

    def _to_input_examples(self, ds: Dataset) -> List[Any]:
        """Converts a standard Hugging Face Dataset into the GLiNER2 training format.

        High level role: Dataset formatter and schema mapper.

        Args:
            ds (Dataset): The Hugging Face dataset instance containing 'text' (str) and
                'label' (int) columns. No default value.

        Returns:
            List[Any]: List of InputExample objects.
        """
        self.tracker.add_log(
            LogType.TOOL, f"Mapping {len(ds)} dataset records to InputExample structures."
        )
        return [self._to_input_example(cast(Dict[str, Any], r)) for r in ds]

    def _to_input_example(self, row: Dict[str, Any]) -> InputExample:
        """Converts a single dataset row to a GLiNER2 InputExample.

        High level role: Single row schema-to-InputExample mapper.

        Args:
            row (Dict[str, Any]): Dataset row dict containing 'text' and 'label'.

        Returns:
            InputExample: Target training instance.
        """
        text = str(row["text"]).strip()
        label_val = int(row["label"])
        label_name = (
            LABEL_NAMES[LABEL_UNSAFE] if label_val == LABEL_UNSAFE else LABEL_NAMES[LABEL_SAFE]
        )
        return InputExample(
            text=text,
            classifications=[
                Classification(
                    task=CLASSIFICATION_TASK,
                    labels=CLASSIFICATION_LABELS,
                    true_label=label_name,
                )
            ],
        )

    def _create_training_config(self) -> TrainingConfig:
        """Helper to create a configured TrainingConfig instance.

        High level role: Training hyperparameters to TrainingConfig constructor.

        Returns:
            TrainingConfig: Configured training config object.
        """
        self.tracker.add_log(LogType.TOOL, "Configuring LoRA TrainingConfig parameters.")
        use_mps = torch.backends.mps.is_available()
        cfg_dict = self._get_config_dict(use_mps)
        return TrainingConfig(**cfg_dict)

    def _get_lora_params(self) -> Dict[str, Any]:
        """Gets dictionary of LoRA adaptation parameters.

        High level role: LoRA parameter dict builder.

        Returns:
            Dict[str, Any]: LoRA config dictionary.
        """
        return {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16.0,
            "lora_dropout": 0.0,
            "lora_target_modules": ["encoder"],
            "save_adapter_only": True,
        }

    def _get_config_dict(self, use_mps: bool) -> Dict[str, Any]:
        """Gets a flat dictionary representation of the training configuration.

        High level role: Configuration dictionary builder.

        Args:
            use_mps (bool): Whether to use MPS hardware acceleration.

        Returns:
            Dict[str, Any]: Combined training config parameters.
        """
        params = {
            "output_dir": self.config.adapter_path,
            "experiment_name": self.config.wandb_run_name or CLASSIFICATION_TASK,
            "num_epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "encoder_lr": self.config.learning_rate,
            "task_lr": 5e-4,
            "logging_steps": self.config.steps_per_report,
            "report_to_wandb": self.config.use_wandb,
            "wandb_project": self.config.wandb_project,
        }
        params.update(self._get_lora_params())
        params.update(self._get_device_specific_params(use_mps))
        return params

    def _get_device_specific_params(self, use_mps: bool) -> Dict[str, Any]:
        """Resolves training config parameters specific to execution hardware.

        High level role: Device parameters resolver.

        Args:
            use_mps (bool): True if using MPS hardware acceleration.
                No default value.

        Returns:
            Dict[str, Any]: Dict containing fp16, bf16, num_workers, pin_memory values.

        Examples:
            >>> params = trainer._get_device_specific_params(True)
            >>> print(params["num_workers"])
            0
        """
        if use_mps:
            return {
                "fp16": False,
                "bf16": False,
                "num_workers": 0,
                "pin_memory": False,
            }
        return {
            "fp16": True,
            "bf16": False,
            "num_workers": 4,
            "pin_memory": True,
        }
