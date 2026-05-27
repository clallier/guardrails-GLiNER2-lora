"""Dedicated API client wrapping training processes and evaluation inference."""

import threading
from typing import Any, Dict, List, Optional

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import DEFAULT_THRESHOLD, LogType
from src.finetuner.core.inference import InferenceEngine
from src.finetuner.core.metrics import MetricsCalculator
from src.finetuner.core.model import ModelFactory
from src.finetuner.core.tracker import TrainingTracker
from src.finetuner.core.trainer import PyTorchTrainer
from src.finetuner.data.dataset import DatasetPipeline


class FinetunerClient:
    """Orchestrates model training threads and evaluation requests for the frontend.

    Encapsulates backend model instantiation and execution loops, providing
    a clean async-like interface for Streamlit UI views.

    Attributes:
        _training_tracker (TrainingTracker): The active training run tracker.
        _eval_tracker (TrainingTracker): The persistent evaluation tracker.
        _active_thread (Optional[threading.Thread]): The background training thread.
        _training_started (bool): States if training has ever started.
        _inference_engine (InferenceEngine): Separated safety classifier inference
            loader and caching module.
        _metrics_calculator (MetricsCalculator): Separated metrics calculator module.
    """

    def __init__(self) -> None:
        """Initializes the backend finetuning client wrapper with dependency composition."""
        self._training_tracker: TrainingTracker = TrainingTracker(GlobalConfig())
        self._eval_tracker: TrainingTracker = TrainingTracker(GlobalConfig(use_wandb=False))
        self._active_thread: Optional[threading.Thread] = None
        self._training_started: bool = False

        self._inference_engine: InferenceEngine = InferenceEngine(self._eval_tracker)
        self._metrics_calculator: MetricsCalculator = MetricsCalculator()

    def start_training(self, config: GlobalConfig) -> TrainingTracker:
        """Spawns a background thread to prepare dataset and run training.

        Args:
            config (GlobalConfig): Target training hyperparameters.

        Returns:
            TrainingTracker: The tracking instance containing real-time logs.
        """
        tracker = TrainingTracker(config)
        self._training_tracker = tracker
        self._training_started = True
        self._inference_engine.clear_cache()

        self._active_thread = threading.Thread(
            target=self._run_training_job, args=(config, tracker), daemon=True
        )
        self._active_thread.start()
        tracker.add_log(LogType.SYSTEM, "Training thread successfully started in background.")
        return tracker

    def is_training_active(self) -> bool:
        """Queries if a background finetuning job is actively running.

        Returns:
            bool: True if the thread is alive, False otherwise.
        """
        return self._active_thread is not None and self._active_thread.is_alive()

    def has_training_started(self) -> bool:
        """Checks if a training run has been initiated during the client session.

        Returns:
            bool: True if training has started, False otherwise.
        """
        return self._training_started

    def get_training_tracker(self) -> TrainingTracker:
        """Gets the active training session tracker.

        Returns:
            TrainingTracker: Active tracker instance.
        """
        return self._training_tracker

    def get_eval_tracker(self) -> TrainingTracker:
        """Gets the active safety tester log tracker instance.

        Returns:
            TrainingTracker: The persistent evaluation tracker.
        """
        return self._eval_tracker

    def evaluate_text(
        self, model_id: str, prompt: str, threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        """Delegates evaluation of user input text to the inference engine.

        Args:
            model_id (str): Model ID or path.
            prompt (str): Text prompt to classify.
            threshold (float): Safety threshold (default: 0.9).

        Returns:
            Dict[str, Any]: Model verdict and score.
        """
        return self._inference_engine.evaluate_text(model_id, prompt, threshold)

    def evaluate_suite(
        self,
        model_id: str,
        max_samples: int = 100,
        threshold: float = 0.5,
        progress_callback: Any = None,
    ) -> Dict[str, Any]:
        """Runs the validation evaluation suite in batches.

        High level role: Validation evaluation suite orchestrator. Delegates loading,
        inference, and metric computation to separate modular engine components.

        Args:
            model_id (str): Hugging Face model ID or path to local trained adapter.
            max_samples (int): Max validation records to evaluate (default: 100).
            threshold (float): Decision threshold to flag unsafe.
            progress_callback (Any): Optional progress callback fraction.

        Returns:
            Dict[str, Any]: Compiled metrics and predictions summary.
        """
        try:
            self.add_log(
                LogType.REQUEST,
                f"Initializing batch validation suite execution (samples: {max_samples}).",
                eval=True,
            )
            model = self._inference_engine.get_model(model_id)
            records = self._inference_engine.load_validation_records(max_samples)

            self.add_log(
                LogType.TOOL, f"Executing batched inference over {len(records)} records.", eval=True
            )
            verdicts = self._inference_engine.run_batched_inference(
                model, records, progress_callback
            )
            metrics = self._calculate_and_log_suite_metrics(records, verdicts, threshold)
            return {"status": "success", "metrics": metrics}
        except Exception as err:
            self.add_log(LogType.ERROR, f"Evaluation suite failed: {str(err)}", eval=True)
            raise err

    def add_log(self, log_type: LogType, message: str, eval: bool = False) -> None:
        """Standard add_log proxy for manual frontend actions.

        High level role: Log entry routing proxy. Forwards the log category and
        message string to the active training or evaluation tracker.

        Args:
            log_type (LogType): Log type (e.g. REQUEST, RESPONSE, TOOL, ERROR, SYSTEM).
            message (str): Plain text log message.
            eval (bool): Route to evaluation tracker if True, training if False.
        """
        tracker = self._eval_tracker if eval else self._training_tracker
        tracker.add_log(log_type, message)

    def _run_training_job(self, config: GlobalConfig, tracker: TrainingTracker) -> None:
        """Private runner method executed in the background thread.

        Args:
            config (GlobalConfig): Target training configurations.
            tracker (TrainingTracker): Run tracker.
        """
        try:
            # 1. Dataset Preparation
            pipeline = DatasetPipeline(config, tracker)
            train_ds, test_ds, val_ds = pipeline.prepare_dataset()

            # 2. Model Factory & Backend Resolution
            factory = ModelFactory(config, tracker)

            # Resolve backend trainer
            trainer = PyTorchTrainer(config, tracker, factory)

            # 3. Ingress Training Loop
            trainer.train(train_ds, test_ds)
            tracker.add_log(LogType.SYSTEM, "Finetuning job completed successfully!")
        except Exception as err:
            tracker.add_log(LogType.ERROR, f"Job thread crashed: {str(err)}")

    def _calculate_and_log_suite_metrics(
        self,
        records: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        """Helper to calculate and log validation suite metrics.

        High level role: Metric calculation and log reporting helper.

        Args:
            records (List[Dict[str, Any]]): True labels list.
            verdicts (List[Dict[str, Any]]): Model predictions list.
            threshold (float): Unsafe threshold.

        Returns:
            Dict[str, Any]: Calculated metrics.
        """
        unsafe_probabilities = [self._inference_engine.get_unsafe_probability(v) for v in verdicts]
        metrics = self._metrics_calculator.compute_metrics(records, unsafe_probabilities, threshold)
        self.add_log(
            LogType.RESPONSE,
            f"Evaluation suite complete. Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1_score']:.4f} (TP: {metrics['tp']}, "
            f"FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']})",
            eval=True,
        )
        return metrics
