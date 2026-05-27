"""Module for handling safety classifier loading, caching, and text evaluation."""

import json
import os
from typing import Any, Dict, List

import torch
from gliner2 import GLiNER2

from src.finetuner.constants import (
    BASE_MODEL_ID,
    CLASSIFICATION_LABELS,
    CLASSIFICATION_TASK,
    DEFAULT_THRESHOLD,
    EVAL_BATCH_SIZE,
    LABEL_NAMES,
    LABEL_SAFE,
    LABEL_UNSAFE,
    LogType,
)
from src.finetuner.core.device import DeviceResolver
from src.finetuner.core.tracker import TrainingTracker


class InferenceEngine:
    """Manages safety classifier model loading, caching, and inference operations.

    High level role: Inference processor and model cache manager.
    Description: This class encapsulates model loading mechanics, maintains an active
    in-memory cache of loaded models/adapters, and executes single-prompt and batched
    safety classification pipelines, logging operations to the evaluation tracker.

    Attributes:
        tracker (TrainingTracker): The log tracker instance for evaluation events.
        _loaded_models (Dict[str, Any]): In-memory cache map of loaded models.
    """

    def __init__(self, tracker: TrainingTracker) -> None:
        """Initializes the inference engine with dependency injection.

        High level role: Constructor.
        Description: Sets up the persistent logs tracker and the models cache map.

        Args:
            tracker (TrainingTracker): The evaluation tracker instance.
        """
        self.tracker: TrainingTracker = tracker
        self._loaded_models: Dict[str, Any] = {}

    def clear_cache(self) -> None:
        """Clears all cached model instances from memory.

        High level role: Cache manager.
        Description: Evicts all in-memory loaded models to free resources.
        """
        self._loaded_models.clear()

    def evaluate_text(
        self, model_id: str, prompt: str, threshold: float = DEFAULT_THRESHOLD
    ) -> Dict[str, Any]:
        """Loads safety classifier and evaluates user input text for safety classification.

        High level role: Single text safety assessor.
        Description: Runs inference on a single prompt and flags unsafe content based on threshold.

        Args:
            model_id (str): Hugging Face model ID or adapter path.
            prompt (str): Text prompt to classify.
            threshold (float): Unsafe threshold (default: DEFAULT_THRESHOLD).

        Returns:
            Dict[str, Any]: Status, score, and safe/unsafe verdict.

        Raises:
            Exception: If inference fails.

        Examples:
            >>> engine.evaluate_text("adapters/best", "test prompt")
        """
        try:
            self.tracker.add_log(
                LogType.REQUEST, f"Evaluating safety for prompt: '{prompt[:40]}...'"
            )
            model = self.get_model(model_id)
            tasks = {CLASSIFICATION_TASK: CLASSIFICATION_LABELS}
            with torch.inference_mode():
                verdict_dict = model.classify_text(prompt, tasks, include_confidence=True)
            unsafe_prob = self.get_unsafe_probability(verdict_dict)
            verdict_str = (
                LABEL_NAMES[LABEL_UNSAFE] if unsafe_prob >= threshold else LABEL_NAMES[LABEL_SAFE]
            )
            self.tracker.add_log(
                LogType.RESPONSE, f"verdict: '{verdict_str}' (unsafe prob: {unsafe_prob:.4f})"
            )
            return {"status": "success", "score": unsafe_prob, "verdict": verdict_str}
        except Exception as err:
            self.tracker.add_log(LogType.ERROR, f"Safety evaluation failed: {str(err)}")
            raise err

    def run_batched_inference(
        self, model: Any, records: List[Dict[str, Any]], progress_callback: Any = None
    ) -> List[Dict[str, Any]]:
        """Runs batch inference over a list of records.

        High level role: Batched validation inference orchestrator.
        Description: Splits dataset records into batches, processes them sequentially,
        and invokes a callback to report progress.

        Args:
            model (Any): The loaded GLiNER2 model.
            records (List[Dict[str, Any]]): Dataset records to process.
            progress_callback (Any, optional): Streamlit progress indicator callback.

        Returns:
            List[Dict[str, Any]]: List of safety verdicts.

        Examples:
            >>> engine.run_batched_inference(model, records, bar.progress)
        """
        verdicts = []
        total = len(records)
        tasks = {CLASSIFICATION_TASK: CLASSIFICATION_LABELS}
        for i in range(0, total, EVAL_BATCH_SIZE):
            batch = records[i : i + EVAL_BATCH_SIZE]
            res = self._process_batch(model, batch, tasks)
            verdicts.extend(res)
            if progress_callback:
                progress_callback(min(1.0, (i + len(batch)) / total))
        return verdicts

    def get_unsafe_probability(self, verdict: Dict[str, Any]) -> float:
        """Extracts the floating-point probability score of the unsafe class.

        High level role: Verdict probability decoder.
        Description: Extracts confidence scores and normalizes them for binary classification.

        Args:
            verdict (Dict[str, Any]): Raw prediction dictionary from the classifier.

        Returns:
            float: Unsafe probability score (0.0 to 1.0).

        Examples:
            >>> engine.get_unsafe_probability(
            ...     {"prompt_safety": {"label": "unsafe", "confidence": 0.95}}
            ... )
            0.95
        """
        res_val = verdict.get(CLASSIFICATION_TASK, {})
        label = res_val.get("label", LABEL_NAMES[LABEL_SAFE])
        conf = res_val.get("confidence", 1.0)
        return conf if label == LABEL_NAMES[LABEL_UNSAFE] else (1.0 - conf)

    def load_validation_records(self, max_samples: int) -> List[Dict[str, Any]]:
        """Loads and parses validation jsonl dataset records from disk.

        High level role: Validation dataset parser.
        Description: Reads records from the cached JSONL dataset up to the requested samples limit.

        Args:
            max_samples (int): Maximum validation records to load.

        Returns:
            List[Dict[str, Any]]: Parsed dataset record dictionaries.

        Raises:
            FileNotFoundError: If the dataset file does not exist on disk.

        Examples:
            >>> engine.load_validation_records(200)
        """
        valid_path = "data/valid.jsonl"
        if not os.path.exists(valid_path):
            raise FileNotFoundError(
                "Validation dataset ('data/valid.jsonl') not found. Please run a training job first"
            )

        self.tracker.add_log(LogType.TOOL, f"Loading validation records from: {valid_path}")
        with open(valid_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        return records[:max_samples] if 0 < max_samples < len(records) else records

    def get_model(self, model_id: str) -> Any:
        """Loads and caches model instances with device optimization.

        High level role: Model loader and accelerator.
        Description: This method resolves the target model or adapter by loading from HF
        or caching, then places it on the most suitable hardware device (MPS, CUDA, CPU).

        Args:
            model_id (str): Hugging Face model ID or local directory path.

        Returns:
            Any: The cached or newly loaded GLiNER2 model placed on active device.

        Raises:
            Exception: If model loading fails.

        Examples:
            >>> engine.get_model("fastino/gliguard-LLMGuardrails-300M")
        """
        if model_id in self._loaded_models:
            self.tracker.add_log(LogType.TOOL, f"Model cache hit for: '{model_id}'")
            return self._loaded_models[model_id]
        return self._load_and_cache_model(model_id)

    def _process_batch(
        self, model: Any, batch: List[Dict[str, Any]], tasks: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Executes hardware-accelerated model classification for a single batch of records.

        High level role: Low-level batch processor.
        Description: Calls batch_classify_text in inference mode to compute probabilities.

        Args:
            model (Any): Loaded safety model.
            batch (List[Dict[str, Any]]): Sliced batch of records.
            tasks (Dict[str, Any]): Label configuration dictionary.

        Returns:
            List[Dict[str, Any]]: Classification verdict results.
        """
        prompts = [r.get("text", "") for r in batch]
        with torch.inference_mode():
            return model.batch_classify_text(
                prompts, tasks, batch_size=len(prompts), include_confidence=True
            )

    def _load_and_cache_model(self, model_id: str) -> Any:
        """Helper to load model from HF, load adapter if needed, and move to optimized device.

        High level role: Low-level model loader.
        Description: Resolves baseline, handles adapter mapping, moves weights to device.

        Args:
            model_id (str): Target model path/ID.

        Returns:
            Any: Configured GLiNER2 model.
        """
        self.tracker.add_log(LogType.TOOL, f"Initializing model load for: '{model_id}'")
        try:
            model = GLiNER2.from_pretrained(BASE_MODEL_ID)
            self._load_adapter_if_needed(model, model_id)
            device = DeviceResolver.get_optimal_device()
            model = model.to(device)
            self._loaded_models[model_id] = model
            self.tracker.add_log(LogType.RESPONSE, f"Model '{model_id}' loaded on '{device}'.")
            return model
        except Exception as err:
            self.tracker.add_log(LogType.ERROR, f"Failed to load model '{model_id}': {str(err)}")
            raise err

    def _load_adapter_if_needed(self, model: Any, model_id: str) -> None:
        """Loads a local custom trained adapter onto the baseline model if applicable.

        High level role: Adapter loader and path resolver.
        Description: Checks if the requested model ID is a custom local adapter rather than
        the baseline model, and loads it onto the base GLiNER2 instance.

        Args:
            model (Any): Base GLiNER2 model.
            model_id (str): Hugging Face model ID or path.
        """
        if model_id != BASE_MODEL_ID:
            model.load_adapter(f"./{model_id}")
