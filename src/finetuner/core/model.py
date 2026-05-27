"""Model Factory for loading and configuring LLM Guardrail models."""

import json
from typing import Any

from gliner2 import GLiNER2
from huggingface_hub import hf_hub_download

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LogType
from src.finetuner.core.tracker import TrainingTracker


class ModelFactory:
    """Factory for instantiating safety moderation models and tokenizers.

    This class decouples model loading mechanics from trainer algorithms,
    implementing Inversion of Control for framework loading.

    Attributes:
        config (GlobalConfig): Target hyperparameters and backend preferences.
        tracker (TrainingTracker): Observability logger.
    """

    def __init__(self, config: GlobalConfig, tracker: TrainingTracker) -> None:
        """Initializes the model factory.

        Args:
            config (GlobalConfig): Hyperparameters and backend settings.
            tracker (TrainingTracker): Activity tracker.
        """
        self.config: GlobalConfig = config
        self.tracker: TrainingTracker = tracker

    def load_model(self) -> Any:
        """Loads and returns the model.

        Returns:
            Loaded model instance.
        """
        model_id = self.config.model_id
        self.tracker.add_log(LogType.TOOL, f"Loading GLiNER2 model: {model_id}")
        model = GLiNER2.from_pretrained(model_id)
        self.tracker.add_log(LogType.RESPONSE, f"Loaded GLiNER2 model: {model_id}")
        return model

    def _resolve_encoder_id(self, model_id: str) -> str:
        """Reads the underlying encoder model ID from a GLiNER2 config, if available.

        High level role: Dynamic encoder backbone resolver.
        Description: This method downloads and inspects the 'encoder_config/config.json' file
        for the given model ID on Hugging Face Hub. If found, it parses the JSON configuration
        and retrieves the underlying encoder model ID
        (e.g., '_name_or_path' or 'model_name_or_path').
        This ensures downstream loading functions utilize the correct base encoder weights
        (e.g., DeBERTa).
        It falls back to the top-level model ID if the configuration is missing, corrupt,
        or does not specify a base encoder path.

        How it works:
        1. Calls `hf_hub_download` to get the local path of 'encoder_config/config.json'.
        2. Opens and parses the JSON file.
        3. Looks up key '_name_or_path' followed by 'model_name_or_path'.
        4. If a valid ID is resolved, it logs the resolution and returns it.
        5. If any exception occurs (e.g., network failure, missing file, JSON parsing error),
           it logs the exception using the granular tracker and falls back to the original model ID.

        Args:
            model_id (str): The top-level Hugging Face model identifier.

        Returns:
            str: The resolved encoder backbone model ID if available,
                otherwise the original model_id.

        Raises:
            None: All exceptions are caught, logged via the tracker, and handled gracefully.

        Examples:
            >>> factory = ModelFactory(config, tracker)
            >>> resolved = factory._resolve_encoder_id("limjiqi/gliner2-base")
            >>> print(resolved)
            microsoft/deberta-v3-base
        """
        try:
            cfg_path = hf_hub_download(model_id, filename="encoder_config/config.json")  # nosec B615
            with open(cfg_path) as f:
                encoder_cfg = json.load(f)
            encoder_id = encoder_cfg.get("_name_or_path") or encoder_cfg.get("model_name_or_path")
            if encoder_id:
                self.tracker.add_log(LogType.TOOL, f"Resolved encoder backbone: {encoder_id}")
                return encoder_id
        except Exception as err:
            self.tracker.add_log(
                LogType.ERROR,
                f"Failed to resolve encoder backbone for {model_id}: {str(err)}. "
                "Falling back to original model ID.",
            )
        return model_id
