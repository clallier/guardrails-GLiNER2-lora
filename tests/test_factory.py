"""Unit tests for ModelFactory backend loading and fallback resolution."""

import unittest
from unittest.mock import MagicMock, patch

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LogType
from src.finetuner.core.model import ModelFactory
from src.finetuner.core.tracker import TrainingTracker


class TestModelFactory(unittest.TestCase):
    """Verifies that ModelFactory handles MLX and PyTorch initializations correctly."""

    def setUp(self) -> None:
        """Sets up target mocks and configurations."""
        self.config = GlobalConfig()
        self.tracker = MagicMock(spec=TrainingTracker)
        self.factory = ModelFactory(self.config, self.tracker)

    @patch("src.finetuner.core.model.GLiNER2.from_pretrained")
    def test_pytorch_model_loading(self, mock_model_load) -> None:
        """Verifies that the PyTorch path initializes AutoModels successfully."""
        mock_model = MagicMock()
        mock_model_load.return_value = mock_model

        model = self.factory.load_model()

        self.assertEqual(model, mock_model)
        self.tracker.add_log.assert_any_call(
            LogType.RESPONSE, f"Loaded GLiNER2 model: {self.config.model_id}"
        )

    @patch("src.finetuner.core.model.hf_hub_download")
    @patch("builtins.open")
    @patch("json.load")
    def test_resolve_encoder_id_success(self, mock_json_load, mock_open, mock_download) -> None:
        """Verifies successful dynamic resolution of the encoder ID from HF config.

        High level role: Success-path validator for encoder backbone resolver.
        Description: This test asserts that when `hf_hub_download` succeeds and the configuration
        contains a valid backbone path under '_name_or_path', the factory correctly parses the
        config and returns the parsed backbone ID. It also verifies that the tool action is logged.

        Args:
            mock_json_load (MagicMock): Mocked json.load function.
            mock_open (MagicMock): Mocked builtins.open function.
            mock_download (MagicMock): Mocked hf_hub_download function.

        Returns:
            None: No return value.

        Raises:
            AssertionError: If returned encoder ID does not match expected mocked value.
        """
        mock_download.return_value = "/dummy/path/encoder_config/config.json"
        mock_json_load.return_value = {"_name_or_path": "microsoft/deberta-v3-base"}

        resolved = self.factory._resolve_encoder_id("limjiqi/gliner2-base")

        self.assertEqual(resolved, "microsoft/deberta-v3-base")
        self.tracker.add_log.assert_any_call(
            LogType.TOOL, "Resolved encoder backbone: microsoft/deberta-v3-base"
        )

    @patch("src.finetuner.core.model.hf_hub_download")
    def test_resolve_encoder_id_failure(self, mock_download) -> None:
        """Verifies fallback behavior and error logging when resolving encoder ID fails.

        High level role: Failure-path validator for encoder backbone resolver.
        Description: This test asserts that when `hf_hub_download` fails (e.g. raises an Exception),
        the factory catches the exception, logs it as an ERROR via the tracker, and gracefully
        falls back to return the original model ID.

        Args:
            mock_download (MagicMock): Mocked hf_hub_download function.

        Returns:
            None: No return value.

        Raises:
            AssertionError: If fallback model ID does not match the input model ID.
        """
        mock_download.side_effect = Exception("HF Hub timeout")

        resolved = self.factory._resolve_encoder_id("limjiqi/gliner2-base")

        self.assertEqual(resolved, "limjiqi/gliner2-base")
        self.tracker.add_log.assert_any_call(
            LogType.ERROR,
            "Failed to resolve encoder backbone for limjiqi/gliner2-base: HF Hub timeout. "
            "Falling back to original model ID.",
        )
