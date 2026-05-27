import unittest
from unittest.mock import MagicMock, patch
from unittest.mock import mock_open as unittest_mock_open

from src.finetuner.constants import (
    BASE_MODEL_ID,
    CLASSIFICATION_TASK,
    LABEL_NAMES,
    LABEL_SAFE,
    LABEL_UNSAFE,
)
from src.streamlit_app.api.client import FinetunerClient


class TestFinetunerClient(unittest.TestCase):
    """Verifies that the Streamlit API client handles threading and inference."""

    def setUp(self) -> None:
        """Sets up target active components."""
        self.client = FinetunerClient()

    def test_training_inactive_initially(self) -> None:
        """Verifies that no active background job exists upon initialization."""
        self.assertFalse(self.client.is_training_active())
        self.assertFalse(self.client.has_training_started())

    @patch("src.finetuner.core.inference.GLiNER2.from_pretrained")
    def test_evaluate_text(self, mock_from_pretrained: MagicMock) -> None:
        """Verifies that evaluate_text returns correct schema and results."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.classify_text.return_value = {
            CLASSIFICATION_TASK: {"label": LABEL_NAMES[LABEL_UNSAFE], "confidence": 1.0}
        }
        mock_from_pretrained.return_value = mock_model

        res = self.client.evaluate_text("mock/model", "unsafe prompt")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(res["verdict"], LABEL_NAMES[LABEL_UNSAFE])
        mock_from_pretrained.assert_called_once_with(BASE_MODEL_ID)
        mock_model.load_adapter.assert_called_once_with("./mock/model")
        mock_model.classify_text.assert_called_once()

    @patch("src.finetuner.core.inference.GLiNER2.from_pretrained")
    @patch("os.path.exists")
    @patch("builtins.open")
    def test_evaluate_suite(
        self, mock_open: MagicMock, mock_exists: MagicMock, mock_from_pretrained: MagicMock
    ) -> None:
        """Verifies that the batch validation evaluation suite runs successfully."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.batch_classify_text.return_value = [
            {CLASSIFICATION_TASK: {"label": LABEL_NAMES[LABEL_SAFE], "confidence": 1.0}},
            {CLASSIFICATION_TASK: {"label": LABEL_NAMES[LABEL_UNSAFE], "confidence": 1.0}},
        ]
        mock_from_pretrained.return_value = mock_model

        # Mock the jsonl dataset presence and reading
        mock_exists.return_value = True

        mock_file_content = """{"text": "safe prompt", "label": 0}
        {"text": "unsafe prompt", "label": 1}"""
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__iter__.return_value = iter(mock_file_content.splitlines())
        mock_open.return_value = mock_file

        res = self.client.evaluate_suite("mock/model", max_samples=10)
        self.assertEqual(res["status"], "success")

        metrics = res["metrics"]
        self.assertEqual(metrics["total_evaluated"], 2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
