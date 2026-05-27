"""Unit tests for DatasetPipeline cleaning, normalization, and partitioning."""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.finetuner.config import GlobalConfig
from src.finetuner.constants import LABEL_SAFE, LABEL_UNSAFE
from src.finetuner.core.tracker import TrainingTracker
from src.finetuner.data.dataset import DatasetPipeline


class TestDatasetPipeline(unittest.TestCase):
    """Verifies safety label normalizations and schema column detection."""

    def setUp(self) -> None:
        """Sets up running components."""
        self.config = GlobalConfig()
        self.tracker = MagicMock(spec=TrainingTracker)
        self.pipeline = DatasetPipeline(self.config, self.tracker)

    def test_normalize_labels(self) -> None:
        """Verifies string, boolean, and numeric inputs convert to 0 and 1."""
        self.assertEqual(self.pipeline._normalize_label("unsafe"), LABEL_UNSAFE)
        self.assertEqual(self.pipeline._normalize_label("safe"), LABEL_SAFE)
        self.assertEqual(self.pipeline._normalize_label("true"), LABEL_UNSAFE)
        self.assertEqual(self.pipeline._normalize_label(True), LABEL_UNSAFE)
        self.assertEqual(self.pipeline._normalize_label("injection"), LABEL_UNSAFE)
        self.assertEqual(self.pipeline._normalize_label(0), LABEL_SAFE)

    def test_detect_columns(self) -> None:
        """Verifies prompt text and target label classification column resolution."""
        cols = ["sentence", "id", "class"]
        text_col, label_col = self.pipeline._detect_columns(cols)
        self.assertEqual(text_col, "sentence")
        self.assertEqual(label_col, "class")

    def test_clean_and_deduplicate(self) -> None:
        """Verifies duplicate text fields are cleanly pruned from dataframe."""
        df = pd.DataFrame([
            {"text": "injection test", "label": 1},
            {"text": "injection test", "label": 1},
            {"text": "safe test", "label": 0},
        ])
        df_cleaned = self.pipeline._clean_and_deduplicate(df)
        self.assertEqual(len(df_cleaned), 2)

    def test_inject_harmless_presets(self) -> None:
        """Verifies harmless safe conversational presets are dynamically appended."""
        records = []
        self.pipeline._inject_harmless_presets(records)
        self.assertGreater(len(records), 0)
        self.assertTrue(all(r["label"] == LABEL_SAFE for r in records))
        self.assertIn("say hi", [r["text"] for r in records])
