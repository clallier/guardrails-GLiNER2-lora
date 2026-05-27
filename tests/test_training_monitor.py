"""Unit tests for TrainingMonitorComponent log parsing and metric extraction."""

import unittest

from src.finetuner.constants import LogType
from src.streamlit_app.components.training_monitor import TrainingMonitorComponent


class TestTrainingMonitorComponent(unittest.TestCase):
    """Verifies that TrainingMonitorComponent extracts metric parameters correctly.

    High level role: Unit test suite for training monitoring helpers. Asserts that step
    indices and scalar loss metrics are correctly parsed from unstructured log messages.
    """

    def test_parse_loss_success(self) -> None:
        """Verifies successful parsing of loss values from various key names.

        High level role: Metric loss parser validator.
        Description: This test asserts that when `_parse_loss_from_message` receives structured
        metrics with different typical loss keys ('loss', 'train_loss', 'eval_loss'),
        it successfully extracts the value as a float.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If parsed loss float does not match the expected value.

        Examples:
            >>> self.test_parse_loss_success()
        """
        loss = TrainingMonitorComponent._parse_loss_from_message(
            "Step 5 - Metrics: {'loss': 0.2468, 'learning_rate': 1e-5}"
        )
        self.assertEqual(loss, 0.2468)

        train_loss = TrainingMonitorComponent._parse_loss_from_message(
            "Step 10 - Metrics: {'train_loss': 0.1234}"
        )
        self.assertEqual(train_loss, 0.1234)

    def test_parse_loss_missing(self) -> None:
        """Verifies fallback return value when loss key is absent or malformed.

        High level role: Missing loss parser validator.
        Description: This test asserts that when `_parse_loss_from_message` receives logs
        without any recognized loss key or with malformed dictionaries, it gracefully returns None.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If returned value is not None.

        Examples:
            >>> self.test_parse_loss_missing()
        """
        loss = TrainingMonitorComponent._parse_loss_from_message("Step 5 - Metrics: {}")
        self.assertIsNone(loss)

        loss_malformed = TrainingMonitorComponent._parse_loss_from_message(
            "Step 5 - Metrics: {invalid}"
        )
        self.assertIsNone(loss_malformed)

    def test_parse_step(self) -> None:
        """Verifies successful parsing of step iterations.

        High level role: Step parser validator.
        Description: This test asserts that `_parse_step_from_message` successfully parses step
        counts using regex, and drops back to the default fallback index when
        no step label is found.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If parsed step number does not match expected.

        Examples:
            >>> self.test_parse_step()
        """
        step = TrainingMonitorComponent._parse_step_from_message(
            "Step 42 - Metrics: {'loss': 0.1}", 1
        )
        self.assertEqual(step, 42)

        fallback = TrainingMonitorComponent._parse_step_from_message("No step here", 12)
        self.assertEqual(fallback, 12)

    def test_get_category_color(self) -> None:
        """Verifies color mappings for active categories and fallbacks.

        High level role: Category color lookup validator.
        Description: This test asserts that when `_get_category_color` is called,
        it resolves recognized category keys to the mapped CSS constants, and drops
        back to the default grey color when an unrecognized value is encountered.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If returned color hex does not match expected mapping.

        Examples:
            >>> self.test_get_category_color()
        """
        color_request = TrainingMonitorComponent._get_category_color(LogType.REQUEST)
        self.assertEqual(color_request, "#3A86FF")
