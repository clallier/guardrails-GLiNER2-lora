"""Unit tests for EvaluatorComponent safety rendering and model selection."""

import unittest

from src.streamlit_app.components.evaluator import EvaluatorComponent


class TestEvaluatorComponent(unittest.TestCase):
    """Verifies that EvaluatorComponent generates correct HTML and selectors.

    High level role: Unit test suite for Evaluator UI logic. Verifies pure functions,
    HTML dashboard card formats, and fallbacks.
    """

    def test_get_verdict_html_safe(self) -> None:
        """Verifies HTML output for safe verdicts.

        High level role: Safe verdict card markup validator.
        Description: This test asserts that when `_get_verdict_html` is called with 'safe',
        it returns the correct glassmorphic class name and safety labels.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If output HTML does not contain expected patterns.

        Examples:
            >>> self.test_get_verdict_html_safe()
        """
        html = EvaluatorComponent._get_verdict_html("safe", 0.0345, "success")
        self.assertIn('class="verdict-card verdict-safe"', html)
        self.assertIn("✅ Verdict: SAFE", html)
        self.assertIn("0.0345", html)

    def test_get_verdict_html_unsafe(self) -> None:
        """Verifies HTML output for unsafe verdicts.

        High level role: Unsafe verdict card markup validator.
        Description: This test asserts that when `_get_verdict_html` is called with 'unsafe',
        it returns the correct glassmorphic class name and jailbreak labels.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If output HTML does not contain expected patterns.

        Examples:
            >>> self.test_get_verdict_html_unsafe()
        """
        html = EvaluatorComponent._get_verdict_html("unsafe", 0.9850, "mock_mode")
        self.assertIn('class="verdict-card verdict-unsafe"', html)
        self.assertIn("⚠️ Verdict: UNSAFE", html)
        self.assertIn("0.9850", html)

    def test_get_safe_html(self) -> None:
        """Verifies pure HTML generation for safe cards directly.

        High level role: Safe HTML snippet function validator.
        Description: This test asserts that when `_get_safe_html` is called,
        it produces the correct glassmorphic markup, classification score representation,
        and inference mode info.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If output HTML does not contain safe verdict templates.

        Examples:
            >>> self.test_get_safe_html()
        """
        html = EvaluatorComponent._get_safe_html(0.0123, "success")
        self.assertIn('class="verdict-card verdict-safe"', html)
        self.assertIn("✅ Verdict: SAFE", html)
        self.assertIn("0.0123", html)

    def test_get_unsafe_html(self) -> None:
        """Verifies pure HTML generation for unsafe cards directly.

        High level role: Unsafe HTML snippet function validator.
        Description: This test asserts that when `_get_unsafe_html` is called,
        it produces the correct glassmorphic markup, classification score representation,
        and inference mode info.

        Args:
            None

        Returns:
            None

        Potential errors:
            AssertionError: If output HTML does not contain unsafe verdict templates.

        Examples:
            >>> self.test_get_unsafe_html()
        """
        html = EvaluatorComponent._get_unsafe_html(0.9912, "active_mode")
        self.assertIn('class="verdict-card verdict-unsafe"', html)
        self.assertIn("⚠️ Verdict: UNSAFE", html)
        self.assertIn("0.9912", html)
