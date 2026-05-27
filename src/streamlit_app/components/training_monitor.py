import ast
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.finetuner.constants import LOG_CATEGORY_COLORS, LOG_DEFAULT_COLOR, LogType
from src.streamlit_app.api.client import FinetunerClient


class TrainingMonitorComponent:
    """Renders active training indicators, metrics, and progress logs.

    Handles real-time console rendering using JetBrains Mono and dynamically
    plots training performance metrics (e.g., loss curves).
    """

    @staticmethod
    def render(client: FinetunerClient) -> None:
        """Renders the metrics and log monitor panels on the current page.

        Args:
            client (FinetunerClient): API client wrapping active state.
        """
        with st.container(border=True):
            st.subheader("📊 Training Monitor & Metrics")

            if not client.has_training_started():
                st.info(
                    "No active training run. "
                    "Configure settings and start a job to monitor live metrics."
                )
                return

            tracker = client.get_training_tracker()

            # Header status indicator
            TrainingMonitorComponent._render_status_header(client)

            # Draw metrics mock/real plots
            TrainingMonitorComponent._render_charts(tracker.get_logs())

            # Raw logs streaming
            st.markdown("##### 💻 Execution Logs")
            TrainingMonitorComponent._render_console_logs(tracker.get_logs())

    @staticmethod
    def _render_status_header(client: FinetunerClient) -> None:
        """Helper to draw the status indicator header.

        Args:
            client (FinetunerClient): Running API client.
        """
        if client.is_training_active():
            st.info(
                "🔄 **Finetuning Job Active** "
                "(Compiling datasets, loading weights, running loops)..."
            )
        else:
            st.success("✅ Job Completed/Idle")

    @staticmethod
    def _parse_loss_from_message(message: str) -> float | None:
        """Parses a metric log message to extract training/evaluation loss values.

        High level role: Pure parser helper. Extracts loss scalar float values from
        structured METRIC log records using safe AST evaluations.

        Args:
            message (str): Log message string containing the structured metrics.

        Returns:
            float | None: Parsed loss value if successful, or None if unavailable/malformed.

        Potential errors:
            None: Safely catches syntax/value errors during parsing.

        Examples:
            >>> TrainingMonitorComponent._parse_loss_from_message(
            ...     "Step 5 - Metrics: {'loss': 0.1234}"
            ... )
            0.1234
        """
        match = re.search(r"Metrics:\s*(\{.*\})", message)
        if not match:
            return None

        try:
            metrics = ast.literal_eval(match.group(1))
            if isinstance(metrics, dict):
                for key in ["loss", "train_loss", "eval_loss", "epoch_loss"]:
                    if key in metrics:
                        return float(metrics[key])
        except (ValueError, SyntaxError, TypeError):
            pass

        return None

    @staticmethod
    def _parse_step_from_message(message: str, default_step: int) -> int:
        """Parses a metric log message to extract the step/iteration number.

        High level role: Pure parser helper. Extracts step iteration integers from
        structured METRIC log records using regular expressions.

        Args:
            message (str): Log message string containing the structured step details.
            default_step (int): Fallback step index if extraction fails.

        Returns:
            int: Resolved step iteration number.

        Potential errors:
            None.

        Examples:
            >>> TrainingMonitorComponent._parse_step_from_message(
            ...     "Step 5 - Metrics: {'loss': 0.1234}", 1
            ... )
            5
        """
        match = re.search(r"Step\s+(\d+)", message)
        return int(match.group(1)) if match else default_step

    @staticmethod
    def _render_charts(logs: List[Dict[str, Any]]) -> None:
        """Plots training performance charts from reported metrics.

        High level role: Metric chart renderer. Parses training logs for actual metric
        values, extracts training/evaluation loss, and draws a real-time line chart.

        Args:
            logs (List[Dict[str, Any]]): Log entries to parse for metrics.

        Returns:
            None: Renders directly to the Streamlit layout.

        Potential errors:
            None.

        Examples:
            >>> TrainingMonitorComponent._render_charts(
            ...     [{"category": "METRIC", "message": "Step 5 - Metrics: {'loss': 0.123}"}]
            ... )
        """
        metric_entries = [entry for entry in logs if entry["category"] == "METRIC"]
        loss_data = []

        for index, entry in enumerate(metric_entries):
            msg = entry.get("message", "")
            loss_val = TrainingMonitorComponent._parse_loss_from_message(msg)
            if loss_val is not None:
                step = TrainingMonitorComponent._parse_step_from_message(msg, index + 1)
                loss_data.append({"Iteration": step, "Loss": loss_val})

        if not loss_data:
            st.caption("Awaiting performance metrics reports from the training loops...")
            return

        df = pd.DataFrame(loss_data)
        st.line_chart(df.set_index("Iteration"))

    @staticmethod
    def _render_console_logs(logs: List[Dict[str, Any]]) -> None:
        """Renders logs in a custom scrolling terminal console.

        Args:
            logs (List[Dict[str, Any]]): Structured log history.
        """
        console_html = '<div class="log-console">'

        for log in reversed(logs):
            category = log["category"]
            color = TrainingMonitorComponent._get_category_color(category)
            msg = f"<span style='color:{color};font-weight:bold;'>[{category}]</span> "
            f"{log['message']}"
            console_html += f"<p>{msg}</p>"

        console_html += "</div>"
        st.markdown(console_html, unsafe_allow_html=True)
        if st.button("Refresh Logs", key="refresh_logs"):
            st.rerun()

    @staticmethod
    def _get_category_color(category: LogType) -> str:
        """Determines the terminal printout color for a given category using app constants.

        High level role: UI log color resolver. Maps text categories to custom terminal
        hex colors configured in global dashboard constants.

        Args:
            category (str): Log type indicator (e.g. 'REQUEST', 'RESPONSE').

        Returns:
            str: Resolved hexadecimal CSS color code.

        Potential errors:
            None: Gracefully falls back to default style colors.

        Examples:
            >>> TrainingMonitorComponent._get_category_color("REQUEST")
            '#3A86FF'
        """
        return LOG_CATEGORY_COLORS.get(category, LOG_DEFAULT_COLOR)
