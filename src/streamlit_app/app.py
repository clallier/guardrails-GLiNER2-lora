"""Main Streamlit entrypoint bootstrapping the finetuner user interface."""

import streamlit as st

from src.streamlit_app.api.client import FinetunerClient
from src.streamlit_app.components.config_panel import ConfigPanelComponent
from src.streamlit_app.components.evaluator import EvaluatorComponent
from src.streamlit_app.components.training_monitor import TrainingMonitorComponent
from src.streamlit_app.styles.loader import load_css


def init_app_state() -> FinetunerClient:
    """Bootstraps application theme and returns the persistent backend client.

    Returns:
        FinetunerClient: API Client instance.
    """
    st.set_page_config(
        page_title="GLiGuard-MLX Safety Finetuner",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ingest custom styling
    load_css()

    # Persist client in session state
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = FinetunerClient()

    return st.session_state["api_client"]


def main() -> None:
    """Draws the primary layout and routes active tabs."""
    client = init_app_state()

    st.title("🔥 GLiGuard-MLX Model Finetuning Dashboard")
    st.caption("A modular interface to finetune safety guardrails using GLiNER2")

    # Configure tabs
    tab_train, tab_eval = st.tabs(["🔥 Finetuning Dashboard", "🧪 Interactive Safety Tester"])

    with tab_train:
        col_conf, col_mon = st.columns([1, 2])
        with col_conf:
            ConfigPanelComponent.render(client)
        with col_mon:
            TrainingMonitorComponent.render(client)

    with tab_eval:
        EvaluatorComponent.render(client)


if __name__ == "__main__":
    main()
