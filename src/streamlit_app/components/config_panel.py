"""Streamlit component rendering the hyperparameter config controls."""

import streamlit as st

from src.finetuner.config import GlobalConfig
from src.streamlit_app.api.client import FinetunerClient


class ConfigPanelComponent:
    """Renders the settings and launch dashboard for training runs.

    Encapsulates all form-inputs and hyperparameter state transitions, delegating
    thread launches to the API client.
    """

    @staticmethod
    def render(client: FinetunerClient) -> None:
        """Draws the hyperparameter settings form on the active page or sidebar.

        Args:
            client (FinetunerClient): API client wrapping training logic.
        """
        with st.container(border=True):
            st.subheader("⚙️ Finetuning Configurations")

            inputs = ConfigPanelComponent._render_inputs()
            (
                model_id,
                lr,
                batch_size,
                epochs,
                test_split,
                use_wandb,
            ) = inputs

            config = GlobalConfig(
                model_id=model_id,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=epochs,
                test_split=test_split,
                use_wandb=use_wandb,
            )

            ConfigPanelComponent._render_launch_button(client, config)

    @staticmethod
    def _render_inputs() -> tuple:
        """Renders input form elements and returns configured values.

        Returns:
            Tuple: Collected input states.
        """
        default_config = GlobalConfig()
        model_id = st.text_input("Model ID", value=default_config.model_id)

        col1, col2 = st.columns(2)
        with col1:
            lr, batch_size = ConfigPanelComponent._render_col1(default_config)
        with col2:
            epochs, test_split = ConfigPanelComponent._render_col2(default_config)

        use_wandb = st.toggle("Track with Weights & Biases (WandB)", value=default_config.use_wandb)
        return (
            model_id,
            lr,
            batch_size,
            epochs,
            test_split,
            use_wandb,
        )

    @staticmethod
    def _render_col1(default_config: GlobalConfig) -> tuple[float, int]:
        """Renders hyperparameters in the left column.

        Args:
            default_config (GlobalConfig): default hyperparameter config.

        Returns:
            tuple[float, int]: lr, batch_size values.
        """
        lr = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=default_config.learning_rate,
            format="%.6f",
        )
        batch_size = st.slider(
            "Batch Size", min_value=1, max_value=64, value=default_config.batch_size
        )
        return lr, batch_size

    @staticmethod
    def _render_col2(default_config: GlobalConfig) -> tuple[int, float]:
        """Renders hyperparameters in the right column.

        Args:
            default_config (GlobalConfig): default hyperparameter config.

        Returns:
            tuple[int, float]: epochs, test_split values.
        """
        epochs = st.slider("Epochs", min_value=1, max_value=10, value=default_config.epochs)
        test_split = st.slider(
            "Train/Test Split", min_value=0.05, max_value=0.5, value=default_config.test_split
        )
        return epochs, test_split

    @staticmethod
    def _render_launch_button(client: FinetunerClient, config: GlobalConfig) -> None:
        """Decoupled helper to draw the launch button state.

        Args:
            client (FinetunerClient): API client.
            config (GlobalConfig): Target run configuration.
        """
        if client.is_training_active():
            st.button("🔥 Finetuning in Progress...", disabled=True)
        else:
            if st.button("🚀 Start Finetuning Run", use_container_width=True):
                client.start_training(config)
                st.rerun()
