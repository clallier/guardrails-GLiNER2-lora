"""Module to dynamically load custom CSS styling into Streamlit components."""

import os

import streamlit as st


def load_css() -> None:
    """Reads the custom main.css stylesheet and injects it into the Streamlit app.

    High level role: Application theme customizer. Injects CSS rules declared
    in main.css directly into the active Streamlit app view

    Args:
        None

    Returns:
        None: Injects the styling directly via Streamlit's st.markdown block.

    Potential errors:
        FileNotFoundError: Raised if the target main.css stylesheet cannot be located.

    Examples:
        >>> load_css()
    """
    css_path = os.path.join(os.path.dirname(__file__), "main.css")
    if not os.path.exists(css_path):
        raise FileNotFoundError(f"Custom stylesheet not found at: {css_path}")

    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
