"""
streamlit_app.py
----------------
Entrypoint for Streamlit Community Cloud deployment.

This file wraps the actual app at app/streamlit_app.py to ensure Community Cloud
can find the entrypoint at the repository root (default location).

Run locally:
    streamlit run streamlit_app.py

Run in cloud:
    Community Cloud automatically discovers and runs this file.
"""

# Import and execute the actual app
from app.streamlit_app import *  # noqa: F401, F403

# The imports above expose all Streamlit components defined in app/streamlit_app.py.
# Streamlit detects the page_config, widgets, and callbacks automatically.
