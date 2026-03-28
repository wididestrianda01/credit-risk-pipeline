"""
streamlit_app.py
----------------
Interactive SHAP dashboard for exploring model predictions and feature importance.

Run locally
-----------
    streamlit run app/streamlit_app.py

Sections
--------
1. Dataset overview and score distribution
2. Global feature importance (beeswarm / bar)
3. Individual prediction explorer (waterfall plot)
4. Fairness metrics by demographic group
"""

import streamlit as st

st.set_page_config(page_title="Credit Risk Explorer", layout="wide")

st.title("Credit Risk — SHAP Dashboard")
st.info("Dashboard under construction. Load a model and dataset to begin.")

# TODO: sidebar — model / dataset upload
# TODO: section 1 — score distribution
# TODO: section 2 — global SHAP importance
# TODO: section 3 — individual waterfall
# TODO: section 4 — fairness metrics
