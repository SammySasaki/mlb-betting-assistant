"""
Sports Betting — Streamlit home page.

Run with:  streamlit run ui/home.py
"""

import streamlit as st

from ui.views import predictions as predictions_view

st.set_page_config(
    page_title="Sports Betting",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ Sports Betting Dashboard")

predictions_view.render()
