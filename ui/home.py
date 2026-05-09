"""
Sports Betting — Streamlit home page.

Run with:  streamlit run ui/home.py
"""

import streamlit as st

from ui.views import predictions as predictions_view
from ui.views import chat as chat_view

st.set_page_config(
    page_title="Sports Betting",
    page_icon="⚾",
    layout="wide",
)

with st.sidebar:
    page = st.radio("Navigation", ["Predictions", "Chat"], label_visibility="collapsed")

st.title("Sports Betting Dashboard")

if page == "Predictions":
    predictions_view.render()
else:
    chat_view.render()
