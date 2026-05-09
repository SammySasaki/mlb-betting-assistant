"""
Chat view — wraps the MLB chatbot LangGraph via the /chat API endpoint.
"""

import streamlit as st

from ui.api.client import send_chat_message


def render() -> None:
    st.header("MLB Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict] = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about stats, lineups, or betting recommendations…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = send_chat_message(user_input)
                except Exception as exc:
                    if "ConnectionError" in type(exc).__name__:
                        reply = "Cannot reach the API. Make sure the FastAPI server is running."
                    else:
                        reply = f"Error: {exc}"
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
