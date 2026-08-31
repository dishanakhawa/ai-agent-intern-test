"""
app.py

Minimal web UI for the Aster & Row support agent. Wraps agent.run_turn()
in a Streamlit chat interface. No polish required by the assignment --
this exists so the demo video has something to click through instead
of a bare terminal.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agent import run_turn

st.set_page_config(page_title="Aster & Row Support", page_icon="🏔️")
st.title("🏔️ Aster & Row Support")
st.caption("Ask about returns, shipping, warranty, or check an order status.")

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_turn(st.session_state.session_id, user_input)
        st.markdown(result["answer"])

        with st.expander("Debug: trace"):
            st.json(result["trace"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})