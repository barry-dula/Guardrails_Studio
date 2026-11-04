"""Guardrails Studio — an interactive Streamlit tour of LangChain guardrails.

Run with:  uv run streamlit run app.py
"""

import streamlit as st

from guardrails_app import ui

st.set_page_config(page_title="Guardrails Studio", page_icon="🛡️", layout="wide")
ui.inject_css()

pages = {
    "Foundations": [
        st.Page("views/s1_overview.py", title="1 · What are guardrails?", icon=":material/shield:", default=True),
        st.Page("views/s2_approaches.py", title="2 · Deterministic vs model", icon=":material/compare_arrows:"),
    ],
    "Built-in middleware": [
        st.Page("views/s3_pii.py", title="3 · PII detection", icon=":material/fingerprint:"),
        st.Page("views/s4_hitl.py", title="4 · Human-in-the-loop", icon=":material/how_to_reg:"),
    ],
    "Custom middleware": [
        st.Page("views/s5_input_filter.py", title="5 · Input filter", icon=":material/filter_alt:"),
        st.Page("views/s6_output_safety.py", title="6 · Output safety", icon=":material/verified_user:"),
    ],
    "Production": [
        st.Page("views/s7_layered.py", title="7 · Layered guardrails", icon=":material/layers:"),
        st.Page("views/s8_healthcare.py", title="8 · Healthcare chatbot", icon=":material/medical_services:"),
    ],
}

nav = st.navigation(pages)
st.session_state["settings"] = ui.sidebar_settings()
nav.run()
