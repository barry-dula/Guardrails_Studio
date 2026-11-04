import re

import pandas as pd
import streamlit as st
from langchain.agents.middleware._redaction import RedactionRule

from guardrails_app import tracing, ui
from guardrails_app.agents import API_KEY_REGEX, build_pii_agent
from guardrails_app.settings import Settings

settings: Settings = st.session_state["settings"]
PAGE = "pii"
STRATEGIES = ["redact", "mask", "hash", "block"]
STRATEGY_HELP = {
    "redact": "`[REDACTED_EMAIL]`",
    "mask": "`****-****-****-5100`",
    "hash": "`<email_hash:a8f5f167>`",
    "block": "raises `PIIDetectionError`",
}

ui.hero(
    "Section 3 · Built-in middleware",
    "PII detection middleware",
    "<code>PIIMiddleware</code> finds personally identifiable information with built-in or custom detectors and "
    "applies a strategy before the model sees it: <b>redact</b>, <b>mask</b>, <b>hash</b> or <b>block</b>.",
    ["email", "credit_card", "ip", "mac_address", "url", "custom regex"],
)

tab_play, tab_agent, tab_code = st.tabs(["🧪 Redaction playground", "🤖 Agent demo", "📄 Code"])

# --------------------------------------------------------------------------- #
with tab_play:
    st.caption("This runs the same redaction engine `PIIMiddleware` uses, locally, without an API key.")
    sample = (
        "Hi, I'm Jane. Email me at jane.doe@example.com or jdoe@work.io.\n"
        "Card: 5105-1051-0510-5100 (exp 04/28). Server 192.168.1.1, MAC 00:1A:2B:3C:4D:5E.\n"
        "Docs: https://secret-site.com/report  ·  key: sk-abcdefghijklmnopqrstuvwxyz123456"
    )
    text = st.text_area("Text to scan", sample, height=120)

    types = ["email", "credit_card", "ip", "mac_address", "url", "api_key (custom regex)"]
    cfg_cols = st.columns(3)
    rules: list[RedactionRule] = []
    for i, ptype in enumerate(types):
        with cfg_cols[i % 3], st.container(border=True):
            on = st.toggle(f"`{ptype}`", value=True, key=f"pt_{ptype}")
            strat = st.segmented_control(
                "Strategy", STRATEGIES[:3] if ptype != "api_key (custom regex)" else STRATEGIES,
                default="mask" if ptype == "credit_card" else "redact",
                key=f"ps_{ptype}", label_visibility="collapsed", disabled=not on,
            ) or "redact"
            if on:
                if ptype.startswith("api_key"):
                    pattern = st.text_input("Detector regex", API_KEY_REGEX, key="api_regex")
                    rules.append(RedactionRule("api_key", strat, pattern))
                else:
                    rules.append(RedactionRule(ptype, strat))

    out, all_matches, blocked = text, [], None
    for rule in rules:
        try:
            resolved = rule.resolve()
        except (ValueError, re.error) as exc:
            st.error(f"Invalid detector for {rule.pii_type}: {exc}")
            continue
        matches = resolved.detector(out)
        if matches and rule.strategy == "block":
            blocked = f"Detected {len(matches)} instance(s) of {rule.pii_type} in text content"
        all_matches += [{"type": m["type"], "value": m["value"], "strategy": rule.strategy} for m in matches]
        if rule.strategy != "block":
            out, _ = resolved.apply(out)

    # highlight detections in the original
    highlighted = ui.esc(text)
    for m in sorted({m["value"] for m in all_matches}, key=len, reverse=True):
        highlighted = highlighted.replace(ui.esc(m), f'<mark class="gr-mark">{ui.esc(m)}</mark>')

    a, b = st.columns(2, gap="large")
    with a:
        ui.kicker("Detected in input")
        st.markdown(f'<div class="gr-mono">{highlighted}</div>', unsafe_allow_html=True)
    with b:
        ui.kicker("What the model receives")
        if blocked:
            st.markdown(f'<div class="gr-mono">{ui.badge("block", "Request blocked")}<br><br>'
                        f'PIIDetectionError: {ui.esc(blocked)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="gr-mono">{ui.esc(out)}</div>', unsafe_allow_html=True)
    if all_matches:
        st.dataframe(pd.DataFrame(all_matches), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
with tab_agent:
    chat_col, side_col = st.columns([1.55, 1], gap="large")
    with side_col:
        with st.container(border=True):
            st.markdown("**Middleware configuration**")
            email = st.selectbox("email strategy", STRATEGIES[:3], index=0)
            card = st.selectbox("credit_card strategy", STRATEGIES[:3], index=1)
            block_keys = st.toggle("Block API keys (`sk-…`)", value=True)
            st.caption("Changing these rebuilds the agent and starts a new chat. "
                       + " · ".join(f"**{k}** → {v}" for k, v in STRATEGY_HELP.items()))
        ui.inspector(PAGE, [(tracing.PII_INPUT, "before_model · email, credit_card, api_key"),
                            (tracing.MODEL, "gpt agent + customer_lookup tool")])
    with chat_col:
        if ui.require_key(settings):
            conv = ui.get_conversation(
                PAGE, (settings.signature(), email, card, block_keys),
                lambda: build_pii_agent(settings, email, card, block_keys),
            )
            ui.chat_panel(
                PAGE, conv,
                examples=[
                    "My email is john.doe@example.com and my card is 5105-1051-0510-5100. Can you look me up?",
                    "Here is my key: sk-abcdefghijklmnopqrstuvwxyz123456",
                    "What did I tell you my email address was?",
                ],
            )

with tab_code:
    st.code(
        f'''agent = create_agent(
    model="gpt-4o",
    tools=[customer_lookup],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"{API_KEY_REGEX}",
                      strategy="block", apply_to_input=True),   # raises PIIDetectionError
    ],
)''',
        language="python",
    )
    st.markdown(
        "- `apply_to_input` scans the latest user message before each model call.\n"
        "- `apply_to_output` scans AI messages, and `apply_to_tool_results` scans tool output.\n"
        "- `detector` takes a regex string or a function `str -> list[PIIMatch]` for custom types."
    )
