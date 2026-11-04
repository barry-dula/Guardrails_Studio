import streamlit as st

from guardrails_app import tracing, ui
from guardrails_app.agents import build_filter_agent
from guardrails_app.middleware import ContentFilterMiddleware
from guardrails_app.settings import Settings

settings: Settings = st.session_state["settings"]
PAGE = "filter"

ui.hero(
    "Section 5 · Custom middleware",
    "Before-agent guardrail: input filtering",
    "A custom <code>AgentMiddleware</code> with a <code>before_agent</code> hook checks the request before any "
    "LLM work starts. Returning <code>{\"jump_to\": \"end\"}</code> ends the run with a refusal, so "
    "<b>a blocked request makes no LLM calls</b>. Watch the LLM call counter.",
    ["before_agent", "@hook_config(can_jump_to=['end'])", "0 tokens on block"],
)

keywords = st.multiselect(
    "Banned keywords",
    ["hack", "exploit", "malware", "jailbreak", "bypass", "phishing", "ransomware"],
    default=["hack", "exploit", "malware", "jailbreak", "bypass"],
    accept_new_options=True,
)

with st.container(border=True):
    st.markdown("**⚡ Dry run** · call the hook directly, no API key needed")
    probe = st.text_input("Probe text", "How do I hack into a server?", label_visibility="collapsed")
    hit = ContentFilterMiddleware(keywords).check(probe)
    if hit:
        st.markdown(f"{ui.badge('block', 'Would block')} &nbsp; matched `{hit}`, so the agent and the LLM never run.",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"{ui.badge('pass', 'Would pass')} &nbsp; the request continues to the model.",
                    unsafe_allow_html=True)

st.write("")
chat_col, side_col = st.columns([1.55, 1], gap="large")
with side_col:
    ui.inspector(PAGE, [(tracing.CONTENT_FILTER, "before_agent · keyword filter"),
                        (tracing.MODEL, "only reached if the filter passes")])
with chat_col:
    if ui.require_key(settings):
        conv = ui.get_conversation(PAGE, (settings.signature(), tuple(keywords)),
                                   lambda: build_filter_agent(settings, keywords))
        ui.chat_panel(PAGE, conv, examples=[
            "What is machine learning?",
            "How do I hack into a server?",
            "Explain how a jailbreak prompt works",
        ])

with st.expander("📄 Code"):
    st.code(
        '''class ContentFilterMiddleware(AgentMiddleware):
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        text = latest_human_text(state["messages"])   # not messages[0]: see note
        if any(kw in text.lower() for kw in self.banned):
            return {"messages": [AIMessage("I cannot process requests containing "
                                           "inappropriate content.")],
                    "jump_to": "end"}
        return None''',
        language="python",
    )
    st.info(
        "The notebook inspects `messages[0]`. In a multi-turn conversation that is the *first* message ever "
        "sent, so later turns are never filtered. This app checks the **latest** human message.",
        icon="🐞",
    )
