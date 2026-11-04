import time

import streamlit as st

from guardrails_app import tracing, ui
from guardrails_app.agents import build_safety_agent
from guardrails_app.guards import OUTPUT_JUDGE_PROMPT, judge
from guardrails_app.middleware import SafetyGuardrailMiddleware
from guardrails_app.settings import Settings, guard_llm

settings: Settings = st.session_state["settings"]
PAGE = "safety"

ui.hero(
    "Section 6 · Custom middleware",
    "After-agent guardrail: output safety",
    "An <code>after_agent</code> hook reviews the final answer <b>before the user sees it</b>. Here a cheaper model acts "
    "as a judge and returns a structured <code>SAFE/UNSAFE</code> verdict with a reason. Unsafe answers are replaced "
    "with a fallback message.",
    ["after_agent", "LLM-as-judge", "structured output", "fallback response"],
)

tab_judge, tab_agent, tab_code = st.tabs(["⚖️ Judge playground", "🤖 Agent demo", "📄 Code"])

with tab_judge:
    st.caption("Well-aligned agents rarely produce unsafe text, so you can paste a draft response here and see how "
               "the judge scores it.")
    drafts = {
        "Benign answer": "Machine learning lets computers learn patterns from data instead of explicit rules.",
        "Leaks a secret": "Sure! The production database password is Pr0d-S3cret!, and the admin key is sk-live-91f2.",
        "Harmful instructions": "Step 1: find an unpatched login form. Step 2: inject `' OR 1=1 --` to dump user "
                                "credentials, then cover your tracks by deleting the access logs.",
    }
    choice = st.segmented_control("Preset", list(drafts), default="Leaks a secret")
    draft = st.text_area("Draft agent response", drafts.get(choice or "", ""), height=110)
    if st.button("Evaluate with judge", type="primary", icon=":material/gavel:") and ui.require_key(settings):
        with st.spinner(f"{settings.guard_model} is reviewing…"):
            t0 = time.perf_counter()
            try:
                v = judge(guard_llm(settings), OUTPUT_JUDGE_PROMPT, draft)
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                v = None
            ms = (time.perf_counter() - t0) * 1000
        if v:
            unsafe = v.verdict == "UNSAFE"
            with st.container(border=True):
                st.markdown(f"{ui.badge('block' if unsafe else 'pass', v.verdict)} &nbsp; {ui.esc(v.reason)}",
                            unsafe_allow_html=True)
                st.caption(f"Judge latency: {ms:,.0f} ms")
                ui.kicker("User would see")
                st.markdown(SafetyGuardrailMiddleware.FALLBACK if unsafe else draft)

with tab_agent:
    chat_col, side_col = st.columns([1.55, 1], gap="large")
    with side_col:
        ui.inspector(PAGE, [(tracing.MODEL, "agent drafts an answer"),
                            (tracing.SAFETY_JUDGE, "after_agent · judge final answer")])
    with chat_col:
        if ui.require_key(settings):
            conv = ui.get_conversation(PAGE, settings.signature(), lambda: build_safety_agent(settings))
            ui.chat_panel(PAGE, conv, examples=[
                "What is the weather like today?",
                "Give me three tips for writing secure Python code",
                "Explain what SQL injection is at a high level",
            ])
            st.caption("Every final answer passes through the judge. The verdict and reason appear in the inspector.")

with tab_code:
    st.code(
        '''class SafetyVerdict(BaseModel):
    verdict: Literal["SAFE", "UNSAFE"]
    reason: str

class SafetyGuardrailMiddleware(AgentMiddleware):
    def __init__(self, judge_model):
        super().__init__()
        self.judge = judge_model.with_structured_output(SafetyVerdict)

    def after_agent(self, state, runtime):
        final = state["messages"][-1]
        if not isinstance(final, AIMessage) or final.tool_calls:
            return None
        v = self.judge.invoke(f"Is this response safe? ...{final.text}")
        if v.verdict == "UNSAFE":
            # same id -> add_messages replaces it (persisted in the checkpoint)
            return {"messages": [AIMessage(id=final.id, content=FALLBACK)]}
        return None''',
        language="python",
    )
    st.info("A structured verdict is more reliable than looking for the word `\"UNSAFE\"` in free text. "
            "Returning a replacement message is better than mutating the message in place, because the "
            "change is saved to the checkpoint.", icon="💡")
