import streamlit as st

from guardrails_app import tracing, ui
from guardrails_app.agents import build_layered_agent
from guardrails_app.settings import Settings

settings: Settings = st.session_state["settings"]
PAGE = "layered"
KEYWORDS = ["hack", "exploit", "malware"]
LAYERS = [
    (tracing.CONTENT_FILTER, "L1 · before_agent · deterministic"),
    (tracing.PII_INPUT, "L2 · before_model · mask credit cards"),
    (tracing.MODEL, "agent model + tools"),
    (tracing.HITL, "L3 · approval for send_email"),
    (tracing.PII_OUTPUT, "L4 · after_model · redact emails"),
    (tracing.SAFETY_JUDGE, "L5 · after_agent · LLM judge"),
]

ui.hero(
    "Section 7 · Production",
    "Layered / combined guardrails",
    "Defense in depth: stack guardrails in <code>middleware=[…]</code> and they run in order. Cheap deterministic "
    "checks run first, then privacy layers, then human approval, then the model-based output review. "
    "Send a message and watch which layers fire.",
    ["5 layers", "deterministic first", "model-based last"],
)

chat_col, side_col = st.columns([1.45, 1], gap="large")
with side_col:
    ui.inspector(PAGE, LAYERS)
with chat_col:
    if ui.require_key(settings):
        conv = ui.get_conversation(PAGE, settings.signature(), lambda: build_layered_agent(settings, KEYWORDS))
        ui.chat_panel(
            PAGE, conv, height=560,
            examples=[
                "How do I exploit a buffer overflow?",
                "My card is 4111 1111 1111 1111 — search for my last order",
                "Email ops@company.com saying the deploy is done",
                "What is defense in depth in security?",
            ],
        )

with st.expander("📄 Code — the full 5-layer stack"):
    st.code(
        """production_agent = create_agent(
    model="gpt-4o",
    tools=[search_web, send_email],
    middleware=[
        ContentFilterMiddleware(["hack", "exploit", "malware"]),             # L1 input, deterministic
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),  # L2 input privacy
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),         # L3 approval
        PIIMiddleware("email", strategy="redact",                         # L4 output privacy
                      apply_to_input=False, apply_to_output=True),
        SafetyGuardrailMiddleware(judge_model),                              # L5 output, model-based
    ],
    checkpointer=InMemorySaver(),
)""",
        language="python",
    )
