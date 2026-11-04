import streamlit as st

from guardrails_app import ui

ui.hero(
    "Section 1 · Foundations",
    "What are guardrails, and why do they matter?",
    "Guardrails validate and filter content at key points in an agent's run. In LangChain they are "
    "<b>middleware</b>: small hooks that run before the agent starts, around every model and tool call, "
    "and after the agent finishes.",
    ["create_agent(middleware=[…])", "before_agent · after_agent", "before_model · after_model", "wrap_tool_call"],
)

ui.kicker("Where guardrails hook into an agent run")
left, right = st.columns([1.05, 1], gap="large")
with left:
    ui.pipeline(
        [
            ("Input guardrails", "before_agent — once per invocation"),
            ("Pre-model checks", "before_model — e.g. PII redaction"),
            ("Model call", "the LLM decides: answer or call a tool"),
            ("Tool gate", "after_model / wrap_tool_call — e.g. human approval"),
            ("Output guardrails", "after_agent — final answer validation"),
        ],
        [],
    )
with right:
    st.markdown(
        """
**A guardrail can do one of four things to a request:**

- **Pass** it on unchanged.
- **Modify** it, e.g. redact an email address before the model sees it.
- **Block** it by jumping straight to the end with a refusal (`jump_to: "end"`).
- **Pause** it with an `interrupt()` until a human approves, edits or rejects the action.

Because the hooks wrap the agent loop, the agent code and the tools stay unchanged.
You add safety by appending to the `middleware=[…]` list.
"""
    )
    st.code(
        """agent = create_agent(
    model="gpt-4o",
    tools=[search, send_email],
    middleware=[
        ContentFilterMiddleware(["hack", "exploit"]),  # input
        PIIMiddleware("email", strategy="redact"),      # privacy
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        SafetyGuardrailMiddleware(),                    # output
    ],
)""",
        language="python",
    )

st.write("")
ui.kicker("Common use cases")
cases = [
    ("🔒", "PII leakage prevention", "Redact emails or mask credit cards before they reach the model or your logs."),
    ("🧨", "Prompt-injection blocking", "Detect adversarial inputs before they can change the agent's behaviour."),
    ("🚫", "Harmful content filtering", "Refuse dangerous requests early, without paying for an LLM call."),
    ("🧾", "Business-rule enforcement", "Require a human sign-off for financial, destructive or external actions."),
    ("✅", "Output quality validation", "Check final answers for safety, compliance and required disclaimers."),
]
cols = st.columns(len(cases))
for col, (icon, title, body) in zip(cols, cases):
    with col:
        ui.card(icon, title, body)

st.write("")
ui.kicker("Explore the course")
links = [
    ("views/s2_approaches.py", "2 · Deterministic vs model-based", ":material/compare_arrows:"),
    ("views/s3_pii.py", "3 · PII detection middleware", ":material/fingerprint:"),
    ("views/s4_hitl.py", "4 · Human-in-the-loop middleware", ":material/how_to_reg:"),
    ("views/s5_input_filter.py", "5 · Custom before_agent input filter", ":material/filter_alt:"),
    ("views/s6_output_safety.py", "6 · Custom after_agent output safety", ":material/verified_user:"),
    ("views/s7_layered.py", "7 · Layered / combined guardrails", ":material/layers:"),
    ("views/s8_healthcare.py", "8 · Real-world: healthcare chatbot", ":material/medical_services:"),
]
grid = st.columns(2)
for i, (path, label, icon) in enumerate(links):
    with grid[i % 2]:
        st.page_link(path, label=label, icon=icon, width="stretch")
