import streamlit as st

from guardrails_app import tracing, ui
from guardrails_app.agents import build_hitl_agent
from guardrails_app.settings import Settings
from guardrails_app.tools import HITL_TOOLS

settings: Settings = st.session_state["settings"]
PAGE = "hitl"

ui.hero(
    "Section 4 · Built-in middleware",
    "Human-in-the-loop middleware",
    "<code>HumanInTheLoopMiddleware</code> pauses the agent before sensitive tool calls with a LangGraph "
    "<code>interrupt</code>. A reviewer can <b>approve</b>, <b>edit</b> or <b>reject</b> the call, and the run resumes "
    "on the same <code>thread_id</code> with <code>Command(resume=…)</code>.",
    ["interrupt_on={…}", "checkpointer required", "approve · edit · reject"],
)

chat_col, side_col = st.columns([1.55, 1], gap="large")
with side_col:
    with st.container(border=True):
        st.markdown("**Approval policy** · `interrupt_on`")
        gated = []
        defaults = {"send_email": True, "delete_records": True, "search_web": False}
        for name, tool in HITL_TOOLS.items():
            if st.toggle(f"`{name}`", value=defaults[name], help=tool.description, key=f"gate_{name}"):
                gated.append(name)
        st.caption("Tools that are switched on need approval; the others run automatically.")
    ui.inspector(PAGE, [(tracing.MODEL, "agent decides which tool to call"),
                        (tracing.HITL, "after_model · interrupt before gated tools")])

with chat_col:
    if ui.require_key(settings):
        conv = ui.get_conversation(PAGE, (settings.signature(), tuple(gated)), lambda: build_hitl_agent(settings, gated))
        ui.chat_panel(
            PAGE, conv,
            examples=[
                "Send an email to team@company.com about the Q4 results",
                "Delete all records from the users table where active=false",
                "Search the web for the latest LangChain release",
            ],
        )

with st.expander("📄 Code"):
    st.code(
        """hitl_agent = create_agent(
    model="gpt-4o",
    tools=[search_web, send_email, delete_records],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "send_email": True, "delete_records": True, "search_web": False,
    })],
    checkpointer=InMemorySaver(),          # state must survive the pause
)
config = {"configurable": {"thread_id": "session_001"}}
result = hitl_agent.invoke({"messages": [...]}, config)    # -> result["__interrupt__"]

hitl_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
hitl_agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Too risky"}]}), config)
hitl_agent.invoke(Command(resume={"decisions": [{"type": "edit",
    "edited_action": {"name": "send_email", "args": {...}}}]}), config)""",
        language="python",
    )
    st.caption("Note: in current LangChain the rejection field is `message` (the notebook used `reason`).")
