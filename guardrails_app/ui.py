"""Shared Streamlit UI: styling, page headers, pipeline diagram, chat + inspector."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from . import tracing
from .agents import BuiltAgent
from .conversation import Conversation, tool_messages_by_call
from .settings import DEFAULT_AGENT_MODEL, DEFAULT_GUARD_MODEL, Settings, env_api_key
from .tracing import TraceEvent

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
:root{
  --gr-surface:#131A26; --gr-surface-2:#182133; --gr-border:#232C3D; --gr-muted:#8A94A8;
  --gr-text:#E6EAF2; --gr-accent:#7C9CFF; --gr-green:#3DD68C; --gr-red:#FF6B6B;
  --gr-amber:#FFB547; --gr-blue:#5AB0FF; --gr-violet:#B48CFF;
}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1280px;}
h1,h2,h3{letter-spacing:-0.01em;}
[data-testid="stSidebarNav"] span{font-size:0.92rem;}

/* hero */
.gr-hero{position:relative; overflow:hidden; border:1px solid var(--gr-border); border-radius:18px;
  padding:26px 28px; margin-bottom:22px;
  background:radial-gradient(1200px 300px at 0% 0%, rgba(124,156,255,.18), transparent 60%),
             radial-gradient(800px 260px at 100% 0%, rgba(180,140,255,.14), transparent 60%), var(--gr-surface);}
.gr-hero .gr-eyebrow{display:inline-flex; gap:8px; align-items:center; font-size:.74rem; font-weight:600;
  letter-spacing:.08em; text-transform:uppercase; color:var(--gr-accent);}
.gr-hero h1{font-size:1.9rem; margin:.35rem 0 .35rem 0; padding:0; line-height:1.2;}
.gr-hero p{color:var(--gr-muted); margin:0; font-size:1rem; max-width:860px;}
.gr-chips{display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;}
.gr-chip{font-size:.78rem; padding:4px 10px; border-radius:999px; border:1px solid var(--gr-border);
  background:rgba(255,255,255,.03); color:var(--gr-text);}

/* cards */
.gr-card{border:1px solid var(--gr-border); background:var(--gr-surface); border-radius:14px; padding:16px 18px; height:100%;}
.gr-card h4{margin:0 0 6px 0; font-size:1rem;}
.gr-card p{margin:0; color:var(--gr-muted); font-size:.9rem; line-height:1.5;}
.gr-card .gr-ico{font-size:1.35rem; margin-bottom:6px;}
.gr-kicker{font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--gr-muted); margin:6px 0 10px;}

/* verdict badges */
.gr-badge{display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:.8rem;
  padding:3px 10px; border-radius:999px; border:1px solid transparent;}
.gr-pass{color:var(--gr-green); background:rgba(61,214,140,.1); border-color:rgba(61,214,140,.3);}
.gr-block{color:var(--gr-red); background:rgba(255,107,107,.1); border-color:rgba(255,107,107,.3);}
.gr-modify{color:var(--gr-amber); background:rgba(255,181,71,.1); border-color:rgba(255,181,71,.3);}
.gr-pause{color:var(--gr-blue); background:rgba(90,176,255,.1); border-color:rgba(90,176,255,.3);}
.gr-idle{color:var(--gr-muted); background:rgba(255,255,255,.03); border-color:var(--gr-border);}
.gr-llm{color:var(--gr-violet); background:rgba(180,140,255,.1); border-color:rgba(180,140,255,.3);}

/* pipeline */
.gr-pipe{display:flex; flex-direction:column; gap:0; margin:4px 0 6px;}
.gr-node{display:flex; align-items:center; justify-content:space-between; gap:10px; padding:9px 12px;
  border:1px solid var(--gr-border); border-radius:11px; background:var(--gr-surface-2); font-size:.86rem;}
.gr-node .gr-n-left{display:flex; align-items:center; gap:10px;}
.gr-node .gr-n-idx{width:22px; height:22px; border-radius:7px; display:grid; place-items:center; font-size:.72rem;
  font-weight:700; background:rgba(124,156,255,.15); color:var(--gr-accent);}
.gr-node .gr-n-hook{font-size:.7rem; color:var(--gr-muted); font-family:'JetBrains Mono',monospace;}
.gr-node.gr-endpoint{background:transparent; border-style:dashed; color:var(--gr-muted);}
.gr-node.s-pass{border-color:rgba(61,214,140,.45); box-shadow:inset 3px 0 0 var(--gr-green);}
.gr-node.s-block{border-color:rgba(255,107,107,.55); box-shadow:inset 3px 0 0 var(--gr-red);}
.gr-node.s-modify{border-color:rgba(255,181,71,.5); box-shadow:inset 3px 0 0 var(--gr-amber);}
.gr-node.s-pause{border-color:rgba(90,176,255,.55); box-shadow:inset 3px 0 0 var(--gr-blue);}
.gr-node.s-llm{border-color:rgba(180,140,255,.5); box-shadow:inset 3px 0 0 var(--gr-violet);}
.gr-node.s-skipped{opacity:.4;}
.gr-arrow{height:14px; margin-left:22px; border-left:2px dotted var(--gr-border);}

/* trace */
.gr-trace{display:flex; flex-direction:column; gap:8px;}
.gr-ev{display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:start; padding:9px 11px;
  border-radius:10px; border:1px solid var(--gr-border); background:var(--gr-surface-2); font-size:.84rem;}
.gr-ev .gr-ev-layer{font-weight:600;}
.gr-ev .gr-ev-detail{color:var(--gr-muted); margin-top:2px; word-break:break-word;}
.gr-empty{color:var(--gr-muted); font-size:.86rem; padding:14px; border:1px dashed var(--gr-border); border-radius:10px; text-align:center;}

/* tool call chip in chat */
.gr-tool{font-family:'JetBrains Mono',monospace; font-size:.78rem; padding:7px 10px; border-radius:9px;
  border:1px solid var(--gr-border); background:rgba(124,156,255,.07); margin:4px 0 8px; color:var(--gr-text);
  white-space:pre-wrap; word-break:break-word;}
.gr-tool b{color:var(--gr-accent);}
.gr-tool .gr-tool-res{color:var(--gr-muted); display:block; margin-top:4px;}

/* highlighted PII */
mark.gr-mark{background:rgba(255,181,71,.22); color:var(--gr-amber); padding:1px 3px; border-radius:4px;}
.gr-mono{font-family:'JetBrains Mono',monospace; font-size:.85rem; line-height:1.7; white-space:pre-wrap;
  word-break:break-word; padding:14px 16px; border-radius:12px; border:1px solid var(--gr-border); background:var(--gr-surface-2);}

.gr-approval{border:1px solid rgba(90,176,255,.45); border-radius:14px; padding:4px 2px;
  background:linear-gradient(180deg, rgba(90,176,255,.08), transparent 70%);}
</style>
"""

KIND_META: dict[str, tuple[str, str, str]] = {
    # kind -> (css class, icon, label)
    "pass": ("gr-pass", "✓", "Passed"),
    "block": ("gr-block", "⛔", "Blocked"),
    "modify": ("gr-modify", "✎", "Modified"),
    "pause": ("gr-pause", "⏸", "Paused"),
    "approve": ("gr-pass", "✓", "Approved"),
    "reject": ("gr-block", "✕", "Rejected"),
    "llm": ("gr-llm", "◆", "LLM call"),
    "error": ("gr-block", "!", "Error"),
    "info": ("gr-idle", "•", "Info"),
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(str(text))


def badge(kind: str, label: str | None = None) -> str:
    cls, icon, default = KIND_META.get(kind, KIND_META["info"])
    return f'<span class="gr-badge {cls}">{icon} {esc(label or default)}</span>'


def hero(section: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = "".join(f'<span class="gr-chip">{esc(c)}</span>' for c in chips or [])
    st.markdown(
        f"""<div class="gr-hero"><span class="gr-eyebrow">🛡️ {esc(section)}</span>
        <h1>{esc(title)}</h1><p>{subtitle}</p>
        {'<div class="gr-chips">' + chips_html + '</div>' if chips_html else ''}</div>""",
        unsafe_allow_html=True,
    )


def card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="gr-card"><div class="gr-ico">{icon}</div><h4>{esc(title)}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    st.markdown(f'<div class="gr-kicker">{esc(text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Settings / sidebar
# --------------------------------------------------------------------------- #
def sidebar_settings() -> Settings:
    with st.sidebar:
        st.markdown("#### ⚙️ Model settings")
        env_key = env_api_key()
        typed = st.text_input(
            "OpenRouter API key",
            type="password",
            placeholder="Loaded from .env" if env_key else "sk-or-…",
            help="Used only for this browser session. Falls back to OPENROUTER_KEY from .env.",
        )
        api_key = typed.strip() or env_key
        c1, c2 = st.columns(2)
        agent_model = c1.text_input("Agent model", DEFAULT_AGENT_MODEL, help="Any OpenRouter model id with tool calling, e.g. anthropic/claude-sonnet-4.5")
        guard_model = c2.text_input("Judge model", DEFAULT_GUARD_MODEL, help="Cheaper model used for safety checks")
        if api_key:
            st.markdown(badge("pass", "API key ready"), unsafe_allow_html=True)
        else:
            st.markdown(badge("modify", "No API key — offline demos only"), unsafe_allow_html=True)
        st.caption("Built on LangChain `create_agent` middleware.")
    return Settings(api_key=api_key, agent_model=agent_model.strip(), guard_model=guard_model.strip())


def require_key(settings: Settings) -> bool:
    if settings.has_key:
        return True
    st.info(
        "Add an OpenRouter API key in the sidebar (or `OPENROUTER_KEY` in `.env`) to chat with this agent.",
        icon="🔑",
    )
    return False


# --------------------------------------------------------------------------- #
# Agent/conversation cache (per browser session)
# --------------------------------------------------------------------------- #
def get_conversation(page: str, signature: tuple, build: Callable[[], BuiltAgent]) -> Conversation:
    """Reuse the conversation while settings/config are unchanged; rebuild otherwise."""
    slot = f"_conv_{page}"
    cached = st.session_state.get(slot)
    if cached is None or cached[0] != signature:
        st.session_state[slot] = (signature, Conversation(build()))
        st.session_state.pop(f"_events_{page}", None)
    return st.session_state[slot][1]


def last_events(page: str) -> list[TraceEvent]:
    return st.session_state.get(f"_events_{page}", [])


# --------------------------------------------------------------------------- #
# Pipeline + trace inspector
# --------------------------------------------------------------------------- #
def layer_status(layer: str, events: list[TraceEvent]) -> tuple[str, str]:
    """Collapse all events for a layer into one (status, label) for the diagram."""
    kinds = [e.kind for e in events if e.layer == layer]
    if not kinds:
        return "idle", "—"
    if layer == tracing.MODEL:
        return "llm", f"{len(kinds)} call{'s' if len(kinds) != 1 else ''}"
    for kind in ("block", "reject", "error", "pause", "modify", "approve", "pass"):
        if kind in kinds:
            return ("pass" if kind == "approve" else "block" if kind in ("reject", "error") else kind), KIND_META[kind][2]
    return "idle", "—"


def pipeline(layers: list[tuple[str, str]], events: list[TraceEvent]) -> None:
    """Render the request path: user → guardrail layers (hook name shown) → user."""
    blocked_at = next((i for i, (name, _) in enumerate(layers) if layer_status(name, events)[0] == "block"), None)
    parts = ['<div class="gr-pipe"><div class="gr-node gr-endpoint"><span>👤 User input</span></div>']
    for i, (name, hook) in enumerate(layers):
        status, label = layer_status(name, events)
        skipped = blocked_at is not None and i > blocked_at and status == "idle"
        css = f"s-{status}" + (" s-skipped" if skipped else "")
        right = badge(status if status != "idle" else "info", label) if status != "idle" else (
            '<span class="gr-n-hook">skipped</span>' if skipped else ""
        )
        parts.append('<div class="gr-arrow"></div>')
        parts.append(
            f'<div class="gr-node {css}"><span class="gr-n-left"><span class="gr-n-idx">{i + 1}</span>'
            f'<span>{esc(name)}<br><span class="gr-n-hook">{esc(hook)}</span></span></span>{right}</div>'
        )
    parts.append('<div class="gr-arrow"></div><div class="gr-node gr-endpoint"><span>💬 Response</span></div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def trace_list(events: list[TraceEvent]) -> None:
    if not events:
        st.markdown('<div class="gr-empty">Send a message to see each guardrail fire.</div>', unsafe_allow_html=True)
        return
    rows = []
    for e in events:
        cls, icon, label = KIND_META.get(e.kind, KIND_META["info"])
        t = datetime.fromtimestamp(e.ts).strftime("%H:%M:%S")
        rows.append(
            f'<div class="gr-ev"><span class="gr-badge {cls}">{icon}</span><div>'
            f'<span class="gr-ev-layer">{esc(e.layer)}</span> <span class="gr-n-hook">· {label} · {t}</span>'
            f'<div class="gr-ev-detail">{esc(e.detail)}</div></div></div>'
        )
    st.markdown('<div class="gr-trace">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def inspector(page: str, layers: list[tuple[str, str]]) -> None:
    events = last_events(page)
    with st.container(border=True):
        st.markdown("**🔎 Guardrail inspector** · last turn")
        llm_calls = sum(1 for e in events if e.kind == "llm")
        actions = sum(1 for e in events if e.kind in ("block", "modify", "pause", "reject"))
        m1, m2 = st.columns(2)
        m1.metric("LLM calls", llm_calls)
        m2.metric("Guardrail actions", actions)
        pipeline(layers, events)
        kicker("Event log")
        trace_list([e for e in events if e.kind != "llm"])


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def _tool_chip(call: dict, result: ToolMessage | None) -> str:
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in call["args"].items())
    res = f'<span class="gr-tool-res">↳ {esc(result.text)}</span>' if result is not None else (
        '<span class="gr-tool-res">↳ not executed</span>'
    )
    return f'<div class="gr-tool">🔧 <b>{esc(call["name"])}</b>({esc(args)}){res}</div>'


def render_messages(messages: list[BaseMessage], avatar: str = "🛡️") -> None:
    results = tool_messages_by_call(messages)
    turn: list[BaseMessage] = []

    def flush() -> None:
        if not turn:
            return
        with st.chat_message("assistant", avatar=avatar):
            for m in turn:
                if isinstance(m, AIMessage):
                    for call in m.tool_calls:
                        st.markdown(_tool_chip(call, results.get(call["id"])), unsafe_allow_html=True)
                    if m.text.strip():
                        st.markdown(m.text)
        turn.clear()

    for m in messages:
        if isinstance(m, HumanMessage):
            flush()
            with st.chat_message("user", avatar="🧑"):
                st.markdown(m.text)
        elif isinstance(m, AIMessage):
            turn.append(m)
        # ToolMessages are rendered inline via their tool call chip
    flush()


def approval_panel(page: str, conv: Conversation) -> bool:
    """Render pending HITL interrupts. Returns True when a decision was submitted."""
    interrupts = conv.pending_interrupts()
    if not interrupts:
        return False
    request = interrupts[0].value
    actions = request.get("action_requests", [])
    configs = {c["action_name"]: c for c in request.get("review_configs", [])}
    with st.container(border=True):
        st.markdown(
            f"{badge('pause', 'Awaiting human approval')} &nbsp;The agent paused before running "
            f"{len(actions)} sensitive tool call{'s' if len(actions) != 1 else ''}.",
            unsafe_allow_html=True,
        )
        decisions: list[dict] = []
        for i, action in enumerate(actions):
            allowed = configs.get(action["name"], {}).get("allowed_decisions", ["approve", "edit", "reject"])
            st.markdown(f"**🔧 `{action['name']}`**")
            st.code(json.dumps(action["args"], indent=2, ensure_ascii=False), language="json")
            labels = {"approve": "✅ Approve", "edit": "✏️ Edit", "reject": "❌ Reject"}
            options = [d for d in ("approve", "edit", "reject") if d in allowed]
            choice = st.segmented_control(
                "Decision", options, default=options[0], format_func=labels.get,
                key=f"{page}_dec_{conv.thread_id}_{i}", label_visibility="collapsed",
            ) or options[0]
            if choice == "edit":
                raw = st.text_area(
                    "Edited arguments (JSON)", json.dumps(action["args"], indent=2, ensure_ascii=False),
                    key=f"{page}_edit_{conv.thread_id}_{i}", height=140,
                )
                try:
                    decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": json.loads(raw)}})
                except json.JSONDecodeError:
                    st.error("Arguments must be valid JSON.")
                    decisions.append({})
            elif choice == "reject":
                reason = st.text_input(
                    "Reason sent back to the agent", "Too risky, needs review.", key=f"{page}_rej_{conv.thread_id}_{i}"
                )
                decisions.append({"type": "reject", "message": reason})
            else:
                decisions.append({"type": "approve"})
        with st.expander("Raw interrupt payload"):
            st.json(request)
        if st.button("Submit decision", type="primary", key=f"{page}_submit_{conv.thread_id}",
                     disabled=any(not d for d in decisions), width="stretch"):
            with st.spinner("Resuming agent…"):
                result = conv.resume(decisions)
            st.session_state[f"_events_{page}"] = result.events
            if result.error:
                st.session_state[f"_error_{page}"] = result.error
            return True
    return False


def chat_panel(
    page: str,
    conv: Conversation,
    examples: list[str],
    placeholder: str = "Message the agent…",
    avatar: str = "🛡️",
    height: int = 520,
) -> None:
    """Full chat experience: history, approvals, example prompts, input and error display."""
    head_l, head_r = st.columns([1, 0.22])
    head_l.markdown("**💬 Agent chat**")
    if head_r.button("New chat", key=f"{page}_reset", icon=":material/refresh:", width="stretch"):
        conv.reset()
        st.session_state.pop(f"_events_{page}", None)
        st.session_state.pop(f"_error_{page}", None)
        st.rerun()

    box = st.container(height=height, border=True)
    with box:
        msgs = conv.messages()
        if not msgs:
            st.markdown('<div class="gr-empty">No messages yet — try one of the examples below.</div>',
                        unsafe_allow_html=True)
        render_messages(msgs, avatar=avatar)
        if err := st.session_state.get(f"_error_{page}"):
            st.error(err, icon="🛑")
        if approval_panel(page, conv):
            st.rerun()

    pending = bool(conv.pending_interrupts())
    nonce = st.session_state.get(f"_nonce_{page}", 0)  # new key after each send clears the pill
    picked = st.pills("Try an example", examples, key=f"{page}_ex_{nonce}", label_visibility="collapsed")
    typed = st.chat_input(
        "Resolve the pending approval first…" if pending else placeholder, key=f"{page}_input", disabled=pending
    )
    prompt = typed or picked
    if prompt and not pending:
        st.session_state[f"_nonce_{page}"] = nonce + 1
        st.session_state.pop(f"_error_{page}", None)
        with box:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=avatar), st.spinner("Running guardrails + agent…"):
                result = conv.send(prompt)
        st.session_state[f"_events_{page}"] = result.events
        if result.error:
            st.session_state[f"_error_{page}"] = result.error
        st.rerun()
