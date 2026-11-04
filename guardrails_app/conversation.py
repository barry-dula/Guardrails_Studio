"""Runs an agent turn-by-turn and turns built-in middleware effects into trace events.

Stateless agents keep history here and get the whole transcript on each call, so
a request blocked by ``PIIMiddleware(strategy="block")`` never enters history.
Checkpointed agents (needed for human-in-the-loop) keep history in LangGraph,
keyed by ``thread_id``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langchain.agents.middleware._redaction import PIIDetectionError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt

from . import tracing
from .agents import BuiltAgent
from .tracing import TraceEvent


@dataclass
class TurnResult:
    events: list[TraceEvent] = field(default_factory=list)
    error: str | None = None


class Conversation:
    def __init__(self, agent: BuiltAgent) -> None:
        self.agent = agent
        self.thread_id = uuid.uuid4().hex
        self._messages: list[BaseMessage] = []

    # -- state ---------------------------------------------------------------- #
    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def messages(self) -> list[BaseMessage]:
        if self.agent.checkpointed:
            return list(self.agent.graph.get_state(self.config).values.get("messages", []))
        return list(self._messages)

    def pending_interrupts(self) -> list[Interrupt]:
        if not self.agent.checkpointed:
            return []
        return list(self.agent.graph.get_state(self.config).interrupts)

    def reset(self) -> None:
        self.thread_id = uuid.uuid4().hex
        self._messages = []
        self.agent.tracer.drain()

    # -- running ---------------------------------------------------------------- #
    def send(self, text: str) -> TurnResult:
        user = HumanMessage(content=text)
        if self.agent.checkpointed:
            payload = {"messages": [user]}
        else:
            payload = {"messages": [*self._messages, user]}
        result = self._run(payload)
        if result.error is None:
            # PII redaction runs in before_model, i.e. after any before_agent content filter.
            after_filter = sum(1 for e in result.events if e.layer == tracing.CONTENT_FILTER)
            result.events[after_filter:after_filter] = self._input_redaction_events(text)
        return result

    def resume(self, decisions: list[dict]) -> TurnResult:
        result = self._run(Command(resume={"decisions": decisions}))
        for decision in decisions:
            kind = "approve" if decision["type"] in ("approve", "edit") else "reject"
            label = {"approve": "Approved", "edit": "Edited & approved", "reject": "Rejected"}[decision["type"]]
            result.events.insert(0, TraceEvent(tracing.HITL, kind, f"{label} by human reviewer"))
        return result

    def _run(self, payload) -> TurnResult:
        events: list[TraceEvent] = []
        try:
            state = self.agent.graph.invoke(payload, config=self.config if self.agent.checkpointed else None)
        except PIIDetectionError as exc:
            events = self.agent.tracer.drain()
            events.append(TraceEvent(tracing.PII_INPUT, "block", str(exc)))
            return TurnResult(events, error=f"Blocked by PII guardrail: {exc}")
        except Exception as exc:  # surface API/config errors in the UI instead of crashing
            events = self.agent.tracer.drain()
            events.append(TraceEvent("Runtime", "error", f"{type(exc).__name__}: {exc}"))
            return TurnResult(events, error=f"{type(exc).__name__}: {exc}")

        if not self.agent.checkpointed:
            self._messages = list(state["messages"])
        events = self.agent.tracer.drain()

        if state.get("__interrupt__"):
            for intr in state["__interrupt__"]:
                for req in intr.value.get("action_requests", []):
                    events.append(TraceEvent(tracing.HITL, "pause", f"Paused before `{req['name']}` — awaiting review"))
        final = state["messages"][-1] if state.get("messages") else None
        if isinstance(final, AIMessage) and "[REDACTED_EMAIL]" in final.text:
            events.append(TraceEvent(tracing.PII_OUTPUT, "modify", "Email address redacted from the response"))
        return TurnResult(events)

    def _input_redaction_events(self, original: str) -> list[TraceEvent]:
        """PIIMiddleware rewrites the stored user message; diff it against what was typed."""
        stored = next((m for m in reversed(self.messages()) if isinstance(m, HumanMessage)), None)
        if stored is not None and stored.text != original:
            return [TraceEvent(tracing.PII_INPUT, "modify", f"Model saw: {stored.text}")]
        return []


def tool_messages_by_call(messages: list[BaseMessage]) -> dict[str, ToolMessage]:
    return {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
