"""Custom guardrail middleware (Sections 5, 6 and 8 of the course).

Differences from the notebook versions, all deliberate:

* Input filters inspect the *latest* human message, not ``messages[0]``. With a
  checkpointer the first message belongs to an earlier turn, so the notebook
  version stops filtering after the first exchange.
* Output guards replace the final ``AIMessage`` by returning a message with the
  same id (the ``add_messages`` reducer swaps it in), rather than mutating it in
  place, so the change is persisted in the checkpoint.
* The healthcare filter uses regexes, so "How do I synthesize drugs at home?"
  is caught (the notebook's literal "drug synthesis" keyword missed it).
"""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from . import tracing
from .guards import OUTPUT_JUDGE_PROMPT, first_match, judge, keyword_patterns
from .tracing import Tracer


def latest_human_text(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return None


# Name stamped on refusals written by guardrails. ``jump_to: "end"`` from
# ``before_agent`` still runs every ``after_agent`` hook, so output guards use
# this to skip reviewing (and paying an LLM to judge) a guardrail's own refusal.
GUARDRAIL_AUTHOR = "guardrail"


def final_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """The model's final answer: text, no pending tool calls, not a guardrail refusal."""
    if not messages:
        return None
    last = messages[-1]
    if (
        isinstance(last, AIMessage)
        and last.name != GUARDRAIL_AUTHOR
        and not last.tool_calls
        and last.text.strip()
    ):
        return last
    return None


class _Traced(AgentMiddleware):
    def __init__(self, tracer: Tracer | None = None) -> None:
        super().__init__()
        self.tracer = tracer

    def _trace(self, layer: str, kind: tracing.EventKind, detail: str = "") -> None:
        if self.tracer:
            self.tracer.add(layer, kind, detail)


# --------------------------------------------------------------------------- #
# Section 5 - before_agent: deterministic input filter
# --------------------------------------------------------------------------- #
class ContentFilterMiddleware(_Traced):
    """Block requests that match banned patterns before any LLM call is made."""

    REFUSAL = "I cannot process requests containing inappropriate content. Please rephrase your request."

    def __init__(
        self,
        banned_keywords: list[str] | None = None,
        *,
        patterns: list[re.Pattern[str]] | None = None,
        refusal: str | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        super().__init__(tracer)
        self.patterns = list(patterns or []) + keyword_patterns(banned_keywords or [])
        self.refusal = refusal or self.REFUSAL

    def check(self, text: str) -> str | None:
        """Return the offending text fragment, or None if the input is allowed."""
        match = first_match(text, self.patterns)
        return match.group(0) if match else None

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        text = latest_human_text(state["messages"])
        if text is None:
            return None
        if hit := self.check(text):
            self._trace(tracing.CONTENT_FILTER, "block", f"Matched “{hit}” — request ended before any LLM call")
            return {
                "messages": [AIMessage(content=self.refusal, name=GUARDRAIL_AUTHOR)],
                "jump_to": "end",
            }
        self._trace(tracing.CONTENT_FILTER, "pass", "No banned pattern found")
        return None


# --------------------------------------------------------------------------- #
# Section 6 - after_agent: model-based output safety
# --------------------------------------------------------------------------- #
class SafetyGuardrailMiddleware(_Traced):
    """Use a small LLM as a judge on the agent's final response."""

    FALLBACK = "I'm unable to provide that response. Please rephrase your request or contact support."

    def __init__(self, judge_model: ChatOpenAI, *, tracer: Tracer | None = None) -> None:
        super().__init__(tracer)
        self.judge_model = judge_model

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        final = final_ai_message(state["messages"])
        if final is None:
            return None
        verdict = judge(self.judge_model, OUTPUT_JUDGE_PROMPT, final.text)
        if verdict.verdict == "UNSAFE":
            self._trace(tracing.SAFETY_JUDGE, "block", f"UNSAFE — {verdict.reason} (response replaced)")
            return {"messages": [AIMessage(id=final.id, content=self.FALLBACK)]}
        self._trace(tracing.SAFETY_JUDGE, "pass", f"SAFE — {verdict.reason}")
        return None


# --------------------------------------------------------------------------- #
# Section 8 - healthcare-specific guardrails
# --------------------------------------------------------------------------- #
HEALTHCARE_BLOCK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsynthesi[sz]\w*\b.{0,40}\bdrugs?\b",
        r"\bdrugs?\b.{0,40}\bsynthesi[sz]\w*",
        r"\b(cook|make|manufactur\w*)\b.{0,30}\b(meth\w*|fentanyl|heroin|cocaine)\b",
        r"\bself[- ]harm\b",
        r"\bsuicide method",
        r"\b(kill|hurt)\s+my\s*self\b",
        r"\bweapons?\b",
        r"\bhack\w*",
    ]
]


class HealthcareSafetyFilter(ContentFilterMiddleware):
    REFUSAL = (
        "I'm a healthcare assistant and can only help with medical questions, appointments, "
        "and health information. If you're in crisis, please call 112 or your local emergency number."
    )

    def __init__(self, *, tracer: Tracer | None = None) -> None:
        super().__init__(patterns=HEALTHCARE_BLOCK_PATTERNS, refusal=self.REFUSAL, tracer=tracer)


class MedicalOutputValidator(_Traced):
    """Append a medical disclaimer to every final answer that lacks one."""

    DISCLAIMER = (
        "\n\n⚕️ *This is general health information, not medical advice. "
        "Please consult a qualified healthcare professional.*"
    )

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        final = final_ai_message(state["messages"])
        if final is None or self.DISCLAIMER.strip() in final.text:
            return None
        self._trace(tracing.DISCLAIMER, "modify", "Disclaimer appended to the final answer")
        return {"messages": [AIMessage(id=final.id, content=final.text + self.DISCLAIMER)]}


# --------------------------------------------------------------------------- #
# Observability helper - counts real model calls
# --------------------------------------------------------------------------- #
class ModelCallCounter(_Traced):
    """Records every model invocation, which shows blocked requests cost nothing."""

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._trace(tracing.MODEL, "llm", "Model call")
        return None
