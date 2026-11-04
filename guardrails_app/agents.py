"""Agent factories — one per course section — built with ``create_agent``.

Middleware order in each list is the order the guardrail layers run in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, PIIMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from . import tools as T
from .middleware import (
    ContentFilterMiddleware,
    HealthcareSafetyFilter,
    MedicalOutputValidator,
    ModelCallCounter,
    SafetyGuardrailMiddleware,
)
from .settings import Settings, agent_llm, guard_llm
from .tracing import Tracer

API_KEY_REGEX = r"sk-[a-zA-Z0-9]{32}"


@dataclass
class BuiltAgent:
    graph: object
    tracer: Tracer
    checkpointed: bool
    layers: list[str] = field(default_factory=list)


# Section 3 ------------------------------------------------------------------ #
def build_pii_agent(settings: Settings, email: str, card: str, block_api_keys: bool) -> BuiltAgent:
    tracer = Tracer()
    middleware = [
        PIIMiddleware("email", strategy=email, apply_to_input=True),
        PIIMiddleware("credit_card", strategy=card, apply_to_input=True),
    ]
    if block_api_keys:
        middleware.append(PIIMiddleware("api_key", detector=API_KEY_REGEX, strategy="block", apply_to_input=True))
    middleware.append(ModelCallCounter(tracer))
    graph = create_agent(model=agent_llm(settings), tools=[T.customer_lookup], middleware=middleware)
    return BuiltAgent(graph, tracer, checkpointed=False)


# Section 4 ------------------------------------------------------------------ #
def build_hitl_agent(settings: Settings, gated_tools: list[str]) -> BuiltAgent:
    tracer = Tracer()
    graph = create_agent(
        model=agent_llm(settings),
        tools=list(T.HITL_TOOLS.values()),
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={name: name in gated_tools for name in T.HITL_TOOLS}),
            ModelCallCounter(tracer),
        ],
        checkpointer=InMemorySaver(),  # required: interrupts persist state between runs
    )
    return BuiltAgent(graph, tracer, checkpointed=True)


# Section 5 ------------------------------------------------------------------ #
def build_filter_agent(settings: Settings, keywords: list[str]) -> BuiltAgent:
    tracer = Tracer()
    graph = create_agent(
        model=agent_llm(settings),
        tools=[T.search_web],
        middleware=[ContentFilterMiddleware(keywords, tracer=tracer), ModelCallCounter(tracer)],
    )
    return BuiltAgent(graph, tracer, checkpointed=False)


# Section 6 ------------------------------------------------------------------ #
def build_safety_agent(settings: Settings, system_prompt: str | None = None) -> BuiltAgent:
    tracer = Tracer()
    graph = create_agent(
        model=agent_llm(settings),
        tools=[T.search_web],
        middleware=[SafetyGuardrailMiddleware(guard_llm(settings), tracer=tracer), ModelCallCounter(tracer)],
        system_prompt=system_prompt,
    )
    return BuiltAgent(graph, tracer, checkpointed=False)


# Section 7 ------------------------------------------------------------------ #
def build_layered_agent(settings: Settings, keywords: list[str]) -> BuiltAgent:
    tracer = Tracer()
    graph = create_agent(
        model=agent_llm(settings),
        tools=[T.search_web, T.send_email],
        middleware=[
            ContentFilterMiddleware(keywords, tracer=tracer),                         # 1 deterministic input
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),       # 2 PII on input
            HumanInTheLoopMiddleware(interrupt_on={"send_email": True, "search_web": False}),  # 3 approval
            # 4 PII on output only. apply_to_input defaults to True, which would hide the
            # recipient from the model and send_email would run with to="[REDACTED_EMAIL]".
            PIIMiddleware("email", strategy="redact", apply_to_input=False, apply_to_output=True),
            SafetyGuardrailMiddleware(guard_llm(settings), tracer=tracer),           # 5 model-based output
            ModelCallCounter(tracer),
        ],
        checkpointer=InMemorySaver(),
    )
    return BuiltAgent(graph, tracer, checkpointed=True)


# Section 8 ------------------------------------------------------------------ #
HEALTHCARE_SYSTEM_PROMPT = (
    "You are a helpful, empathetic healthcare assistant for a clinic. You can search symptoms, "
    "look up medication information, and book appointments. Before booking, make sure you know the "
    "patient's name, the doctor and the date — ask for anything missing. Always remind users to "
    "consult a doctor for diagnosis. Never ask for or repeat email addresses or card numbers."
)


def build_healthcare_bot(settings: Settings) -> BuiltAgent:
    tracer = Tracer()
    graph = create_agent(
        model=agent_llm(settings),
        tools=[T.search_symptoms, T.book_appointment, T.get_medication_info],
        middleware=[
            HealthcareSafetyFilter(tracer=tracer),
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "book_appointment": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "search_symptoms": False,
                    "get_medication_info": False,
                }
            ),
            MedicalOutputValidator(tracer),
            ModelCallCounter(tracer),
        ],
        checkpointer=InMemorySaver(),
        system_prompt=HEALTHCARE_SYSTEM_PROMPT,
    )
    return BuiltAgent(graph, tracer, checkpointed=True)
