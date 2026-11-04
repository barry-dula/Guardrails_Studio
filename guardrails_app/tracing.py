"""A tiny, thread-safe event log that guardrail middleware writes to.

LangGraph may execute nodes on worker threads, so middleware can't write to
Streamlit session state directly. Each agent gets its own ``Tracer``; the UI
drains it after every run and renders the events in the inspector panel.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Literal

EventKind = Literal["pass", "block", "modify", "pause", "approve", "reject", "llm", "error", "info"]

# Canonical layer names, shared by middleware, the trace panel and pipeline diagrams.
CONTENT_FILTER = "Content filter"
PII_INPUT = "PII redaction (input)"
PII_OUTPUT = "PII redaction (output)"
HITL = "Human approval"
MODEL = "LLM"
SAFETY_JUDGE = "Safety judge"
DISCLAIMER = "Medical disclaimer"


@dataclass
class TraceEvent:
    layer: str
    kind: EventKind
    detail: str
    ts: float = field(default_factory=time.time)


class Tracer:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()

    def add(self, layer: str, kind: EventKind, detail: str = "") -> None:
        with self._lock:
            self._events.append(TraceEvent(layer, kind, detail))

    def drain(self) -> list[TraceEvent]:
        with self._lock:
            events, self._events = self._events, []
        return events
