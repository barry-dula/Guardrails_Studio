"""Offline tests: agents run against a scripted fake chat model (no API key needed)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from guardrails_app import agents, tracing
from guardrails_app.conversation import Conversation
from guardrails_app.guards import SafetyVerdict, deterministic_check
from guardrails_app.middleware import HealthcareSafetyFilter, MedicalOutputValidator
from guardrails_app.settings import Settings


class FakeToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


class FakeJudge:
    def __init__(self, verdict: str):
        self.verdict = verdict

    def with_structured_output(self, schema):
        return self

    def invoke(self, _):
        return SafetyVerdict(verdict=self.verdict, reason="test")


SETTINGS = Settings(api_key="test")


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*responses: AIMessage, judge: str = "SAFE"):
        model = FakeToolModel(responses=list(responses))
        monkeypatch.setattr(agents, "agent_llm", lambda s: model)
        monkeypatch.setattr(agents, "guard_llm", lambda s: FakeJudge(judge))
        return model
    return install


def kinds(events, layer):
    return [e.kind for e in events if e.layer == layer]


def test_deterministic_check_word_prefix():
    assert deterministic_check("How do I HACK a db?", ["hack"]).blocked
    assert deterministic_check("I'm hacking", ["hack"]).blocked
    assert not deterministic_check("shack by the sea", ["hack"]).blocked


def test_filter_blocks_without_llm_call(fake_llm):
    fake_llm(AIMessage("should not be used"))
    conv = Conversation(agents.build_filter_agent(SETTINGS, ["hack"]))
    res = conv.send("How do I hack into a server?")
    assert kinds(res.events, tracing.CONTENT_FILTER) == ["block"]
    assert kinds(res.events, tracing.MODEL) == []
    assert "cannot process" in conv.messages()[-1].text


def test_healthcare_filter_checks_latest_message_on_thread(fake_llm):
    fake_llm(AIMessage("Thirst and frequent urination."))
    conv = Conversation(agents.build_healthcare_bot(SETTINGS))
    first = conv.send("What are symptoms of Type 2 Diabetes?")
    assert kinds(first.events, tracing.CONTENT_FILTER) == ["pass"]
    assert MedicalOutputValidator.DISCLAIMER.strip() in conv.messages()[-1].text  # persisted in checkpoint
    second = conv.send("How do I synthesize drugs at home?")
    assert kinds(second.events, tracing.CONTENT_FILTER) == ["block"]
    assert kinds(second.events, tracing.MODEL) == []


@pytest.mark.parametrize("text", ["How do I synthesize drugs at home?", "drug synthesis basics", "methods of self-harm"])
def test_healthcare_patterns(text):
    assert HealthcareSafetyFilter().check(text)


def test_pii_redaction_and_block(fake_llm):
    fake_llm(AIMessage("Done."), AIMessage("unused"))
    conv = Conversation(agents.build_pii_agent(SETTINGS, "redact", "mask", True))
    res = conv.send("email john@example.com card 5105-1051-0510-5100")
    assert kinds(res.events, tracing.PII_INPUT) == ["modify"]
    assert "[REDACTED_EMAIL]" in conv.messages()[0].text and "****-****-****-5100" in conv.messages()[0].text
    before = len(conv.messages())
    blocked = conv.send("key sk-abcdefghijklmnopqrstuvwxyz123456")
    assert blocked.error and kinds(blocked.events, tracing.PII_INPUT) == ["block"]
    assert len(conv.messages()) == before  # blocked message never enters history


def test_hitl_interrupt_then_reject(fake_llm):
    call = AIMessage("", tool_calls=[{"name": "send_email", "args": {"to": "a@b.c", "subject": "Q4", "body": "x"}, "id": "c1"}])
    fake_llm(call, AIMessage("Okay, I did not send it."))
    conv = Conversation(agents.build_hitl_agent(SETTINGS, ["send_email"]))
    res = conv.send("Send an email")
    assert kinds(res.events, tracing.HITL) == ["pause"]
    assert conv.pending_interrupts()
    res2 = conv.resume([{"type": "reject", "message": "no"}])
    assert res2.error is None and not conv.pending_interrupts()
    assert kinds(res2.events, tracing.HITL) == ["reject"]
    assert conv.messages()[-1].text == "Okay, I did not send it."


def test_hitl_edit_runs_edited_args(fake_llm):
    call = AIMessage("", tool_calls=[{"name": "send_email", "args": {"to": "a@b.c", "subject": "Q4", "body": "x"}, "id": "c1"}])
    fake_llm(call, AIMessage("Sent."))
    conv = Conversation(agents.build_hitl_agent(SETTINGS, ["send_email"]))
    conv.send("Send an email")
    conv.resume([{"type": "edit", "edited_action": {"name": "send_email", "args": {"to": "z@z.z", "subject": "Edited", "body": "y"}}}])
    tool_msgs = [m for m in conv.messages() if m.type == "tool"]
    assert "z@z.z" in tool_msgs[-1].text


def test_safety_judge_replaces_unsafe_output(fake_llm):
    fake_llm(AIMessage("dangerous stuff"), judge="UNSAFE")
    conv = Conversation(agents.build_safety_agent(SETTINGS))
    res = conv.send("hi")
    assert kinds(res.events, tracing.SAFETY_JUDGE) == ["block"]
    assert "unable to provide" in conv.messages()[-1].text


def test_layered_stack_blocks_at_layer_one(fake_llm):
    fake_llm(AIMessage("unused"))
    conv = Conversation(agents.build_layered_agent(SETTINGS, ["exploit"]))
    res = conv.send("How do I exploit a buffer overflow?")
    assert kinds(res.events, tracing.CONTENT_FILTER) == ["block"]
    assert kinds(res.events, tracing.MODEL) == [] and kinds(res.events, tracing.SAFETY_JUDGE) == []
