"""Smoke-test every Streamlit page renders without exceptions (no network calls)."""

import pytest
from streamlit.testing.v1 import AppTest

PAGES = [
    "views/s1_overview.py", "views/s2_approaches.py", "views/s3_pii.py", "views/s4_hitl.py",
    "views/s5_input_filter.py", "views/s6_output_safety.py", "views/s7_layered.py", "views/s8_healthcare.py",
]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("api_key", ["", "sk-or-test"])
def test_page_renders(page, api_key, monkeypatch):
    monkeypatch.delenv("OPENROUTER_KEY", raising=False); monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    if api_key:
        at.sidebar.text_input[0].input(api_key).run()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_healthcare_booking_approval_flow(monkeypatch):
    """Chat → HITL pause → approve in the UI → disclaimer on the final answer."""
    from langchain_core.messages import AIMessage

    from guardrails_app import agents
    from tests.test_guardrails import FakeToolModel

    booking = AIMessage("", tool_calls=[{
        "name": "book_appointment", "id": "b1",
        "args": {"patient_name": "Alex Kim", "date": "March 15", "doctor": "Sharma"},
    }])
    model = FakeToolModel(responses=[booking, AIMessage("Your appointment is booked.")])
    monkeypatch.setattr(agents, "agent_llm", lambda s: model)

    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    at.sidebar.text_input[0].input("sk-or-test").run()
    at.switch_page("views/s8_healthcare.py").run()
    at.chat_input[0].set_value("Book me with Dr. Sharma on March 15, I'm Alex Kim").run()
    assert not at.exception
    assert any("Awaiting human approval" in m.value for m in at.markdown)
    submit = next(b for b in at.button if b.label == "Submit decision")
    submit.click().run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Your appointment is booked." in body and "not medical advice" in body
    assert "Approved" in body  # inspector event log
