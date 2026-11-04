import streamlit as st

from guardrails_app import tracing, ui
from guardrails_app.agents import build_healthcare_bot
from guardrails_app.settings import Settings

settings: Settings = st.session_state["settings"]
PAGE = "health"
LAYERS = [
    (tracing.CONTENT_FILTER, "HealthcareSafetyFilter · before_agent"),
    (tracing.PII_INPUT, "PIIMiddleware · email + credit_card"),
    (tracing.MODEL, "clinic assistant + 3 tools"),
    (tracing.HITL, "approval for book_appointment"),
    (tracing.DISCLAIMER, "MedicalOutputValidator · after_agent"),
]

ui.hero(
    "Section 8 · Real-world use case",
    "CareBot: a guarded healthcare assistant",
    "A clinic assistant that <b>blocks</b> harmful or off-topic requests with a crisis-safe message, <b>redacts</b> "
    "patient PII, <b>requires staff approval</b> before booking appointments, and <b>adds a medical disclaimer</b> "
    "to every answer.",
    ["search_symptoms", "get_medication_info", "book_appointment 🔒"],
)

c = st.columns(4)
for col, (icon, title, body) in zip(c, [
    ("🚧", "Topic & harm filter", "Regex rules for drug synthesis, self-harm, weapons and hacking."),
    ("🔏", "Patient privacy", "Emails are redacted and card numbers masked before the model sees them."),
    ("🧑‍⚕️", "Staff approval", "Bookings pause for a human to approve, edit or reject."),
    ("⚕️", "Disclaimers", "Every final answer ends with a not-medical-advice notice."),
]):
    with col:
        ui.card(icon, title, body)
st.write("")

chat_col, side_col = st.columns([1.45, 1], gap="large")
with side_col:
    ui.inspector(PAGE, LAYERS)
with chat_col:
    if ui.require_key(settings):
        conv = ui.get_conversation(PAGE, settings.signature(), lambda: build_healthcare_bot(settings))
        ui.chat_panel(
            PAGE, conv, avatar="🩺", height=560, placeholder="Ask CareBot about symptoms, medication or appointments…",
            examples=[
                "What are symptoms of Type 2 Diabetes?",
                "My email is patient123@gmail.com. What can I take for a headache?",
                "How do I synthesize drugs at home?",
                "Book me an appointment with Dr. Sharma on March 15. My name is Alex Kim.",
            ],
        )

with st.expander("🐞 Fixes compared with the notebook version"):
    st.markdown(
        "- **The filter checks the latest message.** The notebook checked `messages[0]`, so on a reused thread "
        "only the very first question was ever filtered. That's why *“How do I synthesize drugs at home?”* "
        "got through in the notebook.\n"
        "- **Regex patterns instead of a literal phrase.** `\"drug synthesis\"` doesn't match *“synthesize drugs”*.\n"
        "- **The disclaimer is persisted.** The final message is replaced by id rather than mutated in place, and the "
        "check looks for the exact disclaimer, not any mention of “medical advice”.\n"
        "- **Each chat gets its own `thread_id`**, so tests don't leak history into each other."
    )
