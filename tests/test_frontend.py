"""Headless smoke tests for the Streamlit console.

These run the real app script through Streamlit's test harness with the HTTP
layer stubbed, so they catch render-time errors - such as the nested-expander
exception that only appears once an assistant message carries a full trace -
without a browser or a running API.
"""
from __future__ import annotations

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP = "frontend/streamlit_app.py"

USER = {"id": 1, "email": "demo@example.com", "full_name": "Demo User", "is_active": True}

DOCUMENTS = {
    "documents": [
        {"id": 1, "file_name": "ops_playbook.txt", "created_at": "2026-09-05T12:40:11.000000"},
        {"id": 2, "file_name": "strategy_notes.txt", "created_at": "2026-09-04T09:12:44.000000"},
    ]
}

# A response exercising every branch of the trace renderer at once.
CHAT_RESPONSE = {
    "intent": "knowledge_lookup",
    "confidence": 0.92,
    "intent_reason": "The question asks about a documented internal process.",
    "answer": "## Payroll sync\n\nThe playbook defines a **three-tier** path:\n\n1. Tier 1\n2. Tier 2\n",
    "clarification_needed": False,
    "candidate_documents": [],
    "context": [
        "Document: ops_playbook.txt\nContent: On-call acknowledges within 15 minutes.",
        "Document: strategy_notes.txt\nContent: Finance lead is paged for Tier 3.",
    ],
    "plan": [
        {"tool_name": "get_data", "args": {"metric": "payroll"}, "rationale": "Check internal records."},
        {"tool_name": "create_ticket", "args": {"team": "ops", "issue": "review"}, "rationale": "Track follow-up."},
    ],
    "tool_calls": [
        {"name": "get_data", "args": {"metric": "payroll"}, "output": "20 document(s)."},
        {"name": "create_ticket", "args": {"team": "ops"}, "output": "Ticket #42 created."},
    ],
}

CLARIFY_RESPONSE = {
    "intent": "knowledge_lookup",
    "confidence": 0.4,
    "intent_reason": "Several documents match.",
    "answer": "Which document did you mean?",
    "clarification_needed": True,
    "candidate_documents": [
        {"document_id": 1, "file_name": "ops_playbook.txt", "match_count": 3},
        {"document_id": 2, "file_name": "strategy_notes.txt", "match_count": 2},
    ],
    "context": [],
    "plan": [],
    "tool_calls": [],
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400
        self.headers = {"content-type": "application/json"}
        self.content = b"{}"

    def json(self):
        return self._payload


@pytest.fixture
def stub_api(monkeypatch):
    """Route every outbound HTTP call to canned payloads."""
    state = {"chat": CHAT_RESPONSE}

    def fake_request(method, url, **kwargs):
        if url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        if "/auth/me" in url:
            return FakeResponse(USER)
        if "/auth/login" in url:
            return FakeResponse({"access_token": "stub", "token_type": "bearer"})
        if "/chat" in url:
            return FakeResponse(state["chat"])
        if "upload-doc" in url:
            return FakeResponse(DOCUMENTS)
        return FakeResponse({"detail": "not found"}, 404)

    monkeypatch.setattr(requests, "request", fake_request)
    return state


def signed_in_app(timeout: int = 30) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.session_state["access_token"] = "stub"
    app.session_state["current_user"] = USER
    return app


def test_signed_out_app_renders_without_error(stub_api):
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception


def test_signed_in_chat_page_renders(stub_api):
    app = signed_in_app().run()
    assert not app.exception


def test_assistant_message_with_full_trace_renders(stub_api):
    """Regression: the trace expander must not contain nested expanders."""
    app = signed_in_app()
    app.session_state["messages"] = [
        {"role": "user", "content": "what does the playbook say?"},
        {"role": "assistant", "content": CHAT_RESPONSE["answer"], "result": CHAT_RESPONSE},
    ]
    app.run()
    assert not app.exception, f"trace rendering raised: {app.exception}"


def test_clarification_message_renders(stub_api):
    app = signed_in_app()
    app.session_state["messages"] = [
        {"role": "user", "content": "summarize the uploaded document"},
        {"role": "assistant", "content": CLARIFY_RESPONSE["answer"], "result": CLARIFY_RESPONSE},
    ]
    app.run()
    assert not app.exception


def test_failed_request_message_renders(stub_api):
    app = signed_in_app()
    app.session_state["messages"] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Request failed: boom", "result": None},
    ]
    app.run()
    assert not app.exception


def test_documents_page_renders(stub_api):
    app = signed_in_app()
    app.session_state["page"] = "Documents"
    app.session_state["documents"] = DOCUMENTS["documents"]
    app.run()
    assert not app.exception


def test_direct_tools_page_renders(stub_api):
    app = signed_in_app()
    app.session_state["page"] = "Direct tools"
    app.run()
    assert not app.exception


def test_timestamps_are_humanised():
    import sys

    sys.path.insert(0, "frontend")
    from streamlit_app import format_timestamp

    assert format_timestamp("2026-09-05T12:40:11.000000") == "05 Sep 2026, 12:40"
    assert format_timestamp(None) == "unknown"
    assert format_timestamp("not-a-date") == "not-a-date"


def test_context_items_split_into_source_and_text():
    import sys

    sys.path.insert(0, "frontend")
    from streamlit_app import parse_context_item

    name, text = parse_context_item("Document: ops.txt\nContent: On-call acknowledges.")
    assert name == "ops.txt"
    assert text == "On-call acknowledges."

    name, text = parse_context_item("unstructured chunk")
    assert name == "source"
