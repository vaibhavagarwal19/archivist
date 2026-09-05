"""Tests for the agent's deterministic routing layer.

The LangGraph agent asks an LLM to classify intent and plan tool calls, but
every one of those decisions has a deterministic fallback for when the model
is unavailable or returns unparseable output. Those fallbacks are pure
functions, so they are tested here without touching the network.
"""
import pytest

from app.services.langgraph_agent import AgentState, LangGraphAgentService

heuristic_intent = LangGraphAgentService._heuristic_intent
heuristic_should_retrieve = LangGraphAgentService._heuristic_should_retrieve
requires_specific_document = LangGraphAgentService._requires_specific_document
fallback_plan = LangGraphAgentService._fallback_plan


@pytest.mark.parametrize(
    "query,expected",
    [
        ("hello", "greeting"),
        ("Hi", "greeting"),
        ("good morning", "greeting"),
        ("summarize the meeting notes", "summarization"),
        ("give me the minutes of meeting", "summarization"),
        ("generate report for leadership", "reporting"),
        ("send me a weekly update", "reporting"),
        ("create ticket for the payroll bug", "workflow"),
        ("notify the ops team", "workflow"),
        ("escalate this incident", "workflow"),
        ("what does the handbook say about leave?", "knowledge_lookup"),
        ("check the policy document", "knowledge_lookup"),
        ("what is the capital of France", "general_chat"),
    ],
)
def test_heuristic_intent_classification(query: str, expected: str):
    assert heuristic_intent(query) == expected


def test_greeting_requires_exact_match_not_substring():
    """'hi' is a greeting; a sentence that merely contains it is not."""
    assert heuristic_intent("hi") == "greeting"
    assert heuristic_intent("which vendor did we hire for this?") != "greeting"


@pytest.mark.parametrize(
    "intent,expected",
    [
        ("knowledge_lookup", True),
        ("reporting", True),
        ("summarization", True),
        ("workflow", False),
    ],
)
def test_retrieval_decision_follows_intent(intent: str, expected: bool):
    assert heuristic_should_retrieve("any query", intent) is expected


def test_general_chat_retrieves_only_on_document_signals():
    assert heuristic_should_retrieve("what does the uploaded file say", "general_chat") is True
    assert heuristic_should_retrieve("tell me a joke", "general_chat") is False


@pytest.mark.parametrize(
    "query,expected",
    [
        ("summarize the uploaded document", True),
        ("what does this document say", True),
        ("pull details from the uploaded file", True),
        ("summarize our hiring process", False),
    ],
)
def test_specific_document_detection(query: str, expected: bool):
    assert requires_specific_document(query) is expected


def test_workflow_fallback_plans_ticket_and_email_steps():
    state = AgentState(
        query="Escalate the payroll incident and notify the ops team by email",
        intent="workflow",
    )
    plan = fallback_plan(state)

    assert plan.should_execute_tools is True
    tool_names = [step.tool_name for step in plan.steps]
    assert tool_names == ["get_data", "create_ticket", "send_email"]
    assert all(step.rationale for step in plan.steps), "each step should explain itself"


def test_workflow_fallback_omits_unrequested_steps():
    state = AgentState(query="look up our current headcount", intent="workflow")
    plan = fallback_plan(state)
    assert [step.tool_name for step in plan.steps] == ["get_data"]


def test_reporting_fallback_plans_a_report():
    state = AgentState(query="build a weekly ops report", intent="reporting")
    plan = fallback_plan(state)
    assert [step.tool_name for step in plan.steps] == ["generate_report"]


def test_summarization_fallback_plans_a_summary():
    state = AgentState(query="summarize these notes", intent="summarization")
    plan = fallback_plan(state)
    assert [step.tool_name for step in plan.steps] == ["summarize_meeting"]


def test_general_chat_fallback_plans_no_tools():
    state = AgentState(query="what is our mission", intent="general_chat")
    plan = fallback_plan(state)
    assert plan.should_execute_tools is False
    assert plan.steps == []


def test_fallback_prefers_document_brief_over_raw_query():
    """Retrieved context should feed the report prompt, not the bare question."""
    state = AgentState(
        query="summarize it",
        intent="reporting",
        document_brief="Q3 hiring stalled on two open roles.",
    )
    plan = fallback_plan(state)
    assert "hiring stalled" in plan.steps[0].args["prompt"]


def test_fallback_falls_back_to_context_then_query():
    with_context = AgentState(
        query="summarize it", intent="summarization", context=["Retrieved chunk about payroll."]
    )
    assert "payroll" in fallback_plan(with_context).steps[0].args["meeting_notes"]

    bare = AgentState(query="summarize this text", intent="summarization")
    assert fallback_plan(bare).steps[0].args["meeting_notes"] == "summarize this text"
