from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests
import streamlit as st


def _resolve_api_base() -> str:
    """Where the API lives.

    In a container the API is a different host, so this must be injectable.
    Render's `fromService` supplies a bare hostname with no scheme, so add one
    when it is missing - https for real hosts, http for loopback.
    """
    value = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").strip()
    if not value:
        return "http://127.0.0.1:8000"
    if "://" not in value:
        local = value.startswith(("localhost", "127.0.0.1"))
        value = f"{'http' if local else 'https'}://{value}"
    return value.rstrip("/")


# The sidebar can still override this at run time.
DEFAULT_API_BASE = _resolve_api_base()


def init_state() -> None:
    defaults = {
        "api_base": DEFAULT_API_BASE,
        "access_token": None,
        "current_user": None,
        "message": None,
        "health": None,
        "documents": [],
        "chat_result": None,
        "report_result": None,
        "summary_result": None,
        "workflow_result": None,
        "chat_query": "Summarize the hiring blockers from our latest internal documents and tell me who needs follow-up.",
        "chat_use_rag": True,
        "chat_document_id": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_url(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def api_request(
    path: str,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
) -> Any:
    headers: dict[str, str] = {}
    token = st.session_state.get("access_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.request(
        method=method,
        url=build_url(path),
        json=json_body,
        files=files,
        headers=headers,
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or "Request failed.")

    if "application/json" in response.headers.get("content-type", ""):
        return response.json()
    return response.text


def set_message(message_type: str, text: str) -> None:
    st.session_state.message = {"type": message_type, "text": text}


def show_message() -> None:
    message = st.session_state.message
    if not message:
        return
    if message["type"] == "success":
        st.success(message["text"])
    elif message["type"] == "error":
        st.error(message["text"])
    else:
        st.info(message["text"])


def refresh_health() -> None:
    try:
        data = api_request("/health")
        st.session_state.health = {
            "status": data.get("status", "ok"),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        st.session_state.health = {
            "status": f"offline: {exc}",
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }


def hydrate_session() -> None:
    """Load the signed-in user and their documents, if a token is present."""
    if not st.session_state.get("access_token"):
        st.session_state.current_user = None
        st.session_state.documents = []
        return
    try:
        st.session_state.current_user = api_request("/api/v1/auth/me")
        refresh_documents()
    except Exception as exc:
        # An expired or invalid token should drop the session, not wedge the UI.
        st.session_state.access_token = None
        st.session_state.current_user = None
        st.session_state.documents = []
        set_message("error", f"Session ended, please sign in again: {exc}")


def sign_in(email: str, password: str) -> None:
    token = api_request(
        "/api/v1/auth/login",
        method="POST",
        json_body={"email": email, "password": password},
    )
    st.session_state.access_token = token["access_token"]
    hydrate_session()
    set_message("success", f"Signed in as {email}.")


def sign_up(email: str, full_name: str, password: str) -> None:
    api_request(
        "/api/v1/auth/register",
        method="POST",
        json_body={"email": email, "full_name": full_name, "password": password},
    )
    sign_in(email, password)


def sign_out() -> None:
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.documents = []
    for key in ("chat_result", "report_result", "summary_result", "workflow_result"):
        st.session_state[key] = None
    set_message("info", "Signed out.")


def refresh_documents() -> None:
    st.session_state.documents = api_request("/api/v1/upload-doc").get("documents", [])


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Control Center")
        api_base = st.text_input("API base URL", value=st.session_state.api_base)
        if api_base != st.session_state.api_base:
            st.session_state.api_base = api_base

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check API", use_container_width=True):
                refresh_health()
        with col2:
            if st.button("Refresh Docs", use_container_width=True):
                try:
                    refresh_documents()
                    set_message("success", "Document list refreshed.")
                except Exception as exc:
                    set_message("error", str(exc))

        health = st.session_state.health or {"status": "unknown", "checked_at": "not checked"}
        st.caption(f"Health: {health['status']}")
        st.caption(f"Checked: {health['checked_at']}")
        st.divider()
        st.caption(
            "Run the API with `python -m uvicorn app.main:app --reload` and the UI with `streamlit run frontend/streamlit_app.py`."
        )


def render_workspace_status() -> None:
    st.markdown("### Workspace Access")
    user = st.session_state.current_user

    if user:
        st.success(f"Signed in as {user.get('full_name') or user.get('email')}")
        st.code(user.get("email", ""), language="text")
        st.caption("Documents, reports, and workflow actions are scoped to this account.")
        if st.button("Sign out", use_container_width=True):
            sign_out()
            st.rerun()
        return

    st.info("Sign in to use the workspace. Each account sees only its own documents.")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("sign_in_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Sign in", use_container_width=True):
                try:
                    sign_in(email, password)
                    st.rerun()
                except Exception as exc:
                    set_message("error", str(exc))

    with register_tab:
        with st.form("sign_up_form"):
            new_email = st.text_input("Email", key="register_email")
            new_name = st.text_input("Full name", key="register_name")
            new_password = st.text_input(
                "Password", type="password", key="register_password",
                help="At least 8 characters.",
            )
            if st.form_submit_button("Create account", use_container_width=True):
                try:
                    sign_up(new_email, new_name, new_password)
                    st.rerun()
                except Exception as exc:
                    set_message("error", str(exc))


def render_overview() -> None:
    user = st.session_state.current_user
    docs = st.session_state.documents
    c1, c2, c3 = st.columns(3)
    c1.metric("Signed in as", user.get("full_name") or user.get("email") if user else "-")
    c2.metric("Your documents", len(docs))
    c3.metric("Agent tools", "5")


def get_document_scope_options() -> list[dict[str, Any]]:
    options = [{"label": "All uploaded documents", "value": None}]
    options.extend(
        {
            "label": item.get("file_name") or f"Document {item.get('id')}",
            "value": item.get("id"),
        }
        for item in st.session_state.documents
    )
    return options


def render_chat() -> None:
    st.markdown("### AI Chat")
    st.caption(
        "Use AI Chat as the main entry point. The agent can answer directly, retrieve uploaded context, "
        "generate reports, summarize meetings, and run workflow actions."
    )
    with st.expander("Try these chat prompts"):
        st.code(
            "what does the uploaded document say about payroll sync?\n\n"
            "summarize these meeting notes from the uploaded document\n\n"
            "create a weekly operations report about hiring blockers, vendor delays, and payroll risk from the uploaded document\n\n"
            "investigate payroll sync, create a ticket, and notify ops with an update email",
            language="text",
        )
    document_options = get_document_scope_options()
    selected_value = st.session_state.chat_document_id
    selected_index = next(
        (index for index, option in enumerate(document_options) if option["value"] == selected_value),
        0,
    )
    with st.form("chat_form"):
        selected_index = st.selectbox(
            "Document scope",
            options=range(len(document_options)),
            format_func=lambda index: document_options[index]["label"],
            index=selected_index,
            help="Choose one document when your prompt refers to 'the uploaded document'. Leave it on all documents for broad search.",
        )
        query = st.text_area(
            "Prompt",
            value=st.session_state.chat_query,
            height=140,
        )
        use_rag = st.checkbox("Use RAG for document context", value=st.session_state.chat_use_rag)
        submitted = st.form_submit_button("Run chat", use_container_width=True)
    if submitted:
        try:
            st.session_state.chat_query = query
            st.session_state.chat_use_rag = use_rag
            st.session_state.chat_document_id = document_options[selected_index]["value"]
            st.session_state.chat_result = api_request(
                "/api/v1/chat",
                method="POST",
                json_body={
                    "query": query,
                    "use_rag": use_rag,
                    "document_id": document_options[selected_index]["value"],
                },
            )
            set_message("success", "Chat response received.")
        except Exception as exc:
            set_message("error", str(exc))

    result = st.session_state.chat_result
    if not result:
        st.info("Your next agent answer will appear here.")
        return

    if st.session_state.chat_document_id is not None:
        chosen = next(
            (
                item.get("file_name")
                for item in st.session_state.documents
                if item.get("id") == st.session_state.chat_document_id
            ),
            f"Document {st.session_state.chat_document_id}",
        )
        st.caption(f"Scoped to: {chosen}")
    else:
        st.caption("Scoped to: all uploaded documents")

    tool_calls = result.get("tool_calls") or []
    planned_tools = [step.get("tool_name", "tool") for step in (result.get("plan") or [])]
    tool_call_names = [tool.get("name", "tool") for tool in tool_calls]

    status_left, status_right, status_third = st.columns(3)
    status_left.metric("Intent", result.get("intent", "unknown"))
    status_right.metric("Confidence", f"{result.get('confidence', 0):.2f}")
    status_third.metric("Executed tools", len(tool_calls))

    if result.get("intent_reason"):
        st.write("**Reasoning**")
        st.write(result["intent_reason"])

    if result.get("clarification_needed"):
        st.warning(result.get("answer", "Please choose a specific document and try again."))
        candidates = result.get("candidate_documents") or []
        if candidates:
            st.write("**Matching documents**")
            for item in candidates:
                doc_id = item.get("document_id")
                file_name = item.get("file_name")
                match_count = item.get("match_count", 0)
                col1, col2 = st.columns([0.72, 0.28])
                with col1:
                    st.write(f"{file_name} ({match_count} matching chunk(s))")
                with col2:
                    if st.button("Use this", key=f"use-doc-{doc_id}", use_container_width=True):
                        st.session_state.chat_document_id = doc_id
                        set_message("info", f"Selected {file_name}. Run the chat again to continue.")
                        st.rerun()
        return

    if planned_tools:
        st.write("**Agent route**")
        st.write(" -> ".join(planned_tools))

    if any(name == "generate_report" for name in tool_call_names):
        st.success("The agent chose report generation for this chat request.")
    elif any(name == "summarize_meeting" for name in tool_call_names):
        st.success("The agent chose meeting summarization for this chat request.")
    elif any(name in {"get_data", "create_ticket", "send_email"} for name in tool_call_names):
        st.success("The agent chose workflow tool execution for this chat request.")
    elif result.get("context"):
        st.info("The agent answered with retrieved document context.")
    else:
        st.info("The agent answered directly without executing tools.")

    st.write("**Final answer**")
    st.code(result.get("answer", ""), language="markdown")

    plan = result.get("plan") or []
    if plan:
        st.write("**Execution plan**")
        for index, step in enumerate(plan, start=1):
            with st.expander(f"Step {index}: {step.get('tool_name', 'tool')}"):
                st.write(step.get("rationale") or "No rationale provided.")
                st.json(step.get("args") or {})

    context = result.get("context") or []
    if context:
        st.write("**Retrieved context**")
        for index, item in enumerate(context, start=1):
            with st.expander(f"Context {index}"):
                st.write(item)

    if tool_calls:
        st.write("**Tool calls**")
        for index, tool in enumerate(tool_calls, start=1):
            with st.expander(f"Tool {index}: {tool.get('name', 'unknown')}"):
                st.json(tool.get("args") or {})
                st.code(tool.get("output", ""), language="text")

        report_calls = [tool for tool in tool_calls if tool.get("name") == "generate_report"]
        if report_calls:
            st.write("**Agent-generated report output**")
            for tool in report_calls:
                st.code(tool.get("output", ""), language="markdown")

        summary_calls = [tool for tool in tool_calls if tool.get("name") == "summarize_meeting"]
        if summary_calls:
            st.write("**Agent-generated summary output**")
            for tool in summary_calls:
                st.code(tool.get("output", ""), language="markdown")


def render_reports() -> None:
    st.markdown("### Report Generator")
    st.caption(
        "This is the direct report endpoint. You can also ask in AI Chat and let the agent choose "
        "`generate_report` automatically."
    )
    with st.form("report_form"):
        title = st.text_input("Title", value="Weekly Operations Review")
        prompt = st.text_area(
            "Prompt",
            value="Create an executive-ready weekly operations report covering wins, blockers, and recommended next steps.",
            height=140,
        )
        submitted = st.form_submit_button("Create report", use_container_width=True)
    if submitted:
        try:
            st.session_state.report_result = api_request(
                "/api/v1/generate-report",
                method="POST",
                json_body={"title": title, "prompt": prompt},
            )
            set_message("success", "Report generated.")
        except Exception as exc:
            set_message("error", str(exc))

    if st.session_state.report_result:
        result = st.session_state.report_result
        st.write(f"**Title:** {result.get('title')}")
        st.write(f"**Created:** {result.get('created_at')}")
        st.code(result.get("content", ""), language="markdown")


def render_summary() -> None:
    st.markdown("### Meeting Summary")
    st.caption(
        "This is the direct summary endpoint. You can also paste notes into AI Chat and let the agent choose "
        "`summarize_meeting` automatically."
    )
    with st.form("summary_form"):
        notes = st.text_area(
            "Meeting notes",
            value=(
                "We reviewed Q2 hiring goals, flagged vendor delays in onboarding, agreed to pause contractor spend, "
                "and assigned finance to confirm revised timelines by Friday."
            ),
            height=180,
        )
        submitted = st.form_submit_button("Create summary", use_container_width=True)
    if submitted:
        try:
            st.session_state.summary_result = api_request(
                "/api/v1/summarize",
                method="POST",
                json_body={"meeting_notes": notes},
            )
            set_message("success", "Summary created.")
        except Exception as exc:
            set_message("error", str(exc))

    if st.session_state.summary_result:
        st.code(st.session_state.summary_result.get("summary", ""), language="markdown")


def render_workflow() -> None:
    st.markdown("### Workflow Automation")
    st.caption(
        "This runs the direct workflow endpoint. In AI Chat, the agent can now also decide to execute "
        "`get_data`, `create_ticket`, and `send_email` as part of its own plan."
    )
    with st.form("workflow_form"):
        objective = st.text_area(
            "Objective",
            value="Investigate a failed payroll sync, open an operations ticket, and notify stakeholders.",
            height=140,
        )
        recipient_email = st.text_input("Recipient email", value="ops@example.com")
        submitted = st.form_submit_button("Run workflow", use_container_width=True)
    if submitted:
        try:
            st.session_state.workflow_result = api_request(
                "/api/v1/workflow",
                method="POST",
                json_body={"objective": objective, "recipient_email": recipient_email},
            )
            set_message("success", "Workflow executed.")
        except Exception as exc:
            set_message("error", str(exc))

    result = st.session_state.workflow_result
    if not result:
        st.info("Workflow steps will appear here.")
        return

    st.write(f"**Status:** {result.get('status')}")
    st.write(f"**Objective:** {result.get('objective')}")
    for index, step in enumerate(result.get("steps") or [], start=1):
        with st.expander(f"Step {index}: {step.get('name', 'tool')}"):
            st.json(step.get("args") or {})
            st.code(step.get("output", ""), language="text")


def render_documents() -> None:
    st.markdown("### Documents")
    uploaded_file = st.file_uploader("Upload a text-like document", type=None)
    if st.button("Upload document", use_container_width=True, disabled=uploaded_file is None):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
            api_request("/api/v1/upload-doc", method="POST", files=files)
            refresh_documents()
            set_message("success", "Document uploaded successfully.")
        except Exception as exc:
            set_message("error", str(exc))

    documents = st.session_state.documents
    if not documents:
        st.info("No uploaded documents yet.")
        return
    for item in documents:
        with st.container(border=True):
            st.write(f"**{item.get('file_name', 'document')}**")
            st.caption(f"Created: {item.get('created_at')}")


def main() -> None:
    st.set_page_config(
        page_title="Internal AI Agents",
        page_icon="IA",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    if st.session_state.health is None:
        refresh_health()
    if st.session_state.current_user is None:
        hydrate_session()

    st.title("Internal AI Agents")
    st.caption("Streamlit command center for the FastAPI, LangGraph, and RAG backend.")

    render_sidebar()
    show_message()

    # Every workspace panel calls a protected endpoint, so there is nothing
    # meaningful to render until the visitor has signed in.
    if not st.session_state.current_user:
        left, _ = st.columns([0.5, 0.5], gap="large")
        with left:
            render_workspace_status()
        return

    render_overview()

    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        render_workspace_status()
        st.divider()
        render_documents()
    with right:
        render_chat()
        with st.expander("Direct Endpoints", expanded=False):
            st.caption("Optional single-purpose tools. AI Chat can also trigger these capabilities automatically.")
            top_row_left, top_row_right = st.columns(2, gap="large")
            with top_row_left:
                render_reports()
            with top_row_right:
                render_summary()
            st.divider()
            render_workflow()


if __name__ == "__main__":
    main()
