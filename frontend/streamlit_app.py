"""Streamlit console for the Internal AI Agents API.

Chat is the primary surface: the agent decides whether to retrieve, which
tools to run, and what to answer, and every one of those decisions is shown
inline under the reply rather than hidden in a debug panel.
"""
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


DEFAULT_API_BASE = _resolve_api_base()

INTENT_LABELS = {
    "greeting": "Greeting",
    "summarization": "Summarization",
    "reporting": "Reporting",
    "workflow": "Workflow",
    "knowledge_lookup": "Knowledge lookup",
    "general_chat": "General",
}

TOOL_LABELS = {
    "get_data": "Look up internal data",
    "create_ticket": "Create ticket",
    "send_email": "Send email",
    "generate_report": "Generate report",
    "summarize_meeting": "Summarize notes",
}

EXAMPLE_PROMPTS = [
    "What does the ops playbook say about payroll sync?",
    "Summarize the strategy notes into decisions and next steps",
    "Draft a weekly operations report on hiring blockers",
    "Investigate the payroll incident, open a ticket, and email ops",
]

STYLES = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1100px;}
  [data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.07);}

  .brand {font-size: 1.05rem; font-weight: 700; letter-spacing: -.01em;}
  .brand-sub {color: #8b90a3; font-size: .78rem; margin-top: -.2rem;}

  .pill {display:inline-block; padding: .12rem .55rem; border-radius: 999px;
         font-size: .72rem; font-weight: 600; margin-right: .35rem;
         border: 1px solid rgba(255,255,255,.12); color: #c8cce0;}
  .pill-accent {background: rgba(109,124,255,.16); border-color: rgba(109,124,255,.4); color: #b3bcff;}
  .pill-ok {background: rgba(46,204,113,.13); border-color: rgba(46,204,113,.35); color: #86e0ab;}
  .pill-warn {background: rgba(241,196,15,.13); border-color: rgba(241,196,15,.35); color: #f0d180;}

  .doc-row {display:flex; justify-content:space-between; align-items:center;
            padding:.55rem .8rem; border:1px solid rgba(255,255,255,.08);
            border-radius:10px; margin-bottom:.4rem; background: rgba(255,255,255,.02);}
  .doc-name {font-weight:600; font-size:.9rem;}
  .doc-meta {color:#8b90a3; font-size:.75rem;}

  .step {padding:.45rem 0 .45rem .9rem; border-left:2px solid rgba(109,124,255,.45); margin-bottom:.2rem;}
  .step-name {font-weight:600; font-size:.85rem;}
  .step-why {color:#9aa0b5; font-size:.78rem;}

  .src {padding:.5rem .7rem; border-radius:8px; background:rgba(255,255,255,.03);
        border:1px solid rgba(255,255,255,.07); margin-bottom:.35rem; font-size:.82rem;}
  .src-name {color:#b3bcff; font-weight:600; font-size:.78rem;}

  .empty {text-align:center; color:#8b90a3; padding:2.5rem 1rem;}
</style>
"""


# --------------------------------------------------------------------------
# State and transport
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "api_base": DEFAULT_API_BASE,
        "access_token": None,
        "current_user": None,
        "message": None,
        "health": None,
        "documents": [],
        "messages": [],
        "page": "Chat",
        "chat_use_rag": True,
        "chat_document_id": None,
        "pending_prompt": None,
        "report_result": None,
        "summary_result": None,
        "workflow_result": None,
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

    if response.status_code == 204 or not response.content:
        return None
    if "application/json" in response.headers.get("content-type", ""):
        return response.json()
    return response.text


def set_message(message_type: str, text: str) -> None:
    st.session_state.message = {"type": message_type, "text": text}


def show_message() -> None:
    message = st.session_state.message
    if not message:
        return
    {"success": st.success, "error": st.error}.get(message["type"], st.info)(message["text"])
    st.session_state.message = None


def format_timestamp(raw: str | None) -> str:
    """Turn an ISO timestamp into something a person can read."""
    if not raw:
        return "unknown"
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return raw


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------
def refresh_health() -> None:
    try:
        data = api_request("/health")
        st.session_state.health = {"status": data.get("status", "ok"), "ok": True}
    except Exception as exc:
        st.session_state.health = {"status": str(exc)[:60], "ok": False}


def refresh_documents() -> None:
    st.session_state.documents = api_request("/api/v1/upload-doc").get("documents", [])


def hydrate_session() -> None:
    if not st.session_state.get("access_token"):
        st.session_state.current_user = None
        st.session_state.documents = []
        return
    try:
        st.session_state.current_user = api_request("/api/v1/auth/me")
        refresh_documents()
    except Exception as exc:
        st.session_state.access_token = None
        st.session_state.current_user = None
        st.session_state.documents = []
        set_message("error", f"Session ended, please sign in again: {exc}")


def sign_in(email: str, password: str) -> None:
    token = api_request(
        "/api/v1/auth/login", method="POST", json_body={"email": email, "password": password}
    )
    st.session_state.access_token = token["access_token"]
    hydrate_session()


def sign_up(email: str, full_name: str, password: str) -> None:
    api_request(
        "/api/v1/auth/register",
        method="POST",
        json_body={"email": email, "full_name": full_name, "password": password},
    )
    sign_in(email, password)


def sign_out() -> None:
    for key in (
        "access_token",
        "current_user",
        "report_result",
        "summary_result",
        "workflow_result",
    ):
        st.session_state[key] = None
    st.session_state.documents = []
    st.session_state.messages = []


# --------------------------------------------------------------------------
# Sign-in screen
# --------------------------------------------------------------------------
def render_auth_screen() -> None:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("### Internal AI Agents")
        st.caption(
            "A retrieval-backed agent for internal documents. Sign in to upload "
            "documents and query them; each account sees only its own."
        )
        show_message()

        sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])
        with sign_in_tab:
            with st.form("sign_in_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", use_container_width=True, type="primary"):
                    try:
                        sign_in(email, password)
                        st.rerun()
                    except Exception as exc:
                        set_message("error", str(exc))
                        st.rerun()

        with register_tab:
            with st.form("sign_up_form"):
                new_email = st.text_input("Email", key="reg_email")
                new_name = st.text_input("Full name", key="reg_name")
                new_password = st.text_input(
                    "Password", type="password", key="reg_pw", help="At least 8 characters."
                )
                if st.form_submit_button("Create account", use_container_width=True, type="primary"):
                    try:
                        sign_up(new_email, new_name, new_password)
                        st.rerun()
                    except Exception as exc:
                        set_message("error", str(exc))
                        st.rerun()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand">Internal AI Agents</div>'
            '<div class="brand-sub">LangGraph agent · RAG · FastAPI</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        st.session_state.page = st.radio(
            "Navigation",
            ["Chat", "Documents", "Direct tools"],
            label_visibility="collapsed",
        )

        st.divider()
        user = st.session_state.current_user or {}
        st.caption("Signed in as")
        st.markdown(f"**{user.get('full_name') or user.get('email', 'unknown')}**")
        st.caption(user.get("email", ""))

        if st.button("Sign out", use_container_width=True):
            sign_out()
            st.rerun()

        st.divider()
        health = st.session_state.health or {"status": "not checked", "ok": False}
        dot = "🟢" if health.get("ok") else "🔴"
        st.caption(f"{dot} API {health.get('status')}")
        with st.expander("Connection"):
            api_base = st.text_input("API base URL", value=st.session_state.api_base)
            if api_base != st.session_state.api_base:
                st.session_state.api_base = api_base
            if st.button("Re-check", use_container_width=True):
                refresh_health()
                st.rerun()


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def parse_context_item(raw: str) -> tuple[str, str]:
    """Split a retrieved chunk into (document name, text)."""
    name, _, rest = raw.partition("\n")
    if name.startswith("Document: "):
        return name.removeprefix("Document: ").strip(), rest.removeprefix("Content: ").strip()
    return "source", raw


def render_trace(result: dict[str, Any]) -> None:
    """Show how the agent reached its answer."""
    intent = result.get("intent", "general_chat")
    confidence = result.get("confidence") or 0.0
    context = result.get("context") or []
    plan = result.get("plan") or []
    tool_calls = result.get("tool_calls") or []

    summary_bits = [INTENT_LABELS.get(intent, intent)]
    if context:
        summary_bits.append(f"{len(context)} source{'s' if len(context) > 1 else ''}")
    if tool_calls:
        summary_bits.append(f"{len(tool_calls)} tool{'s' if len(tool_calls) > 1 else ''}")

    with st.expander("How I answered  ·  " + "  ·  ".join(summary_bits)):
        pills = [
            f'<span class="pill pill-accent">{INTENT_LABELS.get(intent, intent)}</span>',
            f'<span class="pill">confidence {confidence:.0%}</span>',
        ]
        pills.append(
            '<span class="pill pill-ok">retrieval used</span>'
            if context
            else '<span class="pill">no retrieval</span>'
        )
        st.markdown("".join(pills), unsafe_allow_html=True)

        if result.get("intent_reason"):
            st.caption(result["intent_reason"])

        if plan:
            st.markdown("**Planned steps**")
            for step in plan:
                name = TOOL_LABELS.get(step.get("tool_name", ""), step.get("tool_name", "tool"))
                why = step.get("rationale") or ""
                st.markdown(
                    f'<div class="step"><div class="step-name">{name}</div>'
                    f'<div class="step-why">{why}</div></div>',
                    unsafe_allow_html=True,
                )

        if context:
            st.markdown("**Sources**")
            for item in context:
                doc_name, text = parse_context_item(item)
                snippet = text[:280] + ("…" if len(text) > 280 else "")
                st.markdown(
                    f'<div class="src"><div class="src-name">{doc_name}</div>{snippet}</div>',
                    unsafe_allow_html=True,
                )

        if tool_calls:
            # Rendered inline rather than in per-call expanders: Streamlit does
            # not allow an expander inside another expander.
            st.markdown("**Executed**")
            for call in tool_calls:
                label = TOOL_LABELS.get(call.get("name", ""), call.get("name", "tool"))
                args = call.get("args") or {}
                arg_summary = ", ".join(f"{k}={v}" for k, v in list(args.items())[:3])
                st.markdown(
                    f'<div class="step"><div class="step-name">{label}</div>'
                    f'<div class="step-why">{arg_summary}</div></div>',
                    unsafe_allow_html=True,
                )
                output = call.get("output", "")
                if output:
                    st.caption(output if len(output) < 400 else output[:400] + "…")


def render_clarification(result: dict[str, Any], key_prefix: str) -> None:
    """The agent asked which document was meant; offer the candidates."""
    st.warning(result.get("answer") or "Which document did you mean?")
    for item in result.get("candidate_documents") or []:
        doc_id = item.get("document_id")
        left, right = st.columns([0.75, 0.25])
        left.markdown(
            f"**{item.get('file_name')}**  \n"
            f"<span class='doc-meta'>{item.get('match_count', 0)} matching chunk(s)</span>",
            unsafe_allow_html=True,
        )
        if right.button("Use this", key=f"{key_prefix}-{doc_id}", use_container_width=True):
            st.session_state.chat_document_id = doc_id
            set_message("info", f"Scoped to {item.get('file_name')}. Ask again to continue.")
            st.rerun()


def send_chat(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        result = api_request(
            "/api/v1/chat",
            method="POST",
            json_body={
                "query": prompt,
                "use_rag": st.session_state.chat_use_rag,
                "document_id": st.session_state.chat_document_id,
            },
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": result.get("answer", ""), "result": result}
        )
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Request failed: {exc}", "result": None}
        )


def render_chat_page() -> None:
    docs = st.session_state.documents
    options = [("All documents", None)] + [(d["file_name"], d["id"]) for d in docs]
    current = st.session_state.chat_document_id
    index = next((i for i, (_, v) in enumerate(options) if v == current), 0)

    left, right = st.columns([0.62, 0.38])
    with left:
        choice = st.selectbox(
            "Scope",
            range(len(options)),
            index=index,
            format_func=lambda i: options[i][0],
            label_visibility="collapsed",
        )
        st.session_state.chat_document_id = options[choice][1]
    with right:
        st.session_state.chat_use_rag = st.toggle(
            "Search documents", value=st.session_state.chat_use_rag
        )

    if not st.session_state.messages:
        st.markdown(
            '<div class="empty">Ask the agent about your documents, or have it '
            "generate a report, summarize notes, or run a workflow.</div>",
            unsafe_allow_html=True,
        )
        if not docs:
            st.info("You have no documents yet. Upload one from the Documents tab for grounded answers.")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_PROMPTS):
            if cols[i % 2].button(example, key=f"ex-{i}", use_container_width=True):
                st.session_state.pending_prompt = example
                st.rerun()

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            result = message.get("result")
            if result and result.get("clarification_needed"):
                render_clarification(result, key_prefix=f"clar-{i}")
                continue
            # Rendered as markdown so headings, lists and bold in the model's
            # output display properly instead of as literal source text.
            st.markdown(message["content"])
            if result:
                render_trace(result)

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    prompt = st.chat_input("Ask anything about your internal documents")
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    if prompt:
        send_chat(prompt)
        st.rerun()


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------
def render_documents_page() -> None:
    st.markdown("#### Documents")
    st.caption("Uploaded text is chunked, embedded, and searchable by the agent.")

    with st.form("upload_form", clear_on_submit=True):
        upload = st.file_uploader("Add a document", type=["txt", "md", "csv", "json", "log"])
        if st.form_submit_button("Upload", type="primary") and upload is not None:
            try:
                api_request(
                    "/api/v1/upload-doc",
                    method="POST",
                    files={"file": (upload.name, upload.getvalue(), upload.type or "text/plain")},
                )
                refresh_documents()
                set_message("success", f"Uploaded {upload.name}.")
            except Exception as exc:
                set_message("error", str(exc))
            st.rerun()

    docs = st.session_state.documents
    if not docs:
        st.markdown('<div class="empty">No documents yet.</div>', unsafe_allow_html=True)
        return

    st.caption(f"{len(docs)} document(s)")
    for doc in docs:
        left, right = st.columns([0.86, 0.14])
        left.markdown(
            f'<div class="doc-row"><div><div class="doc-name">{doc["file_name"]}</div>'
            f'<div class="doc-meta">Added {format_timestamp(doc.get("created_at"))}</div></div></div>',
            unsafe_allow_html=True,
        )
        if right.button("Delete", key=f"del-{doc['id']}", use_container_width=True):
            try:
                api_request(f"/api/v1/upload-doc/{doc['id']}", method="DELETE")
                refresh_documents()
                set_message("success", f"Deleted {doc['file_name']}.")
            except Exception as exc:
                set_message("error", str(exc))
            st.rerun()


# --------------------------------------------------------------------------
# Direct tools
# --------------------------------------------------------------------------
def render_tools_page() -> None:
    st.markdown("#### Direct tools")
    st.caption(
        "These call single endpoints directly. Chat can invoke the same capabilities "
        "on its own when the request calls for them."
    )

    report_tab, summary_tab, workflow_tab = st.tabs(["Report", "Summary", "Workflow"])

    with report_tab:
        with st.form("report_form"):
            title = st.text_input("Title", value="Weekly Operations Report")
            prompt = st.text_area("What should it cover?", height=120)
            if st.form_submit_button("Generate", type="primary"):
                try:
                    st.session_state.report_result = api_request(
                        "/api/v1/generate-report",
                        method="POST",
                        json_body={"title": title, "prompt": prompt},
                    )
                except Exception as exc:
                    set_message("error", str(exc))
                    st.rerun()
        if st.session_state.report_result:
            st.markdown(st.session_state.report_result.get("content", ""))

    with summary_tab:
        with st.form("summary_form"):
            notes = st.text_area("Meeting notes", height=160)
            if st.form_submit_button("Summarize", type="primary"):
                try:
                    st.session_state.summary_result = api_request(
                        "/api/v1/summarize", method="POST", json_body={"meeting_notes": notes}
                    )
                except Exception as exc:
                    set_message("error", str(exc))
                    st.rerun()
        if st.session_state.summary_result:
            st.markdown(st.session_state.summary_result.get("summary", ""))

    with workflow_tab:
        with st.form("workflow_form"):
            objective = st.text_input("Objective", value="Investigate the payroll sync failure")
            recipient = st.text_input("Notify", value="ops@example.com")
            if st.form_submit_button("Run workflow", type="primary"):
                try:
                    st.session_state.workflow_result = api_request(
                        "/api/v1/workflow",
                        method="POST",
                        json_body={"objective": objective, "recipient_email": recipient},
                    )
                except Exception as exc:
                    set_message("error", str(exc))
                    st.rerun()
        result = st.session_state.workflow_result
        if result:
            st.caption(f"Status: {result.get('status')}")
            for step in result.get("steps", []):
                with st.expander(TOOL_LABELS.get(step.get("name", ""), step.get("name", "step"))):
                    st.json(step.get("args") or {})
                    st.write(step.get("output", ""))


# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Internal AI Agents",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(STYLES, unsafe_allow_html=True)
    init_state()

    if st.session_state.health is None:
        refresh_health()
    if st.session_state.access_token and st.session_state.current_user is None:
        hydrate_session()

    if not st.session_state.current_user:
        render_auth_screen()
        return

    render_sidebar()
    show_message()

    page = st.session_state.page
    if page == "Chat":
        render_chat_page()
    elif page == "Documents":
        render_documents_page()
    else:
        render_tools_page()


if __name__ == "__main__":
    main()
