from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.db.session import SessionLocal
from app.models.user import User
from app.schemas.chat import CandidateDocument, ChatResponse, PlannedStep, ToolCallRecord
from app.services.internal_action_service import InternalActionService
from app.services.langchain_factory import get_langchain_chat_model
from app.services.report_service import ReportService
from app.services.summary_service import SummaryService
from app.services.vector_store import VectorStoreService

IntentType = Literal[
    "greeting",
    "summarization",
    "reporting",
    "workflow",
    "knowledge_lookup",
    "general_chat",
]


class AgentState(BaseModel):
    query: str
    user_id: int | None = None
    use_rag: bool = True
    document_id: int | None = None
    intent: IntentType = "general_chat"
    confidence: float = 0.0
    intent_reason: str = ""
    should_retrieve: bool = False
    clarification_needed: bool = False
    candidate_documents: list[CandidateDocument] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    document_brief: str = ""
    plan: list[PlannedStep] = Field(default_factory=list)
    answer: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class IntentClassification(BaseModel):
    intent: IntentType
    reason: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RetrievalDecision(BaseModel):
    should_retrieve: bool
    reason: str = Field(min_length=1)


class FollowUpDecision(BaseModel):
    follow_up_needed: bool
    question: str = ""
    reason: str = Field(min_length=1)


ToolName = Literal[
    "get_data",
    "create_ticket",
    "send_email",
    "generate_report",
    "summarize_meeting",
]


class PlannedToolStep(BaseModel):
    tool_name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ActionPlan(BaseModel):
    should_execute_tools: bool
    reasoning: str = Field(min_length=1)
    steps: list[PlannedToolStep] = Field(default_factory=list)


class DocumentUnderstanding(BaseModel):
    relevant_sections: list[str] = Field(default_factory=list)
    compressed_context: str = ""
    reasoning: str = Field(min_length=1)


class LangGraphAgentService:
    def __init__(self) -> None:
        self.model = get_langchain_chat_model(temperature=0)
        self.vector_store = VectorStoreService()
        self.action_service = InternalActionService()
        self.report_service = ReportService()
        self.summary_service = SummaryService()
        self.graph = self._build_graph()

    async def _invoke_json_model(
        self,
        schema: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
    ) -> BaseModel:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        f"{system_prompt}\n\n"
                        "Return valid JSON only. Do not add markdown fences, commentary, or any extra text."
                    ),
                ),
                ("human", user_prompt),
            ]
        )
        chain = prompt | self.model
        response = await chain.ainvoke({})
        content = getattr(response, "content", str(response)).strip()
        if content.startswith("```"):
            content = content.strip("`")
            if "\n" in content:
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return JSON: {content}")
        return schema.model_validate(json.loads(content[start : end + 1]))

    @staticmethod
    def _heuristic_intent(query: str) -> IntentType:
        normalized = query.strip().lower()
        greeting_terms = {"hi", "hello", "hey", "hello there", "good morning", "good afternoon", "good evening"}
        if normalized in greeting_terms:
            return "greeting"
        if any(term in normalized for term in ["summarize", "summary", "meeting notes", "minutes of meeting"]):
            return "summarization"
        if any(term in normalized for term in ["report", "status update", "weekly update", "generate report"]):
            return "reporting"
        if any(
            term in normalized
            for term in ["send email", "notify", "create ticket", "workflow", "automate", "escalate", "follow up"]
        ):
            return "workflow"
        if any(
            term in normalized
            for term in [
                "policy",
                "document",
                "knowledge base",
                "handbook",
                "uploaded file",
                "uploaded document",
                "what does",
            ]
        ):
            return "knowledge_lookup"
        return "general_chat"

    @staticmethod
    def _heuristic_should_retrieve(query: str, intent: IntentType) -> bool:
        normalized = query.strip().lower()
        if intent in {"knowledge_lookup", "reporting", "summarization"}:
            return True
        if intent == "workflow":
            return False
        return any(
            term in normalized
            for term in [
                "document",
                "uploaded",
                "handbook",
                "policy",
                "knowledge",
                "what should we do",
                "what does",
                "based on",
            ]
        )

    @staticmethod
    def _requires_specific_document(query: str) -> bool:
        normalized = query.strip().lower()
        return any(
            phrase in normalized
            for phrase in [
                "uploaded document",
                "uploaded file",
                "that document",
                "this document",
                "the document",
                "from the document",
                "from the uploaded",
            ]
        )

    @staticmethod
    def _context_as_text(context: list[str]) -> str:
        return "\n\n".join(context).strip()

    @classmethod
    def _preferred_content_source(cls, state: AgentState) -> str:
        if state.document_brief.strip():
            return state.document_brief
        context_text = cls._context_as_text(state.context)
        if context_text:
            return context_text
        return state.query

    @classmethod
    def _fallback_plan(cls, state: AgentState) -> ActionPlan:
        normalized = state.query.strip().lower()

        if state.intent == "workflow":
            steps = [
                PlannedToolStep(
                    tool_name="get_data",
                    args={"metric": state.query},
                    rationale="Gather a local internal data snapshot before taking action.",
                )
            ]

            if any(term in normalized for term in ["ticket", "incident", "issue", "escalate"]):
                steps.append(
                    PlannedToolStep(
                        tool_name="create_ticket",
                        args={
                            "team": "operations",
                            "issue": state.query,
                            "priority": "high",
                        },
                        rationale="Create an operations ticket because the request asks for escalation or incident handling.",
                    )
                )

            if any(term in normalized for term in ["email", "notify", "send"]):
                steps.append(
                    PlannedToolStep(
                        tool_name="send_email",
                        args={
                            "recipient": "ops@example.com",
                            "subject": "Internal agent workflow update",
                            "body": f"Workflow request processed: {state.query}",
                        },
                        rationale="Send a notification because the request asks for email or stakeholder updates.",
                    )
                )

            return ActionPlan(
                should_execute_tools=True,
                reasoning="Planner fallback created a workflow action sequence from the request.",
                steps=steps,
            )

        if state.intent == "reporting":
            return ActionPlan(
                should_execute_tools=True,
                reasoning="Planner fallback chose report generation for a reporting request.",
                steps=[
                    PlannedToolStep(
                        tool_name="generate_report",
                        args={
                            "title": "AI Generated Internal Report",
                            "prompt": cls._preferred_content_source(state),
                        },
                        rationale="Create the requested report and return the saved result.",
                    )
                ],
            )

        if state.intent == "summarization":
            return ActionPlan(
                should_execute_tools=True,
                reasoning="Planner fallback chose summarization for a summarization request.",
                steps=[
                    PlannedToolStep(
                        tool_name="summarize_meeting",
                        args={"meeting_notes": cls._preferred_content_source(state)},
                        rationale="Summarize the provided notes or text.",
                    )
                ],
            )

        return ActionPlan(
            should_execute_tools=False,
            reasoning="Planner fallback determined that no tools are needed.",
            steps=[],
        )

    @staticmethod
    def _system_prompt(intent: IntentType) -> str:
        prompts = {
            "greeting": (
                "You are a friendly internal AI assistant. Reply naturally, briefly, and warmly. "
                "Keep the response to one sentence."
            ),
            "summarization": (
                "You are an internal summarization assistant. Summarize clearly and concisely. "
                "Use sections when helpful: overview, key points, decisions, and next steps."
            ),
            "reporting": (
                "You are an internal reporting assistant. Produce professional, business-ready answers "
                "with concise structure and concrete details."
            ),
            "workflow": (
                "You are an internal workflow assistant. Explain the steps taken and outcomes clearly."
            ),
            "knowledge_lookup": (
                "You are an internal knowledge assistant. Prefer retrieved internal context when relevant. "
                "If context is insufficient, say so clearly and answer conservatively."
            ),
            "general_chat": (
                "You are an internal enterprise AI assistant. Answer clearly, directly, and helpfully."
            ),
        }
        return prompts[intent]

    async def _detect_intent_node(self, state: AgentState) -> dict[str, Any]:
        try:
            result = await self._invoke_json_model(
                IntentClassification,
                (
                    "Classify the user's request into exactly one supported intent.\n"
                    "Allowed intents: greeting, summarization, reporting, workflow, knowledge_lookup, general_chat.\n"
                    "Use these rules:\n"
                    "- greeting: simple hello without a real task\n"
                    "- summarization: summarize notes, text, or meeting content\n"
                    "- reporting: create a report, status update, or structured business write-up\n"
                    "- workflow: take actions like notify, email, ticket, escalate, automate\n"
                    "- knowledge_lookup: answer from uploaded/internal documents, policy, handbook, or company knowledge\n"
                    "- general_chat: everything else\n"
                    "Set confidence between 0 and 1."
                ),
                f"User query:\n{state.query}",
            )
            return {
                "intent": result.intent,
                "confidence": result.confidence,
                "intent_reason": result.reason,
            }
        except Exception:
            fallback_intent = self._heuristic_intent(state.query)
            return {
                "intent": fallback_intent,
                "confidence": 0.25,
                "intent_reason": (
                    "Intent classification fallback triggered after the structured classifier failed. "
                    f"Using heuristic intent={fallback_intent}."
                ),
            }

    async def _plan_retrieval_node(self, state: AgentState) -> dict[str, Any]:
        if not state.use_rag:
            return {"should_retrieve": False}

        try:
            decision = await self._invoke_json_model(
                RetrievalDecision,
                (
                    "Decide whether internal document retrieval is needed before answering.\n"
                    "Prefer retrieval for uploaded documents, policy, handbook, meeting notes stored in documents, "
                    "or whenever grounding in internal content would improve the answer."
                ),
                (
                    f"Intent: {state.intent}\n"
                    f"Intent reason: {state.intent_reason}\n"
                    f"User query: {state.query}\n"
                    f"Selected document id: {state.document_id}"
                ),
            )
            return {"should_retrieve": decision.should_retrieve}
        except Exception:
            return {"should_retrieve": self._heuristic_should_retrieve(state.query, state.intent)}

    async def _retrieve_context_node(self, state: AgentState) -> dict[str, Any]:
        if not state.should_retrieve:
            return {"context": []}

        matches = self.vector_store.search_chunks(
            state.query,
            k=4,
            owner_id=state.user_id,
            document_id=state.document_id,
        )
        if not matches:
            return {"context": []}

        grouped: dict[tuple[int, str], int] = {}
        for item in matches:
            key = (item["document_id"], item["file_name"])
            grouped[key] = grouped.get(key, 0) + 1

        if state.document_id is None and self._requires_specific_document(state.query) and len(grouped) > 1:
            candidates = [
                CandidateDocument(document_id=document_id, file_name=file_name, match_count=count)
                for (document_id, file_name), count in sorted(grouped.items(), key=lambda entry: (-entry[1], entry[0][1]))
            ]
            answer = (
                "I found multiple uploaded documents that could match your request. "
                "Please pick one document in the chat selector and try again."
            )
            return {
                "clarification_needed": True,
                "candidate_documents": candidates,
                "answer": answer,
                "context": [],
            }

        context = [
            f"Document: {item['file_name']}\nContent: {item['content']}"
            for item in matches
        ]
        return {"context": context, "candidate_documents": [], "clarification_needed": False}

    async def _clarify_request_node(self, state: AgentState) -> dict[str, Any]:
        if state.clarification_needed:
            return {}

        if state.intent not in {"summarization", "reporting"}:
            return {}

        context_text = self._context_as_text(state.context)
        try:
            decision = await self._invoke_json_model(
                FollowUpDecision,
                (
                    "Decide whether the request needs a follow-up question before continuing.\n"
                    "Use follow_up_needed=true only when the user request is underspecified in a way that materially changes the output.\n"
                    "Examples:\n"
                    "- report request without audience/type such as leadership vs operational\n"
                    "- summary request that does not identify which document to summarize when multiple document sources are in play\n"
                    "If the request is specific enough, set follow_up_needed=false."
                ),
                (
                    f"Intent: {state.intent}\n"
                    f"User query: {state.query}\n"
                    f"Selected document id: {state.document_id}\n"
                    f"Retrieved context present: {'yes' if context_text else 'no'}\n"
                    f"Retrieved context preview:\n{context_text[:1500]}"
                ),
            )
        except Exception:
            normalized = state.query.lower()
            if state.intent == "reporting" and not any(term in normalized for term in ["leadership", "operational", "executive"]):
                decision = FollowUpDecision(
                    follow_up_needed=True,
                    question="Do you want a leadership report or an operational report?",
                    reason="Report audience/type is not specified.",
                )
            else:
                decision = FollowUpDecision(
                    follow_up_needed=False,
                    question="",
                    reason="The request is specific enough to continue.",
                )

        if decision.follow_up_needed:
            return {
                "clarification_needed": True,
                "answer": decision.question,
                "intent_reason": f"{state.intent_reason} Follow-up: {decision.reason}",
            }
        return {}

    async def _understand_document_node(self, state: AgentState) -> dict[str, Any]:
        if state.intent not in {"summarization", "reporting"}:
            return {}

        context_text = self._context_as_text(state.context)
        if not context_text:
            return {}

        try:
            understanding = await self._invoke_json_model(
                DocumentUnderstanding,
                (
                    "Analyze the retrieved document context and extract only the most relevant sections for the user's task.\n"
                    "Return a short list of relevant sections and a compressed_context string that keeps only the essential facts."
                ),
                (
                    f"User query: {state.query}\n\n"
                    f"Retrieved context:\n{context_text}"
                ),
            )
            return {"document_brief": understanding.compressed_context}
        except Exception:
            return {"document_brief": context_text[:3000]}

    async def _plan_actions_node(self, state: AgentState) -> dict[str, Any]:
        try:
            plan = await self._invoke_json_model(
                ActionPlan,
                (
                    "Plan internal assistant actions.\n"
                    "Available tools:\n"
                    "- get_data\n"
                    "- create_ticket\n"
                    "- send_email\n"
                    "- generate_report\n"
                    "- summarize_meeting\n"
                    "Return should_execute_tools=false if a normal answer is enough.\n"
                    "When tools are needed, return a short ordered list of steps with practical arguments."
                ),
                (
                    f"Intent: {state.intent}\n"
                    f"Intent confidence: {state.confidence}\n"
                    f"Intent reason: {state.intent_reason}\n"
                    f"User query: {state.query}\n"
                    f"Selected document id: {state.document_id}\n"
                    f"Retrieved context count: {len(state.context)}\n"
                    f"Document brief:\n{state.document_brief[:1500]}"
                ),
            )
        except Exception:
            plan = self._fallback_plan(state)

        return {
            "plan": (
                [
                    PlannedStep(tool_name=step.tool_name, args=step.args, rationale=step.rationale)
                    for step in plan.steps
                ]
                if plan.should_execute_tools
                else []
            ),
            "intent_reason": state.intent_reason
            if not plan.reasoning
            else f"{state.intent_reason} Planner: {plan.reasoning}",
        }

    async def _execute_tool_step(self, state: AgentState, step: PlannedStep) -> ToolCallRecord:
        if state.user_id is None:
            raise ValueError("A user id is required to execute workflow tools.")

        with SessionLocal() as db:
            owner = db.get(User, state.user_id)
            if owner is None:
                raise ValueError(f"User {state.user_id} not found.")

            if step.tool_name == "get_data":
                metric = str(step.args.get("metric") or state.query)
                return self.action_service.get_data(db, owner, metric)
            if step.tool_name == "create_ticket":
                team = str(step.args.get("team") or "operations")
                issue = str(step.args.get("issue") or state.query)
                priority = str(step.args.get("priority") or "high")
                return self.action_service.create_ticket(db, owner, team, issue, priority)
            if step.tool_name == "send_email":
                recipient = str(step.args.get("recipient") or "ops@example.com")
                subject = str(step.args.get("subject") or "Internal agent workflow update")
                body = str(step.args.get("body") or f"Workflow request processed: {state.query}")
                return self.action_service.send_email(db, owner, recipient, subject, body)
            if step.tool_name == "generate_report":
                title = str(step.args.get("title") or "AI Generated Internal Report")
                prompt = str(step.args.get("prompt") or state.query)
                report = await self.report_service.generate_report(db, owner, title, prompt)
                return ToolCallRecord(
                    name="generate_report",
                    args={"title": title, "prompt": prompt},
                    output=f"Report #{report.id} generated with title '{report.title}'.\n\n{report.content}",
                )
            if step.tool_name == "summarize_meeting":
                meeting_notes = str(step.args.get("meeting_notes") or state.query)
                summary = await self.summary_service.summarize_meeting(meeting_notes)
                return ToolCallRecord(
                    name="summarize_meeting",
                    args={"meeting_notes": meeting_notes},
                    output=summary,
                )

        raise ValueError(f"Unsupported tool name: {step.tool_name}")

    async def _execute_actions_node(self, state: AgentState) -> dict[str, Any]:
        if not state.plan:
            return {"tool_calls": []}

        tool_calls: list[ToolCallRecord] = []
        execution_errors: list[str] = []

        for step in state.plan:
            try:
                tool_calls.append(await self._execute_tool_step(state, step))
            except Exception as exc:
                execution_errors.append(f"{step.tool_name}: {exc}")

        answer_lines = [f"Executed {len(tool_calls)} planned workflow step(s)."]
        answer_lines.extend(f"- {record.output}" for record in tool_calls)
        if execution_errors:
            answer_lines.append("Some planned steps could not be completed:")
            answer_lines.extend(f"- {message}" for message in execution_errors)

        return {"tool_calls": tool_calls, "answer": "\n".join(answer_lines)}

    async def _respond_node(self, state: AgentState) -> dict[str, Any]:
        if state.answer and (state.intent == "workflow" or state.clarification_needed):
            return {}

        context_block = state.document_brief or ("\n\n".join(state.context) if state.context else "No internal context retrieved.")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt(state.intent)),
                (
                    "human",
                    "User query:\n{query}\n\nRetrieved context:\n{context}\n\n"
                    "Answer the user directly. Do not invent tool actions that were not executed.",
                ),
            ]
        )
        chain = prompt | self.model
        response = await chain.ainvoke({"query": state.query, "context": context_block})
        return {"answer": getattr(response, "content", str(response))}

    @staticmethod
    def _route_after_intent(state: AgentState) -> str:
        return "plan_retrieval"

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        if state.clarification_needed:
            return "respond"
        return "clarify_request"

    @staticmethod
    def _route_after_clarify(state: AgentState) -> str:
        if state.clarification_needed:
            return "respond"
        if state.intent in {"summarization", "reporting"}:
            return "understand_document"
        return "plan_actions"

    @staticmethod
    def _route_after_understanding(_: AgentState) -> str:
        return "plan_actions"

    @staticmethod
    def _route_after_action_plan(state: AgentState) -> str:
        if state.plan:
            return "execute_actions"
        return "respond"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("detect_intent", self._detect_intent_node)
        graph.add_node("plan_retrieval", self._plan_retrieval_node)
        graph.add_node("retrieve", self._retrieve_context_node)
        graph.add_node("clarify_request", self._clarify_request_node)
        graph.add_node("understand_document", self._understand_document_node)
        graph.add_node("plan_actions", self._plan_actions_node)
        graph.add_node("execute_actions", self._execute_actions_node)
        graph.add_node("respond", self._respond_node)

        graph.set_entry_point("detect_intent")
        graph.add_conditional_edges(
            "detect_intent",
            self._route_after_intent,
            {
                "plan_retrieval": "plan_retrieval",
            },
        )
        graph.add_conditional_edges(
            "plan_retrieval",
            lambda state: "retrieve" if state.should_retrieve else "plan_actions",
            {
                "retrieve": "retrieve",
                "plan_actions": "plan_actions",
            },
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieval,
            {
                "clarify_request": "clarify_request",
                "respond": "respond",
            },
        )
        graph.add_conditional_edges(
            "clarify_request",
            self._route_after_clarify,
            {
                "respond": "respond",
                "understand_document": "understand_document",
                "plan_actions": "plan_actions",
            },
        )
        graph.add_conditional_edges(
            "understand_document",
            self._route_after_understanding,
            {
                "plan_actions": "plan_actions",
            },
        )
        graph.add_conditional_edges(
            "plan_actions",
            self._route_after_action_plan,
            {
                "execute_actions": "execute_actions",
                "respond": "respond",
            },
        )
        graph.add_edge("execute_actions", END)
        graph.add_edge("respond", END)
        return graph.compile()

    async def run(
        self,
        query: str,
        use_rag: bool = True,
        user_id: int | None = None,
        document_id: int | None = None,
    ) -> ChatResponse:
        result = await self.graph.ainvoke(
            AgentState(query=query, use_rag=use_rag, user_id=user_id, document_id=document_id)
        )
        if isinstance(result, AgentState):
            state = result
        else:
            state = AgentState.model_validate(result)

        return ChatResponse(
            intent=state.intent,
            confidence=state.confidence,
            intent_reason=state.intent_reason,
            answer=state.answer,
            clarification_needed=state.clarification_needed,
            candidate_documents=state.candidate_documents,
            context=state.context,
            plan=state.plan,
            tool_calls=state.tool_calls,
        )
