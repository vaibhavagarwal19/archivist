from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    use_rag: bool = True
    document_id: int | None = None
    # Continues an existing conversation; omit to start a new one.
    thread_id: str | None = None


class CandidateDocument(BaseModel):
    document_id: int
    file_name: str
    match_count: int = 0


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    output: str


class PlannedStep(BaseModel):
    tool_name: str
    args: dict = Field(default_factory=dict)
    rationale: str = ""


class ChatResponse(BaseModel):
    thread_id: str = ""
    intent: str
    confidence: float = 0.0
    intent_reason: str = ""
    answer: str
    clarification_needed: bool = False
    candidate_documents: list[CandidateDocument] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    plan: list[PlannedStep] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # Prompt-injection categories detected in retrieved content.
    injection_flags: list[str] = Field(default_factory=list)
    # Token, cost and latency breakdown for this request.
    usage: dict = Field(default_factory=dict)
