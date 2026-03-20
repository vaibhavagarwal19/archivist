from datetime import datetime

from pydantic import BaseModel

from app.schemas.chat import ToolCallRecord


class ReportRequest(BaseModel):
    title: str
    prompt: str


class ReportResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SummaryRequest(BaseModel):
    meeting_notes: str


class SummaryResponse(BaseModel):
    summary: str


class WorkflowRequest(BaseModel):
    objective: str
    recipient_email: str = "ops@example.com"


class WorkflowResponse(BaseModel):
    status: str
    objective: str
    steps: list[ToolCallRecord]
