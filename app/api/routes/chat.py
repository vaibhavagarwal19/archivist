from functools import lru_cache

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.langgraph_agent import LangGraphAgentService

router = APIRouter(prefix="/chat")


@lru_cache(maxsize=1)
def get_agent_service() -> LangGraphAgentService:
    """Build the agent once per process.

    Constructing it compiles the LangGraph, instantiates the chat model and
    loads the embedding model, so doing it per request added avoidable latency
    to every message. The service holds no per-request state - the caller's
    identity is passed into run() - so a single instance is safe to share.
    """
    return LangGraphAgentService()


@router.post("", response_model=ChatResponse)
async def chat_with_agent(
    payload: ChatRequest, current_user: User = Depends(get_current_user)
) -> ChatResponse:
    service = get_agent_service()
    return await service.run(
        payload.query,
        use_rag=payload.use_rag,
        user_id=current_user.id,
        document_id=payload.document_id,
    )
