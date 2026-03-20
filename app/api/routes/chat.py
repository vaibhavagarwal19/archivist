from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.langgraph_agent import LangGraphAgentService


router = APIRouter(prefix="/chat")


@router.post("", response_model=ChatResponse)
async def chat_with_agent(
    payload: ChatRequest, current_user: User = Depends(get_current_user)
) -> ChatResponse:
    service = LangGraphAgentService()
    return await service.run(
        payload.query,
        use_rag=payload.use_rag,
        user_id=current_user.id,
        document_id=payload.document_id,
    )
