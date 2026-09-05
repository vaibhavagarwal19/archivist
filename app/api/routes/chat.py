from functools import lru_cache

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.conversation_service import ConversationService, new_thread_id
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
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    conversations = ConversationService()
    thread_id = payload.thread_id or new_thread_id()

    # History is loaded server-side and scoped to the caller, so a client
    # cannot inject turns it never had by editing the request body.
    history = conversations.recent_messages(db, current_user, thread_id)

    response = await get_agent_service().run(
        payload.query,
        use_rag=payload.use_rag,
        user_id=current_user.id,
        document_id=payload.document_id,
        history=history,
    )
    response.thread_id = thread_id

    conversations.append(db, current_user, thread_id, "user", payload.query)
    conversations.append(db, current_user, thread_id, "assistant", response.answer)
    return response


@router.delete("/{thread_id}", status_code=204)
async def clear_conversation(
    thread_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    ConversationService().clear(db, current_user, thread_id)
