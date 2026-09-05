from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.report import WorkflowRequest, WorkflowResponse
from app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/workflow")


@router.post("", response_model=WorkflowResponse)
async def execute_workflow(
    payload: WorkflowRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WorkflowResponse:
    service = WorkflowService()
    return await service.execute(db, current_user, payload.objective, payload.recipient_email)
