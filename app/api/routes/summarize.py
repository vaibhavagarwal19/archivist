from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.report import SummaryRequest, SummaryResponse
from app.services.summary_service import SummaryService

router = APIRouter(prefix="/summarize")


@router.post("", response_model=SummaryResponse)
async def summarize_meeting(
    payload: SummaryRequest, current_user: User = Depends(get_current_user)
) -> SummaryResponse:
    _ = current_user
    service = SummaryService()
    return SummaryResponse(summary=await service.summarize_meeting(payload.meeting_notes))
