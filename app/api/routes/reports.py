from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.report import ReportRequest, ReportResponse
from app.services.report_service import ReportService


router = APIRouter(prefix="/generate-report")


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def generate_report(
    payload: ReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReportResponse:
    service = ReportService()
    return await service.generate_report(db, current_user, payload.title, payload.prompt)
