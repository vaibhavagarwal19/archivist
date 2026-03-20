from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentUploadResponse
from app.services.document_service import DocumentService
from app.tasks.document_tasks import post_process_document


router = APIRouter(prefix="/upload-doc")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentListResponse:
    service = DocumentService()
    return DocumentListResponse(documents=service.list_documents(db, current_user))


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DocumentUploadResponse:
    service = DocumentService()
    document = await service.save_upload(db, current_user, file)
    try:
        post_process_document.delay(document.id)
    except Exception:
        # The upload should still succeed if Redis/Celery is temporarily unavailable.
        pass
    return DocumentUploadResponse(message="Document uploaded successfully", document=document)
