import logging

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.document import DocumentListResponse, DocumentUploadResponse
from app.services.document_service import DocumentService
from app.tasks.document_tasks import post_process_document

logger = logging.getLogger(__name__)

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
    enqueue_post_processing(document.id)
    return DocumentUploadResponse(message="Document uploaded successfully", document=document)


def enqueue_post_processing(document_id: int) -> bool:
    """Queue optional async post-processing; never fail the upload over it.

    Returns whether the job was accepted, so callers and tests can tell the
    difference between "queued" and "broker unavailable".
    """
    settings = get_settings()
    if not settings.enable_background_tasks:
        return False
    try:
        post_process_document.delay(document_id)
        return True
    except Exception as exc:
        logger.warning("Could not queue post-processing for document %s: %s", document_id, exc)
        return False
