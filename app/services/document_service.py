import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document
from app.models.user import User
from app.services.vector_store import VectorStoreService
from app.utils.text import chunk_text

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = VectorStoreService()

    @staticmethod
    def _safe_file_name(original_name: str) -> str:
        source = Path(original_name or "document.txt").name
        stem = Path(source).stem or "document"
        suffix = Path(source).suffix or ".txt"
        safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
        return f"{safe_stem}_{uuid4().hex[:8]}{suffix}"

    async def save_upload(self, db: Session, owner: User, upload: UploadFile) -> Document:
        Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_file_name(upload.filename)
        file_path = Path(self.settings.upload_dir) / safe_name

        raw_bytes = await upload.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        content = raw_bytes.decode("utf-8", errors="ignore")
        if not content.strip():
            raise HTTPException(status_code=400, detail="Only text-like documents are supported right now")

        file_path.write_bytes(raw_bytes)

        document = Document(
            file_name=upload.filename or safe_name,
            file_path=str(file_path),
            content=content,
            owner_id=owner.id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        chunks = chunk_text(content)
        self.vector_store.add_documents(
            chunks,
            [
                {
                    "document_id": document.id,
                    "file_name": document.file_name,
                    "owner_id": owner.id,
                }
                for _ in chunks
            ],
        )
        return document

    def delete_document(self, db: Session, owner: User, document_id: int) -> None:
        """Delete a document, its file, and its vectors.

        Ownership is enforced here: requesting another user's document returns
        404 rather than 403, so the endpoint does not confirm it exists.
        """
        document = db.get(Document, document_id)
        if document is None or document.owner_id != owner.id:
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            self.vector_store.delete_document(document_id)
        except Exception as exc:  # noqa: BLE001 - index cleanup must not block deletion
            logger.warning("Could not remove vectors for document %s: %s", document_id, exc)

        file_path = Path(document.file_path)
        try:
            file_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove file %s: %s", file_path, exc)

        db.delete(document)
        db.commit()

    def list_documents(self, db: Session, owner: User) -> list[Document]:
        return (
            db.query(Document)
            .filter(Document.owner_id == owner.id)
            .order_by(Document.created_at.desc())
            .all()
        )
