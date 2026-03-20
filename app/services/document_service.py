from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document
from app.models.user import User
from app.services.vector_store import VectorStoreService
from app.utils.text import chunk_text


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

    def list_documents(self, db: Session, owner: User) -> list[Document]:
        return (
            db.query(Document)
            .filter(Document.owner_id == owner.id)
            .order_by(Document.created_at.desc())
            .all()
        )
