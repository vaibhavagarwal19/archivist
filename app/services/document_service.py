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

# Extensions this assistant can actually read.
SUPPORTED_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log", ".yaml", ".yml"}

# Formats people most often try to upload. Named explicitly so the error can
# tell them what to do instead of failing with a generic decode message.
BINARY_SUFFIX_HINTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".tar", ".mp4", ".mp3",
}

BINARY_MAGIC_NUMBERS: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "PDF"),
    (b"PK\x03\x04", "ZIP-based Office (docx/xlsx/pptx)"),
    (b"\xd0\xcf\x11\xe0", "legacy Office (doc/xls)"),
    (b"\x89PNG", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF8", "GIF image"),
)


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

    @staticmethod
    def _decode_text(file_name: str, raw_bytes: bytes) -> str:
        """Accept only real UTF-8 text, and say plainly why anything else fails.

        This used to decode with errors="ignore", so a PDF or .docx uploaded
        successfully and was indexed as mojibake: retrieval then returned
        unreadable chunks and the agent quoted them as if they were content.
        Rejecting up front is far kinder than answering from garbage.
        """
        suffix = Path(file_name).suffix.lower()
        if suffix in BINARY_SUFFIX_HINTS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"{suffix} files are not supported yet. This assistant reads plain text "
                    f"({', '.join(sorted(SUPPORTED_SUFFIXES))}). Export the document as .txt and upload that."
                ),
            )

        for magic, label in BINARY_MAGIC_NUMBERS:
            if raw_bytes.startswith(magic):
                raise HTTPException(
                    status_code=415,
                    detail=f"This looks like a {label} file, not text. Upload a plain text export instead.",
                )

        # A NUL byte in the first block is the classic "this is binary" signal.
        if b"\x00" in raw_bytes[:8192]:
            raise HTTPException(
                status_code=415, detail="This file appears to be binary, not text."
            )

        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=415,
                detail="File must be UTF-8 encoded text. Re-save it as UTF-8 and try again.",
            ) from exc

        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file contains no readable text")
        return content

    async def save_upload(self, db: Session, owner: User, upload: UploadFile) -> Document:
        Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_file_name(upload.filename)
        file_path = Path(self.settings.upload_dir) / safe_name

        raw_bytes = await upload.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        content = self._decode_text(upload.filename or safe_name, raw_bytes)

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
        except Exception as exc:  # index cleanup must not block deletion
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
