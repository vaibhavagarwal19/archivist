from datetime import datetime

from pydantic import BaseModel


class DocumentRead(BaseModel):
    id: int
    file_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentRead


class DocumentListResponse(BaseModel):
    documents: list[DocumentRead]
