from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    company_id: str
    user_id: str
    filename: str | None = None
    text: str


class QueryRequest(BaseModel):
    company_id: str
    message: str
    top_k: int = Field(default=3, ge=1, le=10)


class DocumentCreateResponse(BaseModel):
    document_id: UUID
    chunks_created: int
    summary: str | None = None


class DocumentResponse(BaseModel):
    id: UUID
    company_id: str
    user_id: str
    filename: str | None = None
    content: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryResponse(BaseModel):
    answer: str
    chunks_used: int
    llm_used: bool
    llm_error: str | None = None
    used_document_context: bool
    best_similarity: float | None = None
