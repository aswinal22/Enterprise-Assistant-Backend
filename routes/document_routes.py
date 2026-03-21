from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.document import Document
from models.document_chunk import DocumentChunk
from schemas import DocumentCreate, DocumentCreateResponse, DocumentResponse
from services.chunking import chunk_text
from services.embeddings import generate_embedding
from services.llm_service import generate_summary

router = APIRouter(tags=["documents"])


def _try_generate_embedding(text: str) -> list[float] | None:
    try:
        return generate_embedding(text)
    except Exception:
        return None


@router.post("/documents", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> DocumentCreateResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document text cannot be empty.")

    document = Document(
        company_id=payload.company_id,
        user_id=payload.user_id,
        filename=payload.filename,
        content=text,
    )
    db.add(document)
    db.flush()

    chunks = chunk_text(text)
    db.add_all(
        [
            DocumentChunk(
                company_id=payload.company_id,
                document_id=document.id,
                chunk_index=index,
                chunk_text=chunk,
                embedding=_try_generate_embedding(chunk),
            )
            for index, chunk in enumerate(chunks)
        ]
    )
    db.commit()
    db.refresh(document)

    summary = None
    try:
        summary = generate_summary(text)
    except Exception:
        pass  # Optional: log the error if needed

    return DocumentCreateResponse(document_id=document.id, chunks_created=len(chunks), summary=summary)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(document)
