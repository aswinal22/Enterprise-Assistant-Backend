from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from logging_utils import get_logger
from models.document import Document
from models.document_chunk import DocumentChunk
from schemas import DocumentCreate, DocumentCreateResponse, DocumentResponse
from services.chunking import chunk_text
from services.embeddings import generate_embedding
from services.llm_service import generate_summary

router = APIRouter(tags=["documents"])
logger = get_logger(__name__)


def _try_generate_embedding(text: str) -> list[float] | None:
    try:
        return generate_embedding(text)
    except Exception:
        logger.exception("Chunk embedding generation failed chunk_length=%s", len(text))
        return None


@router.post("/documents", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)) -> DocumentCreateResponse:
    text = payload.text.strip()
    if not text:
        logger.warning(
            "Rejected empty document upload company_id=%s user_id=%s filename=%s",
            payload.company_id,
            payload.user_id,
            payload.filename,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document text cannot be empty.")

    logger.info(
        "Creating document company_id=%s user_id=%s filename=%s text_length=%s",
        payload.company_id,
        payload.user_id,
        payload.filename,
        len(text),
    )

    document = Document(
        company_id=payload.company_id,
        user_id=payload.user_id,
        filename=payload.filename,
        content=text,
    )
    db.add(document)
    db.flush()
    logger.info("Document persisted document_id=%s", document.id)

    chunks = chunk_text(text)
    logger.info("Creating document chunks document_id=%s chunk_count=%s", document.id, len(chunks))
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
    logger.info("Document creation committed document_id=%s", document.id)

    summary = None
    try:
        summary = generate_summary(text)
        logger.info("Document summary generated document_id=%s", document.id)
    except Exception:
        logger.exception("Document summary generation failed document_id=%s", document.id)

    return DocumentCreateResponse(document_id=document.id, chunks_created=len(chunks), summary=summary)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    logger.info("Fetching document document_id=%s", document_id)
    document = db.get(Document, document_id)
    if document is None:
        logger.warning("Document not found document_id=%s", document_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    logger.info("Document fetched successfully document_id=%s", document_id)
    return DocumentResponse.model_validate(document)
