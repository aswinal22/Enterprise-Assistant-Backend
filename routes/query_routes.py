import math
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.document_chunk import DocumentChunk
from schemas import QueryRequest, QueryResponse
from services.embeddings import generate_embedding
from services.llm_service import generate_answer, generate_general_answer

router = APIRouter(tags=["query"])

SIMILARITY_THRESHOLD = 0.2
MIN_CONTEXT_CONFIDENCE = 0.35


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return numerator / (first_norm * second_norm)


def _sort_key(chunk: DocumentChunk) -> datetime:
    return chunk.created_at or datetime.min


def _clean_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit].rstrip()


def _clean_fragment(text: str, question: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    question_terms = [term for term in question.lower().split() if len(term) > 3]

    for term in question_terms:
        index = lowered.find(term)
        if index != -1:
            start = max(0, index - 80)
            end = min(len(normalized), index + limit)
            return normalized[start:end].strip()

    return normalized[:limit].strip()


def _fallback_answer(question: str, chunks: list[DocumentChunk], used_document_context: bool) -> str:
    if not used_document_context:
        return (
            f"Direct answer: I could not generate a model-written general answer for the question "
            f"\"{question}\".\n\n"
            "Notes:\n"
            "- No sufficiently related document context was found.\n"
            "- The LLM request also failed, so there is no reliable generated answer available."
        )

    bullet_points = [
        f"- {_clean_fragment(chunk.chunk_text, question)}"
        for chunk in chunks
        if chunk.chunk_text.strip()
    ]

    if not bullet_points:
        return "Direct answer: I could not find relevant context for this question."

    return (
        f"Direct answer: I could not generate a model-written answer for the question "
        f"\"{question}\", so here is a clean summary from the matched document content.\n\n"
        "Key points:\n"
        + "\n".join(bullet_points)
    )


def _format_general_question_answer(answer: str) -> str:
    return (
        "Note: this question did not match any company-specific policy documents, "
        "so this response is a general answer not based on company policy.\n\n"
        + answer
    )


def _tokenize(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if len(token.strip(".,:;!?()[]{}\"'")) > 2}


def _keyword_similarity(question: str, chunk_text: str) -> float:
    question_terms = _tokenize(question)
    chunk_terms = _tokenize(chunk_text)
    if not question_terms or not chunk_terms:
        return 0.0

    overlap = question_terms.intersection(chunk_terms)
    return len(overlap) / len(question_terms)


def _chunk_similarity(question: str, query_embedding: list[float] | None, chunk: DocumentChunk) -> float:
    if query_embedding is not None and chunk.embedding:
        return _cosine_similarity(query_embedding, chunk.embedding)
    return _keyword_similarity(question, chunk.chunk_text)


def _select_relevant_chunks(
    chunks: list[DocumentChunk],
    question: str,
    query_embedding: list[float] | None,
    top_k: int,
) -> tuple[list[DocumentChunk], float | None]:
    ranked_chunks = [
        (chunk, _chunk_similarity(question, query_embedding, chunk))
        for chunk in chunks
        if chunk.chunk_text
    ]

    if not ranked_chunks:
        return [], None

    ranked_chunks.sort(key=lambda item: (item[1], _sort_key(item[0])), reverse=True)
    best_similarity = ranked_chunks[0][1]
    selected_chunks = [chunk for chunk, score in ranked_chunks if score >= SIMILARITY_THRESHOLD][:top_k]

    return selected_chunks, best_similarity


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query message cannot be empty.")

    chunks = db.query(DocumentChunk).filter(DocumentChunk.company_id == payload.company_id).all()

    try:
        query_embedding = generate_embedding(question)
    except Exception:
        query_embedding = None

    selected_chunks, best_similarity = _select_relevant_chunks(chunks, question, query_embedding, payload.top_k)
    used_document_context = bool(selected_chunks and best_similarity is not None and best_similarity >= MIN_CONTEXT_CONFIDENCE)
    context = "\n\n".join(chunk.chunk_text for chunk in selected_chunks) if used_document_context else ""

    llm_error = None
    try:
        if used_document_context:
            answer = generate_answer(question, context)
        else:
            answer = _format_general_question_answer(generate_general_answer(question))
        llm_used = True
    except Exception as exc:
        llm_used = False
        llm_error = str(exc)
        answer = _fallback_answer(question, selected_chunks if used_document_context else [], used_document_context)
        if not used_document_context:
            answer = _format_general_question_answer(answer)

    return QueryResponse(
        answer=answer,
        chunks_used=len(selected_chunks),
        llm_used=llm_used,
        llm_error=llm_error,
        used_document_context=used_document_context,
        best_similarity=best_similarity,
    )
