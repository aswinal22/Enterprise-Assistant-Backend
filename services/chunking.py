import re

from logging_utils import get_logger


SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"\n\s*\n+")
logger = get_logger(__name__)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_paragraphs(text: str) -> list[str]:
    return [
        cleaned
        for paragraph in PARAGRAPH_BOUNDARY_PATTERN.split(text)
        if (cleaned := _normalize_whitespace(paragraph))
    ]


def _split_sentences(paragraph: str) -> list[str]:
    sentences = [
        cleaned
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(paragraph)
        if (cleaned := _normalize_whitespace(sentence))
    ]
    return sentences or [paragraph]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    words = sentence.split()
    segments: list[str] = []
    current_words: list[str] = []

    for word in words:
        tentative = " ".join([*current_words, word]).strip()
        if current_words and len(tentative) > max_chars:
            segments.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        segments.append(" ".join(current_words))

    return segments


def _append_with_overlap(chunks: list[str], current_parts: list[str], overlap_chars: int) -> list[str]:
    chunk = " ".join(current_parts).strip()
    if not chunk:
        return []

    chunks.append(chunk)
    if overlap_chars <= 0:
        return []

    overlap_source = chunk[-overlap_chars:].strip()
    if not overlap_source:
        return []

    return [overlap_source]


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        logger.warning("Received empty text for chunking")
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    paragraphs = _split_paragraphs(normalized_text) or [_normalize_whitespace(normalized_text)]

    for paragraph in paragraphs:
        sentence_units: list[str] = []
        for sentence in _split_sentences(paragraph):
            sentence_units.extend(_split_long_sentence(sentence, chunk_size))

        for sentence in sentence_units:
            sentence_length = len(sentence)
            separator_length = 1 if current_parts else 0

            if current_parts and current_length + separator_length + sentence_length > chunk_size:
                current_parts = _append_with_overlap(chunks, current_parts, overlap)
                current_length = len(" ".join(current_parts))
                separator_length = 1 if current_parts else 0

            current_parts.append(sentence)
            current_length += separator_length + sentence_length

        if current_parts:
            paragraph_break_length = 1
            if current_length + paragraph_break_length > chunk_size:
                current_parts = _append_with_overlap(chunks, current_parts, overlap)
                current_length = len(" ".join(current_parts))

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    final_chunks = [chunk for chunk in chunks if chunk]
    logger.info(
        "Chunking completed chunk_count=%s input_length=%s chunk_size=%s overlap=%s",
        len(final_chunks),
        len(normalized_text),
        chunk_size,
        overlap,
    )
    return final_chunks
