"""
Shared chunk schema for the verses-rag app (SPEC §5.1).

BaseChunk holds the common metadata every chunk carries regardless of source type.
Type-specific chunks subclass it and add their §5.2 fields:
    Chunk(BaseChunk)        -> Bible    (verses_rag/ingestion/bible_processor.py)
    ArticleChunk(BaseChunk) -> articles (verses_rag/ingestion/article_processor.py)

Keeping the common fields in one place prevents the two from drifting.

Dataclass inheritance rule: every field here is REQUIRED (no default), so subclasses
may add either more required fields or defaulted ones in any order. Do NOT add a
defaulted field to BaseChunk — Python would then force every subclass field to be
defaulted too (a field with a default cannot precede one without, across the
base→subclass boundary).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseChunk:
    doc_id: str
    chunk_id: str
    chunk_index: int
    source_type: str
    title: str
    source_path: str
    content_hash: str
    ingested_at: str
    status: str
    content: str
