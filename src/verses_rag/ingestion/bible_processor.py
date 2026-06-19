"""
BibleProcessor for the Scripture+Article RAG app (SPEC §4.2, D2).

Input: a flat-array KJV JSON of the form
    [{"ari": "0:1:1", "name": "Genesis 1:1",
      "verse": "In the beginning God created the heaven and the earth."}, ...]

Output: list[Chunk] honoring the §5.1/§5.2 metadata schema, ready to embed.

Design notes:
  - `name` is the source of truth for book/chapter/verse; `ari` is only
    cross-checked (its 0-based book index is inferred, not assumed correct).
  - No pericope data exists in this source, so `pericope` is always None and
    chunking is verse-window (within a chapter) or per-verse. See CHUNK_STRATEGY.
  - IDs are DETERMINISTIC (content-addressed), unlike the uuid4 in
    pinecone_healthcare_assistant.py — required for the idempotent re-ingest in §4.2.
  - Bad/unknown records are logged and skipped, never fatal (R4 philosophy).

Run:
    python bible_processor.py            # uses DEFAULT_PATH below
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from verses_rag.canon import BOOK_ORDER, OT_COUNT
from verses_rag.schema import BaseChunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("bible_processor")

# Note the space in the volume name: use pathlib, never a bare quoted string.
DEFAULT_PATH = Path("/Volumes/X10 Pro/RAG/kjv.json")

EXPECTED_VERSE_COUNT = 31_102  # standard KJV versification; used only as a sanity log

# "Song of Solomon 2:1" -> ("Song of Solomon", "2", "1"); non-greedy book capture.
_REF_RE = re.compile(r"^(.*?)\s+(\d+):(\d+)$")


@dataclass
class Verse:
    book: str
    book_order: int
    testament: str
    chapter: int
    verse: int
    text: str


@dataclass
class Chunk(BaseChunk):
    # --- §5.2 Bible-specific (common §5.1 fields inherited from BaseChunk) ---
    translation: str
    testament: str
    book: str
    book_order: int
    chapter: int
    verse_start: int
    verse_end: int
    pericope: str | None = None


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def parse_record(rec: dict) -> Verse | None:
    """Parse one JSON verse record into a Verse, or None if unusable."""
    name = (rec.get("name") or "").strip()
    text = (rec.get("verse") or "").strip()
    if not name or not text:
        log.warning("skipping record with missing name/verse: %r", rec.get("ari"))
        return None

    m = _REF_RE.match(name)
    if not m:
        log.warning("unparseable reference, skipping: %r", name)
        return None
    book, chap_s, verse_s = m.group(1), m.group(2), m.group(3)

    order = BOOK_ORDER.get(book)
    if order is None:
        # e.g. "Psalm" vs "Psalms", "Song of Songs" vs "Song of Solomon", apocrypha.
        log.warning("book not in KJV canon list, skipping: %r", book)
        return None

    chapter, verse = int(chap_s), int(verse_s)

    # Cross-check ari's 0-based book index if present; warn but trust `name`.
    ari = rec.get("ari", "")
    ari_parts = ari.split(":")
    if len(ari_parts) == 3 and ari_parts[0].isdigit():
        if int(ari_parts[0]) + 1 != order:
            log.warning("ari/name book mismatch (%s vs %s); using name", ari, book)

    testament = "OT" if order <= OT_COUNT else "NT"
    return Verse(book, order, testament, chapter, verse, text)


class BibleProcessor:
    """Type-specific processor for KJV scripture (SPEC D2).

    chunk_strategy:
        "window" -> group `window_size` consecutive verses (step = window_size -
                    overlap), never crossing a chapter boundary. Good retrieval context.
        "verse"  -> one chunk per verse. Perfect citation granularity, weaker context.
    include_verse_numbers: prefix each verse with its number in the chunk text.
        Default False keeps embeddings clean; verse_start/end already live in metadata.
    """

    def __init__(
        self,
        chunk_strategy: str = "window",
        window_size: int = 5,
        overlap: int = 1,
        include_verse_numbers: bool = False,
        status: str = "active",
    ):
        if chunk_strategy not in ("window", "verse"):
            raise ValueError("chunk_strategy must be 'window' or 'verse'")
        if chunk_strategy == "window" and not (0 <= overlap < window_size):
            raise ValueError("require 0 <= overlap < window_size")
        self.chunk_strategy = chunk_strategy
        self.window_size = window_size
        self.overlap = overlap
        self.include_verse_numbers = include_verse_numbers
        self.status = status

    @classmethod
    def from_settings(cls, cfg) -> BibleProcessor:
        """Build from a BibleChunkingSettings block (config/settings.py)."""
        return cls(
            chunk_strategy=cfg.chunk_strategy,
            window_size=cfg.window_size,
            overlap=cfg.overlap,
            include_verse_numbers=cfg.include_verse_numbers,
            status=cfg.status,
        )

    # --- loading ---
    @staticmethod
    def load(path: Path | str) -> list[dict]:
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"expected a JSON array of verse records, got {type(data)}"
            )
        return data

    # --- base interface: process(raw_doc) -> list[Chunk] ---
    def process(
        self, raw_doc: list[dict], source_path: str = str(DEFAULT_PATH)
    ) -> list[Chunk]:
        verses = [v for v in (parse_record(r) for r in raw_doc) if v is not None]
        if abs(len(verses) - EXPECTED_VERSE_COUNT) > 5:
            log.info(
                "parsed %d verses (standard KJV is %d) — fine if your source differs",
                len(verses),
                EXPECTED_VERSE_COUNT,
            )
        ingested_at = datetime.now(UTC).isoformat()
        chunks: list[Chunk] = []
        # File is in canonical order, so consecutive grouping by (book, chapter) is safe.
        for (_order, _chapter), grp in groupby(
            verses, key=lambda v: (v.book_order, v.chapter)
        ):
            chunks.extend(self._chunk_chapter(list(grp), source_path, ingested_at))
        log.info("produced %d chunks (%s strategy)", len(chunks), self.chunk_strategy)
        return chunks

    def process_file(self, path: Path | str = DEFAULT_PATH) -> list[Chunk]:
        return self.process(self.load(path), source_path=str(path))

    # --- chunking ---
    def _windows(self, n: int) -> Iterator[tuple[int, int]]:
        if self.chunk_strategy == "verse":
            for i in range(n):
                yield i, i + 1
            return
        step = self.window_size - self.overlap
        i = 0
        while i < n:
            yield i, min(i + self.window_size, n)
            if i + self.window_size >= n:
                break
            i += step

    def _render(self, window: list[Verse]) -> str:
        if self.include_verse_numbers:
            return " ".join(f"{v.verse} {v.text}" for v in window)
        return " ".join(v.text for v in window)

    def _chunk_chapter(
        self, verses: list[Verse], source_path: str, ingested_at: str
    ) -> list[Chunk]:
        out: list[Chunk] = []
        book = verses[0].book
        doc_id = _sha("KJV", book)  # one "document" per book
        for idx, (lo, hi) in enumerate(self._windows(len(verses))):
            window = verses[lo:hi]
            text = self._render(window)
            vs, ve = window[0].verse, window[-1].verse
            chunk_id = _sha("KJV", book, str(verses[0].chapter), f"{vs}-{ve}")
            out.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    chunk_index=idx,
                    source_type="bible",
                    title=f"{book} {verses[0].chapter}:{vs}"
                    + (f"-{ve}" if ve != vs else ""),
                    source_path=source_path,
                    content_hash=_sha(text),
                    ingested_at=ingested_at,
                    status=self.status,
                    content=text,
                    translation="KJV",
                    testament=window[0].testament,
                    book=book,
                    book_order=window[0].book_order,
                    chapter=verses[0].chapter,
                    verse_start=vs,
                    verse_end=ve,
                    pericope=None,
                )
            )
        return out


def main():
    from verses_rag.config.settings import get_settings

    s = get_settings()
    proc = BibleProcessor.from_settings(s.bible)
    try:
        chunks = proc.process_file(s.kjv_path)
    except FileNotFoundError:
        log.error(
            "KJV file not found at %s — check corpus_dir / kjv_filename.", s.kjv_path
        )
        return

    print(f"\nTotal chunks: {len(chunks)}")
    print("First 3 chunks:\n")
    for c in chunks[:3]:
        print(f"[{c.title}]  ({c.testament}, book_order={c.book_order})")
        print(f"  id={c.chunk_id[:12]}…  verses {c.verse_start}-{c.verse_end}")
        print(f"  {c.content[:120]}{'…' if len(c.content) > 120 else ''}\n")

    # Quick per-verse comparison on the same source.
    pv = BibleProcessor(chunk_strategy="verse").process_file(s.kjv_path)
    print(f"Per-verse strategy would instead produce {len(pv)} chunks.")


if __name__ == "__main__":
    main()
