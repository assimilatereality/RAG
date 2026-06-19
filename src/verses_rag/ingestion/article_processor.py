"""
ArticleProcessor for the Scripture+Article RAG app (SPEC §4.2, D2).

Handles blog/website prose. Key feature for this corpus: articles cite scripture
(book chapter:verse), so we extract those references as metadata to LINK articles
to the KJV Bible corpus. Linking is by REFERENCE, never by quote text — articles
quote various translations, but the corpus Bible is KJV.

Corpus format (from the real sample in /Volumes/X10 Pro/RAG/articles):
  - Title = the FILENAME (without .txt). Not the first content line.
  - An optional header block of "Date: ..." and "Author: ..." lines at the top,
    parsed into published_date / author and stripped from the chunked body.
  - A scripture epigraph (quote + "– Book C:V" citation), then the article body.
    The citation may be a non-KJV translation, so we link by REFERENCE, not text.
  - Explicit / loader-supplied meta (e.g. HTML <meta>) takes precedence over the
    in-file header when both are present.

Requires:  uv add langchain-text-splitters
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Canon + scripture-reference resolver live in the shared module.
from verses_rag.canon import extract_scripture_refs
from verses_rag.schema import BaseChunk

log = logging.getLogger("article_processor")


# --- in-file header parsing (article corpus convention) ---------------------
_HEADER_RE = re.compile(r"^\s*(date|author)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y")


def _parse_date(s: str) -> str:
    """ISO 'YYYY-MM-DD' if a known format is recognized, else the raw string."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def _parse_header(text: str) -> tuple[dict, str]:
    """Pull leading 'Date:'/'Author:' lines into meta; return (meta, body).

    Consumes leading blank lines and header lines up to the first real content
    line (e.g. the scripture epigraph). With no header, body == text unchanged.
    """
    meta: dict = {}
    lines = text.splitlines()
    consumed = 0
    for i, ln in enumerate(lines):
        if not ln.strip():
            consumed = i + 1
            continue
        m = _HEADER_RE.match(ln)
        if not m:
            break
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "date":
            meta["published_date"] = _parse_date(val)
        else:
            meta["author"] = val
        consumed = i + 1
    body = "\n".join(lines[consumed:]).lstrip("\n")
    return meta, body


# --- chunk schema ------------------------------------------------------------
@dataclass
class ArticleChunk(BaseChunk):
    # §5.2 article-specific (common §5.1 fields inherited from BaseChunk)
    author: str | None = None
    source_url: str | None = None
    site_name: str | None = None
    published_date: str | None = None
    retrieved_date: str | None = None
    topics: list[str] = field(default_factory=list)
    # cross-corpus link to the Bible
    scripture_refs: list[str] = field(default_factory=list)  # refs in THIS chunk
    doc_scripture_refs: list[str] = field(
        default_factory=list
    )  # all refs in the article


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class ArticleProcessor:
    """Type-specific processor for prose articles (SPEC D2)."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        prepend_title: bool = True,
        extract_refs: bool = True,
        status: str = "active",
    ):
        self.prepend_title = prepend_title
        self.extract_refs = extract_refs
        self.status = status
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @classmethod
    def from_settings(cls, cfg) -> ArticleProcessor:
        """Build from an ArticleChunkingSettings block (config/settings.py)."""
        return cls(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            prepend_title=cfg.prepend_title,
            extract_refs=cfg.extract_refs,
            status=cfg.status,
        )

    @staticmethod
    def load(path: Path | str) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace")

    def process(
        self,
        raw_text: str,
        source_path: str,
        meta: dict | None = None,
    ) -> list[ArticleChunk]:
        # In-file header (Date:/Author:) parsed out; explicit/loader meta wins over it.
        header_meta, body = _parse_header(raw_text)
        meta = {**header_meta, **(meta or {})}

        # Title is the FILENAME (article convention), with _/- as spaces, unless supplied.
        stem = (
            Path(source_path).stem.replace("_", " ").replace("-", " ")
            if source_path
            else None
        )
        title = meta.get("title") or stem or "Untitled"

        # Index anchor = the INITIAL quoted scripture (the epigraph), i.e. the first
        # reference in the body. Other in-text refs are intentionally ignored: they
        # appear in messy forms ("Psalm 19 ... (v. 1)") that can't be parsed reliably.
        # The regex already skips those (needs an adjacent chapter:verse); here we
        # further restrict recording to just the primary anchor.
        all_refs = extract_scripture_refs(body) if self.extract_refs else []
        primary = all_refs[0] if all_refs else None
        doc_refs = [primary] if primary else []
        doc_id = _sha("article", source_path)
        ingested_at = datetime.now(UTC).isoformat()

        out: list[ArticleChunk] = []
        for idx, piece in enumerate(self._splitter.split_text(body)):
            content = f"{title}\n\n{piece}" if self.prepend_title else piece
            chunk_refs = (
                [primary]
                if primary and primary in extract_scripture_refs(piece)
                else []
            )
            chash = _sha(content)
            out.append(
                ArticleChunk(
                    doc_id=doc_id,
                    chunk_id=_sha(doc_id, chash),
                    chunk_index=idx,
                    source_type="article",
                    title=title,
                    source_path=source_path,
                    content_hash=chash,
                    ingested_at=ingested_at,
                    status=self.status,
                    content=content,
                    author=meta.get("author"),
                    source_url=meta.get("source_url"),
                    site_name=meta.get("site_name"),
                    published_date=meta.get("published_date"),
                    retrieved_date=meta.get("retrieved_date"),
                    topics=meta.get("topics", []),
                    scripture_refs=chunk_refs,
                    doc_scripture_refs=doc_refs,
                )
            )
        log.info("'%s' -> %d chunks, %d scripture refs", title, len(out), len(doc_refs))
        return out

    def process_file(
        self, path: Path | str, meta: dict | None = None
    ) -> list[ArticleChunk]:
        return self.process(self.load(path), source_path=str(path), meta=meta)


def main():
    # Mirrors the real corpus format: filename = title, Date/Author header, epigraph.
    sample = (
        "Date: June 7, 2026\n"
        "Author: Yael Eckstein\n\n"
        "The law of the LORD is perfect, refreshing the soul. The statutes of the "
        "LORD are trustworthy, making wise the simple. – Psalm 19:7\n\n"
        + "This article reflects on the perfection of God's law for daily living. "
        * 5
    )
    proc = ArticleProcessor(chunk_size=400, chunk_overlap=60)
    chunks = proc.process(
        sample, source_path="/Volumes/X10 Pro/RAG/articles/The Perfect Law.txt"
    )

    print(f"\nTitle:  {chunks[0].title}")
    print(f"Author: {chunks[0].author}")
    print(f"Date:   {chunks[0].published_date}")
    print(f"Doc scripture refs: {chunks[0].doc_scripture_refs}")
    print(f"Total chunks: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c.chunk_index}] refs={c.scripture_refs}  id={c.chunk_id[:12]}…")
        print(f"  {c.content[:120]}{'…' if len(c.content) > 120 else ''}\n")


if __name__ == "__main__":
    main()
