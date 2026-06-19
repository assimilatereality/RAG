"""
Loader router for the verses-rag ingestion pipeline (SPEC §4.2).

Maps a file extension to a loader and returns extracted text + format metadata as a
LoadedDoc. Unknown types and load failures are logged and skipped (return None) so a
single bad file never aborts the batch (SPEC R4).

The KJV Bible JSON is NOT handled here — it goes through BibleProcessor.process_file
directly. This router serves the article corpus (.txt/.md/.html/.pdf/.docx).

Optional dependencies (lazy-imported; only needed for their format):
    .html/.htm  ->  uv add beautifulsoup4
    .pdf        ->  uv add pymupdf
    .docx       ->  uv add python-docx
The text loaders (.txt/.md) need nothing.

The `meta` dict uses the same keys ArticleProcessor.process(meta=...) consumes:
    title, author, source_url, site_name, published_date, retrieved_date
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("loader_router")


@dataclass
class LoadedDoc:
    text: str
    source_path: str
    ext: str
    meta: dict = field(default_factory=dict)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


# --- per-format loaders: (path) -> (text, meta) ------------------------------
def load_text(path: Path) -> tuple[str, dict]:
    return _read_text(path), {}


def load_html(path: Path) -> tuple[str, dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise RuntimeError(
            "beautifulsoup4 not installed — `uv add beautifulsoup4`"
        ) from e

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = "\n".join(
        ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()
    )

    def _meta(attr: str, val: str) -> str | None:
        t = soup.find("meta", attrs={attr: val})
        c = t.get("content") if t else None
        return c.strip() if c else None

    title = (
        soup.title.string.strip() if soup.title and soup.title.string else None
    ) or _meta("property", "og:title")
    meta = {
        "title": title,
        "author": _meta("name", "author") or _meta("property", "article:author"),
        "site_name": _meta("property", "og:site_name"),
        "published_date": _meta("property", "article:published_time"),
        "source_url": _meta("property", "og:url")
        or (soup.find("link", rel="canonical") or {}).get("href"),
    }
    return text, {k: v for k, v in meta.items() if v}  # drop empties


def load_pdf(path: Path) -> tuple[str, dict]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("pymupdf not installed — `uv add pymupdf`") from e

    doc = fitz.open(path)
    pages = doc.page_count
    text = "\n".join(page.get_text() for page in doc)
    md = doc.metadata or {}
    doc.close()

    if len(text.strip()) < 20 * max(pages, 1):
        log.warning(
            "%s: little extractable text across %d page(s) — likely scanned; "
            "OCR is not implemented (SPEC R4)",
            path.name,
            pages,
        )

    meta = {"title": md.get("title"), "author": md.get("author")}
    return text, {k: v for k, v in meta.items() if v}


def load_docx(path: Path) -> tuple[str, dict]:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError("python-docx not installed — `uv add python-docx`") from e

    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    cp = d.core_properties
    meta = {
        "title": cp.title or None,
        "author": cp.author or None,
        "published_date": cp.created.isoformat() if cp.created else None,
    }
    return text, {k: v for k, v in meta.items() if v}


# extension -> loader
LOADERS: dict[str, Callable[[Path], tuple[str, dict]]] = {
    ".txt": load_text,
    ".md": load_text,
    ".text": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_file(path: Path | str) -> LoadedDoc | None:
    """Load one file. Returns None (logged) for unknown types or any failure."""
    path = Path(path)
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        log.info("no loader for %s (%s) — skipping", path.name, ext or "no extension")
        return None
    try:
        text, meta = loader(path)
    except Exception as e:  # missing dep, parse error, corrupt file, etc.
        log.warning("failed to load %s: %s", path.name, e)
        return None
    if not text.strip():
        log.warning("%s produced empty text — skipping", path.name)
        return None
    meta.setdefault(
        "retrieved_date", _mtime_iso(path)
    )  # proxy: when the file was saved
    return LoadedDoc(text=text, source_path=str(path), ext=ext, meta=meta)


def iter_corpus(directory: Path | str, recursive: bool = True) -> Iterator[LoadedDoc]:
    """Yield a LoadedDoc per loadable file under `directory`, skipping the rest."""
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            doc = load_file(p)
            if doc is not None:
                yield doc


def main():
    import sys
    import tempfile

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        docs = list(iter_corpus(target))
        print(f"Loaded {len(docs)} document(s) from {target}")
        for d in docs:
            print(
                f"  {Path(d.source_path).name}  [{d.ext}]  {len(d.text)} chars  meta={d.meta}"
            )
        return

    # Dependency-free self-test (only .txt/.md exercised; .json shows the skip path).
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "healing_words.txt").write_text(
            "The Healing Power of Words\n\nProverbs 12:18 reminds us about the tongue.\n",
            encoding="utf-8",
        )
        (tmp / "notes.md").write_text(
            "# Some Notes\n\nSee James 3:5.\n", encoding="utf-8"
        )
        (tmp / "data.json").write_text("{}", encoding="utf-8")  # no loader -> skipped
        docs = list(iter_corpus(tmp))
        print(f"\nSelf-test: loaded {len(docs)} of 3 files (.json skipped)\n")
        for d in docs:
            print(f"  {Path(d.source_path).name}  [{d.ext}]  {len(d.text)} chars")
            print(f"    meta={d.meta}\n")


if __name__ == "__main__":
    main()
