"""
Ingest orchestrator (SPEC §4.2, Phase 1).

Chains the ingestion components into one pass:

    articles dir --(loader_router.iter_corpus)--> LoadedDoc
                 --(classifier.classify_doc)----> article | bible
        article  --> ArticleProcessor.process(text, path, meta)
        bible    --> FLAGGED & skipped (raw-scripture text has no processor here)
    KJV JSON     --(BibleProcessor.process_file)-> Bible chunks   [separate path]

Returns the combined list[BaseChunk] (Chunk + ArticleChunk) plus an IngestReport.
Embedding / upsert is Phase 2 and intentionally NOT done here.

Idempotency (SPEC §4.2): chunk_ids are content-addressed, so re-ingesting unchanged
files yields identical ids. Within a run, duplicate ids are de-duplicated and counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from verses_rag.config.settings import Settings, get_settings
from verses_rag.ingestion.article_processor import ArticleProcessor
from verses_rag.ingestion.bible_processor import BibleProcessor
from verses_rag.ingestion.classifier import classify_doc
from verses_rag.ingestion.loader_router import iter_corpus
from verses_rag.schema import BaseChunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ingest")


@dataclass
class IngestReport:
    files_seen: int = 0
    articles_processed: int = 0
    bible_flagged: int = 0  # text files classified as scripture (skipped)
    bible_json_chunks: int = 0
    article_chunks: int = 0
    duplicates_removed: int = 0
    flagged_paths: list[str] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return self.bible_json_chunks + self.article_chunks

    def summary(self) -> str:
        return (
            f"files_seen={self.files_seen} articles={self.articles_processed} "
            f"bible_flagged={self.bible_flagged} | chunks total={self.total_chunks} "
            f"(bible={self.bible_json_chunks}, article={self.article_chunks}) "
            f"dupes_removed={self.duplicates_removed}"
        )


def ingest(
    settings: Settings | None = None,
    articles_dir: Path | str | None = None,
    include_bible: bool = True,
    treat_all_as_articles: bool = False,
) -> tuple[list[BaseChunk], IngestReport]:
    """Run the full ingestion pass.

    Args:
        settings: override; defaults to get_settings().
        articles_dir: override; defaults to <corpus_dir>/articles.
        include_bible: also ingest the KJV JSON via BibleProcessor.
        treat_all_as_articles: skip classification and process every loaded file as an
            article. Use when the articles dir is trusted to contain only articles
            (avoids the rare false-positive where an article named like a canonical
            book — e.g. 'Genesis.txt' — would be flagged as scripture).
    """
    s = settings or get_settings()
    articles_dir = Path(articles_dir) if articles_dir else s.articles_dir
    report = IngestReport()

    article_proc = ArticleProcessor.from_settings(s.article)
    by_id: dict[str, BaseChunk] = {}

    def _add(chunks: list[BaseChunk]) -> int:
        added = 0
        for c in chunks:
            if c.chunk_id in by_id:
                report.duplicates_removed += 1
            else:
                by_id[c.chunk_id] = c
                added += 1
        return added

    # --- Articles: loader -> classify -> process ---
    if articles_dir.exists():
        for doc in iter_corpus(articles_dir):
            report.files_seen += 1
            if not treat_all_as_articles:
                verdict = classify_doc(doc)
                if verdict.source_type == "bible":
                    report.bible_flagged += 1
                    report.flagged_paths.append(doc.source_path)
                    log.warning(
                        "flagged as scripture text, skipping (no raw-Bible processor): %s",
                        doc.source_path,
                    )
                    continue
            report.article_chunks += _add(
                article_proc.process(doc.text, doc.source_path, doc.meta)
            )
            report.articles_processed += 1
    else:
        log.warning("articles dir not found: %s", articles_dir)

    # --- Bible: structured JSON, separate path ---
    if include_bible:
        if s.kjv_path.exists():
            bible_proc = BibleProcessor.from_settings(s.bible)
            report.bible_json_chunks += _add(bible_proc.process_file(s.kjv_path))
        else:
            log.warning("KJV JSON not found, skipping Bible: %s", s.kjv_path)

    log.info("ingest complete: %s", report.summary())
    return list(by_id.values()), report


def main():
    import sys

    # Articles-only is the fast default for a smoke test; pass --bible for the full run.
    include_bible = "--bible" in sys.argv
    chunks, report = ingest(include_bible=include_bible)

    print("\n" + report.summary())
    if report.flagged_paths:
        print("\nFlagged (review — possibly misnamed articles):")
        for p in report.flagged_paths:
            print(f"  {p}")

    arts = [c for c in chunks if c.source_type == "article"]
    print(f"\nSample article chunks ({len(arts)} total):")
    for c in arts[:3]:
        refs = getattr(c, "doc_scripture_refs", [])
        print(f"  {c.title[:38]:38} anchor={refs}  {c.chunk_id[:10]}…")


if __name__ == "__main__":
    main()
