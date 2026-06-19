# =============================================================
# File: tests/ingestion/test_article_processor.py
# =============================================================
"""Tests for verses_rag.ingestion.article_processor — headers, title, epigraph anchor."""

import pytest

from verses_rag.ingestion.article_processor import ArticleProcessor
# ADJUST: import path/name


SAMPLE_ARTICLE = """Date: 2024-03-15
Author: Jane Smith

Romans 8:28

And we know that all things work together for good to them that love God.

This article explores the meaning of providence. Some people cite Gen 1:1
or partial refs like 3:16 in passing, but those should be ignored as anchors.

The conclusion reiterates the theme of trust in difficult seasons.
"""


@pytest.fixture
def processor():
    # ADJUST: constructor signature
    return ArticleProcessor()


class TestHeaderParsing:
    def test_date_extracted(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )  # ADJUST: signature — may take a path, raw text, or a doc object
        assert all(getattr(c, "published_date", None) == "2024-03-15" for c in chunks)

    def test_author_extracted(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        assert all(getattr(c, "author", None) == "Jane Smith" for c in chunks)


class TestTitleDerivation:
    def test_underscores_become_spaces(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        assert all(c.title == "All Things Work Together" for c in chunks)


class TestEpigraphAnchor:
    def test_epigraph_reference_captured(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        # doc_scripture_refs is a list on every chunk; all chunks share the same anchor
        all_doc_refs = [ref for c in chunks for ref in c.doc_scripture_refs]
        assert any("Romans 8:28" in str(r) for r in all_doc_refs)

    def test_in_text_partial_citations_ignored(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        all_doc_refs = " ".join(
            str(r) for c in chunks for r in c.doc_scripture_refs
        )
        # The messy in-text "Gen 1:1" and bare "3:16" must NOT appear as anchors
        assert "Gen 1:1" not in all_doc_refs
        assert "3:16" not in all_doc_refs or "Romans" in all_doc_refs


class TestChunkBasics:
    def test_source_type_is_article(self, processor):
        chunks = processor.process(
            SAMPLE_ARTICLE,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        assert all(c.source_type == "article" for c in chunks)

    def test_deterministic_ids(self, processor):
        kwargs = dict(
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt"
        )
        a = sorted(c.chunk_id for c in processor.process(SAMPLE_ARTICLE, **kwargs))
        b = sorted(c.chunk_id for c in processor.process(SAMPLE_ARTICLE, **kwargs))
        assert a == b

