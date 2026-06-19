# =============================================================
# File: tests/ingestion/test_classifier.py
# =============================================================
"""Tests for verses_rag.ingestion.classifier — path/filename heuristics ONLY.

Key principle (classifier inversion): Bible detection relies solely on
path/filename signals. Reference density signals an ARTICLE, not scripture.
"""

import pytest

from verses_rag.ingestion.classifier import classify
# ADJUST: function name — may be classify_document, is_bible, etc.
# ADJUST: return values assumed to be "bible" / "article" strings


ARTICLE_TEXT = "This is a prose article of sufficient length. " * 5
DENSE_REF_TEXT = (
    "See Genesis 1:1, Exodus 3:14, Romans 8:28, John 3:16, Psalm 23:1. " * 5
)


class TestPathSignals:
    def test_bible_directory_path(self):
        c = classify("", source_path="/Volumes/X10 Pro/RAG/bible/kjv.json")
        assert c.source_type == "bible"

    def test_canonical_book_filename(self):
        c = classify("", source_path="/some/dir/Genesis.txt")
        assert c.source_type == "bible"

    def test_numbered_book_filename(self):
        c = classify("", source_path="/some/dir/1_Corinthians.txt")
        assert c.source_type == "bible"


class TestArticleDefaults:
    def test_plain_article_path(self):
        c = classify(
            ARTICLE_TEXT,
            source_path="/Volumes/X10 Pro/RAG/articles/All_Things_Work_Together.txt",
        )
        assert c.source_type == "article"

    def test_non_book_filename_is_article(self):
        c = classify(ARTICLE_TEXT, source_path="/some/dir/Thoughts_On_Providence.md")
        assert c.source_type == "article"


class TestClassifierInversion:
    """Content full of verse references must NOT flip classification to bible."""

    def test_reference_dense_article_stays_article(self):
        c = classify(
            DENSE_REF_TEXT,
            source_path="/Volumes/X10 Pro/RAG/articles/Verse_Heavy_Essay.txt",
        )
        assert c.source_type == "article"

    def test_book_name_inside_article_filename_context(self):
        # Exact-stem match only — "Reflections_on_the_Gospel_of_John" != "John"
        c = classify(
            ARTICLE_TEXT,
            source_path="/Volumes/X10 Pro/RAG/articles/Reflections_on_the_Gospel_of_John.txt",
        )
        assert c.source_type == "article"
