"""Tests for verses_rag.ingestion.canon — book list, ordering, ref extraction."""

import pytest

from verses_rag.canon import (
    KJV_BOOKS,
    BOOK_ORDER,
    OT_COUNT,
    extract_scripture_refs,
)


class TestCanonStructure:
    def test_sixty_six_books(self):
        assert len(KJV_BOOKS) == 66

    def test_ot_count(self):
        assert OT_COUNT == 39

    def test_first_and_last_books(self):
        assert KJV_BOOKS[0] == "Genesis"
        assert KJV_BOOKS[-1] == "Revelation"

    def test_testament_boundary(self):
        # Index 38 = Malachi (last OT), index 39 = Matthew (first NT)
        assert KJV_BOOKS[OT_COUNT - 1] == "Malachi"
        assert KJV_BOOKS[OT_COUNT] == "Matthew"

    def test_book_order_is_consistent_with_list(self):
        # BOOK_ORDER should map every book name to its position.
        # ADJUST: if BOOK_ORDER is 1-based (1–66 per spec §5.2), keep as-is;
        # if it's 0-based, change `i + 1` to `i`.
        for i, book in enumerate(KJV_BOOKS):
            assert BOOK_ORDER[book] == i + 1

    def test_no_duplicate_books(self):
        assert len(set(KJV_BOOKS)) == 66


class TestExtractScriptureRefs:
    def test_simple_reference(self):
        refs = extract_scripture_refs("As it says in Genesis 1:1, in the beginning...")
        assert len(refs) >= 1
        # ADJUST: depends on return shape — string vs parsed tuple/dict.
        # If refs are strings:
        assert any("Genesis 1:1" in str(r) for r in refs)

    def test_verse_range(self):
        refs = extract_scripture_refs("Read Romans 8:28-30 carefully.")
        assert len(refs) >= 1
        assert any("Romans" in str(r) for r in refs)

    def test_numbered_book(self):
        refs = extract_scripture_refs("See 1 Corinthians 13:4 for the definition of love.")
        assert len(refs) >= 1
        assert any("Corinthians" in str(r) for r in refs)

    def test_no_references(self):
        refs = extract_scripture_refs("This text mentions no scripture at all.")
        assert refs == [] or refs is None or len(refs) == 0

    def test_multiple_references(self):
        text = "Compare John 3:16 with Romans 5:8 on this point."
        refs = extract_scripture_refs(text)
        assert len(refs) == 2

    def test_does_not_match_bare_numbers(self):
        # "30-day notice" or "Chapter 4" style text should not produce refs
        refs = extract_scripture_refs("Employees must give a 30-day notice per Chapter 4.")
        assert len(refs) == 0

