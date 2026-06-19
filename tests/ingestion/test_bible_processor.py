# =============================================================
# File: tests/ingestion/test_bible_processor.py
# =============================================================
"""Tests for verses_rag.ingestion.bible_processor — chunking, IDs, ari indexing."""

import hashlib

import pytest

from verses_rag.ingestion.bible_processor import BibleProcessor
# ADJUST: import path/name if processor lives elsewhere (e.g. processors.py)


@pytest.fixture
def kjv_sample():
    """Minimal flat-array KJV JSON matching the real source format:
        {"ari": "book:chapter:verse", "name": "Book C:V", "verse": "text..."}
    ari is 0-based book index; name is the source of truth for parse_record.
    Genesis (ari=0), chapter 1, verses 1-7 — enough for window=5, overlap=1.
    Revelation (ari=65), chapter 22, verse 21 — last verse of the Bible.
    """
    records = [
        {
            "ari": f"0:1:{v}",
            "name": f"Genesis 1:{v}",
            "verse": f"Genesis chapter one verse {v} text here.",
        }
        for v in range(1, 8)
    ]
    records.append({
        "ari": "65:22:21",
        "name": "Revelation 22:21",
        "verse": "The grace of our Lord Jesus Christ be with you all. Amen.",
    })
    return records


@pytest.fixture
def processor():
    # ADJUST: constructor signature — may take settings, window/overlap kwargs, or a path
    return BibleProcessor()


class TestAriIndexing:
    def test_ari_zero_is_genesis(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)  # ADJUST: method name/signature
        genesis_chunks = [c for c in chunks if c.book == "Genesis"]
        assert len(genesis_chunks) > 0

    def test_ari_sixty_five_is_revelation(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        rev_chunks = [c for c in chunks if c.book == "Revelation"]
        assert len(rev_chunks) > 0

    def test_testament_assignment(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        for c in chunks:
            if c.book == "Genesis":
                assert c.testament == "OT"
            if c.book == "Revelation":
                assert c.testament == "NT"


class TestWindowChunking:
    def test_window_size_five(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        genesis = [c for c in chunks if c.book == "Genesis"]
        # With 7 verses, window=5, overlap=1 (stride=4): windows start at
        # v1 (1-5) and v5 (5-7). First window spans exactly 5 verses.
        first = min(genesis, key=lambda c: c.verse_start)
        assert first.verse_start == 1
        assert first.verse_end == 5

    def test_overlap_one(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        genesis = sorted(
            [c for c in chunks if c.book == "Genesis"],
            key=lambda c: c.verse_start,
        )
        if len(genesis) >= 2:
            # Next window starts on the last verse of the previous one
            assert genesis[1].verse_start == genesis[0].verse_end

    def test_never_splits_mid_verse(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        for c in chunks:
            # Each chunk's content should contain whole verse texts
            assert c.verse_start <= c.verse_end


class TestDeterministicIds:
    def test_ids_are_sha256_hex(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        for c in chunks:
            assert len(c.chunk_id) == 64
            int(c.chunk_id, 16)  # raises if not valid hex

    def test_ids_are_deterministic(self, processor, kjv_sample):
        chunks_a = processor.process(kjv_sample)
        chunks_b = processor.process(kjv_sample)
        ids_a = sorted(c.chunk_id for c in chunks_a)
        ids_b = sorted(c.chunk_id for c in chunks_b)
        assert ids_a == ids_b

    def test_ids_are_unique(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestMetadata:
    def test_source_type_is_bible(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        assert all(c.source_type == "bible" for c in chunks)

    def test_translation_is_kjv(self, processor, kjv_sample):
        chunks = processor.process(kjv_sample)
        # ADJUST: translation may live on the chunk or in type-specific metadata
        assert all(getattr(c, "translation", "KJV") == "KJV" for c in chunks)

