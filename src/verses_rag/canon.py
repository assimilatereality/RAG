"""
Shared canon + scripture-reference resolver for the verses-rag app.

A neutral home for the data and logic used across modules — bible_processor,
article_processor, the document classifier, and the §4.7 reference-resolver tool —
so no module has to import the canon from a sibling. Maps to SPEC §5.1/§5.2 + §4.7.
"""

from __future__ import annotations

import re

# Canonical 66-book Protestant/KJV order. index + 1 == book_order; first 39 = OT.
KJV_BOOKS = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",  # 39 OT
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",  # 27 NT
]
OT_COUNT = 39
BOOK_ORDER = {name: i + 1 for i, name in enumerate(KJV_BOOKS)}


# --- scripture reference resolver -------------------------------------------
# Common abbreviations; extend as the corpus surfaces more. Full names always work.
ABBREV: dict[str, list[str]] = {
    "Genesis": ["Gen", "Gn"],
    "Exodus": ["Exod", "Ex"],
    "Psalms": ["Psalm", "Ps", "Psa"],
    "Proverbs": ["Prov", "Prv"],
    "Ecclesiastes": ["Eccl"],
    "Isaiah": ["Isa"],
    "Jeremiah": ["Jer"],
    "Matthew": ["Matt", "Mt"],
    "Mark": ["Mk"],
    "Luke": ["Lk"],
    "John": ["Jn"],
    "Romans": ["Rom"],
    "1 Corinthians": ["1 Cor"],
    "2 Corinthians": ["2 Cor"],
    "Galatians": ["Gal"],
    "Ephesians": ["Eph"],
    "Philippians": ["Phil"],
    "Hebrews": ["Heb"],
    "Revelation": ["Rev"],
    "Song of Solomon": ["Song of Songs", "Song", "Canticles"],
}

# surface form (lowercased) -> canonical book name
_SURFACE: dict[str, str] = {b.lower(): b for b in KJV_BOOKS}
for _canon, _variants in ABBREV.items():
    for _v in _variants:
        _SURFACE[_v.lower()] = _canon

# Longest-first alternation so "1 John" beats "John", "Song of Songs" matches whole.
_ALT = "|".join(re.escape(s) for s in sorted(_SURFACE, key=len, reverse=True))
_REF_RE = re.compile(rf"\b({_ALT})\.?\s+(\d+):(\d+)(?:[-–—](\d+))?", re.IGNORECASE)


def extract_scripture_refs(text: str) -> list[str]:
    """Return normalized 'Book C:V' / 'Book C:V-W' refs, deduped, in order.

    Limitation: cross-chapter ranges ('Prov 12:18-13:2') are not parsed correctly;
    the trailing chapter is captured as a verse. Rare in practice — flagged, not fixed.
    """
    refs: list[str] = []
    seen: set[str] = set()
    for m in _REF_RE.finditer(text):
        canon = _SURFACE.get(m.group(1).lower().rstrip("."))
        if not canon:
            continue
        ch, v1, v2 = m.group(2), m.group(3), m.group(4)
        ref = f"{canon} {ch}:{v1}" + (f"-{v2}" if v2 else "")
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs
