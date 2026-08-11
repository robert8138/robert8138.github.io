#!/usr/bin/env python3
"""
Resolve a book title to (a) an identity and (b) a cover — two separate jobs.

v1 conflated them: it hunted for an ISBN whose cover happened to exist, and
picked whichever file was biggest. That drifts to foreign-language editions,
audiobook reissues and knock-off "Summary of ..." titles, because those are
sometimes the only editions with a scanned cover.

v2 splits the two:

  IDENTITY  an English ISBN-13 for the work        -> the join key to _books/
  COVER     the work's `cover_i` from Open Library -> what we display

`cover_i` is the cover Open Library itself picked for the work, so it is
almost always the edition a reader would recognise. It is also a single
lookup with no guessing: covers.openlibrary.org/b/id/<cover_i>-L.jpg

Guards: restrict the search to English, and reject derivative titles.
"""

import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SEARCH = "https://openlibrary.org/search.json"
COVER_BY_ID = "https://covers.openlibrary.org/b/id/{}-L.jpg?default=false"

# Titles that are about a book rather than the book itself.
DERIVATIVE = re.compile(
    r"\b(summar(y|ies|ized)|workbook|study guide|analysis of|key takeaways|busy people)\b",
    re.I,
)


def search(title):
    qs = urllib.parse.urlencode({
        "title": title,
        "language": "eng",
        "fields": "title,author_name,first_publish_year,isbn,cover_i,edition_count",
        "limit": 5,
    })
    with urllib.request.urlopen(f"{SEARCH}?{qs}", timeout=30) as r:
        return json.load(r).get("docs", [])


def cover_size(cover_id):
    try:
        with urllib.request.urlopen(COVER_BY_ID.format(cover_id), timeout=20) as r:
            return len(r.read())
    except Exception:
        return 0


def pick(docs):
    """First non-derivative doc that has a cover, preferring wide editions."""
    real = [d for d in docs if not DERIVATIVE.search(d.get("title") or "")]
    real = [d for d in real if d.get("cover_i")]
    if not real:
        return None
    # edition_count is a decent proxy for "this is the well-known work".
    real.sort(key=lambda d: -(d.get("edition_count") or 0))
    return real[0]


def english_isbn13(doc):
    """An ISBN-13 to use as the join key. 978-0/1 are English-language blocks."""
    isbns = doc.get("isbn") or []
    for pref in ("9780", "9781", "978"):
        for i in isbns:
            if i.startswith(pref):
                return i
    return isbns[0] if isbns else None


def resolve(title):
    docs = search(title)
    doc = pick(docs)
    if not doc:
        return {"query": title, "status": "NO_MATCH"}

    cover_id = doc["cover_i"]
    size = cover_size(cover_id)
    if size < 8000:
        return {"query": title, "status": "WEAK_COVER",
                "matched": doc.get("title"), "cover_id": cover_id, "kb": size // 1024}

    return {
        "query": title,
        "status": "OK",
        "matched": doc.get("title"),
        "author": (doc.get("author_name") or [None])[0],
        "year": doc.get("first_publish_year"),
        "isbn": english_isbn13(doc),
        "cover_id": cover_id,
        "editions": doc.get("edition_count"),
        "kb": size // 1024,
    }


if __name__ == "__main__":
    titles = [ln.strip() for ln in sys.stdin if ln.strip()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(resolve, titles):
            print(json.dumps(r))
