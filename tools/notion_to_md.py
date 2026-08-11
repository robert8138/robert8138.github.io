#!/usr/bin/env python3
"""
Convert a Notion "Book Summary" page into a Jekyll _books/*.md file.

Input is the JSON blob the Notion MCP `fetch` tool returns. Large pages get
written to a file by the harness instead of returned inline, and that file is
exactly what this reads -- so long notes never have to pass through context.

    python3 tools/notion_to_md.py fetched.json \
        --isbn 9780735211308 --cover-id 12539702 --author "James Clear" \
        --out _books/atomic-habits.md

What it has to fix, and why:

  headings      Notion pages start at h1, but the book page layout already
                renders the title as h1, so everything shifts down one.
  callouts      <callout> is Notion's highlight box. A blockquote is the
                closest thing the site's stylesheet has.
  colors        {color="green"} / <span color="red"> mark "what I do well"
                against "areas of improvement". The section headings already
                say that, so the markers are dropped rather than translated.
  highlights    <span color="yellow_bg">x</span> becomes bold.
  images        Notion signs S3 URLs for 300 seconds, so every link is dead
                by the time it gets here. Run tools/fetch_notion_images.py
                immediately after the fetch, then pass --slug to rewrite the
                links to those local copies.
  indentation   Notion nests with tabs; markdown wants two spaces per level.
"""

import argparse
import html
import json
import os
import re
import sys

# Book pages in _books/, keyed by the Notion page id they were synced from, so
# cross-references between notes become working internal links.
KNOWN_PAGES = {
    "a71fe00e0c8f4ebbb858cb5208a37e52": ("Deep Work", "/books/deep-work/"),
    "b64e7623203b46628ccf1fca620b3cc5": ("So Good They Can't Ignore You",
                                         "/books/so-good-they-cant-ignore-you/"),
    "094e4b6c49bb43daa5112ae3c81adb5c": ("The One Thing", "/books/the-one-thing/"),
    "211c967518b04318bd928e30cf0117fc": ("Ultralearning", "/books/ultralearning/"),
    "6665df5ead6f4e6da1d0c097050921b9": ("Atomic Habits", "/books/atomic-habits/"),
}


def load(path):
    raw = open(path, encoding="utf-8").read().strip()
    # The tool returns JSON; a hand-saved page might just be the text.
    try:
        blob = json.loads(raw)
        return blob.get("text", raw), blob.get("title", "")
    except json.JSONDecodeError:
        return raw, ""


def properties(text):
    m = re.search(r"<properties>\s*(\{.*?\})\s*</properties>", text, re.S)
    return json.loads(m.group(1)) if m else {}


def body(text):
    m = re.search(r"<content>\n?(.*)\n?</content>", text, re.S)
    return m.group(1) if m else text


def demote_headings(line):
    m = re.match(r"^(#{1,5}) +(.*)$", line)
    if not m:
        return line
    return "#" * (len(m.group(1)) + 1) + " " + m.group(2).strip()


def inline(s):
    """Clean up Notion's inline markup."""
    # Highlighted spans read as emphasis.
    s = re.sub(r'<span color="[a-z_]*_bg">(.*?)</span>', r"**\1**", s, flags=re.S)
    # Plain coloured spans carry no meaning once the section heading says it.
    s = re.sub(r'<span color="[a-z_]+">(.*?)</span>', r"\1", s, flags=re.S)
    s = re.sub(r'\s*\{color="[a-z_]+"\}', "", s)

    # Cross-references to other book notes.
    def mention(m):
        pid = m.group(1).rstrip("/").split("/")[-1].split("?")[0].replace("-", "")
        if pid in KNOWN_PAGES:
            title, url = KNOWN_PAGES[pid]
            return f'[{title}]({{{{ "{url}" | prepend: site.baseurl }}}})'
        return "a related note"

    s = re.sub(r'<mention-page url="([^"]+)"\s*/>', mention, s)


    # Collapse doubled emphasis left by Notion's bold-with-trailing-space habit
    # (`**text **` renders literally in kramdown).
    s = re.sub(r"\*\*(\s*)([^*]+?)(\s*)\*\*", lambda m: m.group(1) + "**" + m.group(2).strip() + "**" + m.group(3), s)
    s = s.replace("****", "")
    s = s.replace("\\>", ">")
    return s


def swap_images(src, manifest, alt):
    """
    Point image links at the local copies fetch_notion_images.py saved.

    The join is document order -- the Nth S3 link becomes the Nth entry in the
    manifest -- because the signed URLs carry no stable identity. Anything with
    no local copy becomes a visible TODO rather than a silently broken image.
    """
    counter = {"n": 0}

    def repl(_m):
        counter["n"] += 1
        n = counter["n"]
        path = manifest.get(str(n)) or manifest.get(n)
        if not path:
            return ("<!-- TODO image %d: no local copy. Re-fetch the page and run "
                    "tools/fetch_notion_images.py within 300s. -->" % n)
        return f'![{alt} — figure {n}]({{{{ "{path}" | prepend: site.baseurl }}}})'

    return re.sub(r"!\[\]\(https://prod-files-secure\.s3[^)]*\)", repl, src)


def convert(text, manifest=None, alt="Figure"):
    src = body(text)
    src = swap_images(src, manifest or {}, alt)

    # Structural tags that have no markdown equivalent.
    src = re.sub(r"<table_of_contents[^>]*/>\n?", "", src)
    src = re.sub(r"</?columns?(_list)?>\n?", "", src)
    src = re.sub(r'<column ratio="[^"]*">\n?', "", src)
    src = re.sub(r"</column>\n?", "", src)
    src = src.replace("<empty-block/>", "")

    # Callouts -> blockquotes. Inner <br><br> are paragraph breaks.
    def callout(m):
        inner = m.group(1)
        inner = re.sub(r"<br\s*/?>\s*<br\s*/?>", "\n\n", inner)
        inner = re.sub(r"<br\s*/?>", "\n\n", inner)
        inner = "\n".join(ln.strip() for ln in inner.strip().split("\n"))
        inner = inline(inner)
        quoted = "\n".join(
            ("> " + ln) if ln.strip() else ">" for ln in inner.split("\n")
        )
        return quoted + "\n"

    src = re.sub(r"<callout[^>]*>\n?(.*?)</callout>", callout, src, flags=re.S)

    out = []
    for raw in src.split("\n"):
        line = raw.replace("<br>", " ").replace("<br/>", " ")

        # Tab indentation -> two spaces per level, keeping list markers.
        depth = len(line) - len(line.lstrip("\t"))
        line = "  " * depth + line.lstrip("\t")

        stripped = line.strip()
        if stripped == "---":          # Notion divider under a heading
            continue
        if not stripped:
            out.append("")
            continue

        line = demote_headings(line) if line.lstrip().startswith("#") else line

        # Images inherited their indentation from the Notion column they sat
        # in. Four or more leading spaces is a code block in kramdown, so any
        # standalone image has to sit flush left.
        if stripped.startswith("!["):
            line = stripped

        out.append(inline(line).rstrip())

    # Squeeze runs of blank lines.
    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip()
    return html.unescape(text_out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fetched")
    ap.add_argument("--isbn", required=True)
    ap.add_argument("--cover-id", required=True)
    ap.add_argument("--author", required=True)
    ap.add_argument("--title")
    ap.add_argument("--one-liner", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", help="image manifest slug from fetch_notion_images.py")
    a = ap.parse_args()

    manifest = {}
    if a.slug:
        mpath = os.path.join("images", "books", f"{a.slug}.manifest.json")
        if os.path.exists(mpath):
            manifest = json.load(open(mpath))

    text, _ = load(a.fetched)
    props = properties(text)
    title = a.title or props.get("Name") or "Untitled"
    created = (props.get("Created") or "")[:4]
    notion_url = props.get("url", "")

    fm = [
        "---",
        "layout: book",
        f'title: "{title}"',
        f'author: "{a.author}"',
        f'isbn: "{a.isbn}"',
        f'cover_id: "{a.cover_id}"',
    ]
    if a.one_liner:
        fm.append(f'one_liner: "{a.one_liner}"')
    if created:
        fm.append(f"read: {created}")
    if notion_url:
        fm.append(f"notion_url: {notion_url}")
    fm += [
        "---",
        "",
        "<!-- Synced from Notion (Book Summary database). Notion is the source of truth. -->",
        "",
        "",
    ]

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(fm) + convert(text, manifest, title))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
