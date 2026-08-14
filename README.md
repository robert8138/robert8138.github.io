# robert8138.github.io

Robert Chang's personal site — a Jekyll blog with a bookshelf of reading notes and an
unlinked Traditional Chinese mirror of the writing.

**Live:** <https://robert8138.github.io>

Built with Jekyll 4 and deployed by GitHub Actions (`.github/workflows/pages.yml`) on every
push to `master`. `_site/` is build output and is gitignored — never commit it.

---

## Local development

`bundle exec` does not work on the author's machine (`/usr/bin/bundle` resolves to system
Ruby 2.6 and dies in `activate_bin_path`). Use the Homebrew Ruby that already has Jekyll
installed as a gem, and call `jekyll` directly:

```sh
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH"

jekyll serve --port 4111   # http://localhost:4111
jekyll build               # writes _site/
```

Dependencies are declared in `Gemfile` (`jekyll ~> 4.3`, `webrick`, and `jekyll-feed`); CI
installs them with `bundler-cache` and builds with `bundle exec jekyll build`, which works
fine there. The Sass build prints a pile of `slash-div` deprecation warnings from
`_sass/_base.scss` and `_sass/_layout.scss` — noise, not failures.

---

## Repository layout

| Path | What's in it |
| --- | --- |
| `index.html` | English home page: post list, filtered to `lang: en` |
| `zh/` | Chinese home page (`/zh/`) and about page (`/zh/about/`) |
| `about.md`, `bookshelf.html`, `feed.xml` | The other top-level pages; `feed.xml` is hand-written and English-only |
| `_posts/` | Published English posts (3) |
| `_posts/zh/` | Chinese translations (3), served from `/zh/...` |
| `_posts_archive/` | Old 2015 posts (8), kept for reference. The leading underscore means Jekyll ignores the directory, so nothing here is published |
| `_drafts/` | Work in progress. Gitignored — drafts stay local |
| `_books/` | One Markdown file per book note (28), output at `/books/<slug>/` |
| `_data/` | `bookshelf.yml` (shelves and their books), `i18n.yml` (UI strings per language), `settings.yml` (presentations list) |
| `_layouts/` | `default`, `page`, `post`, `book`, `bookshelf` |
| `_includes/` | `head`, `header` (nav), `book-cover`, `comments` (Disqus), `google_analytics` |
| `_sass/` | `_base`, `_layout`, `_bookshelf`, `_syntax-highlighting`, `_cjk` (CJK typography, scoped to `html[lang^="zh"]`) |
| `css/main.scss` | The only stylesheet entry point; imports the partials above |
| `images/` | Post images at the top level; `images/books/` holds diagrams pulled out of book notes |
| `presentations/` | Two self-contained slide decks, served as-is |
| `tools/` | Python helpers for the book-note pipeline (see below) |

---

## Adding a blog post

Create `_posts/YYYY-MM-DD-slug.md`. The URL follows Jekyll's default
`/:year/:month/:day/:title.html`.

```yaml
---
layout: post
title:  "Airbnb's Semantic Layer: Compute"
date:   2026-08-13 09:00:00 +0800
comments: True
ref: semantic-layer-compute
excerpt: "One or two sentences. Used on the home page, in <meta name=description>, and in the RSS feed."
---
```

- `ref` — a stable slug shared with the post's translation. Only needed if a Chinese version
  exists or is planned (see below).
- `comments` — set truthy to render the Disqus thread.
- `excerpt` — worth writing by hand; Jekyll otherwise falls back to the first paragraph.
- `lang: en` is applied automatically by the `defaults` block in `_config.yml`; don't set it.
- `_layouts/post.html` also renders optional `author` and `meta` fields in the byline.

Drafts go in `_drafts/`, which is gitignored, so nothing half-finished can be pushed by
accident. Serve with `jekyll serve --drafts` to preview them.

Nav tabs come from `nav_order` front matter on pages (`nav_title` overrides the label). Posts
don't participate.

---

## The Chinese mirror

The site has a Traditional Chinese mirror at `/zh/` that is **deliberately not linked from
anywhere on the English site**. Readers reach it by knowing the URL or by finding a post
through a Chinese-language search. Keep it that way when adding pages.

How it works:

1. **Location decides language.** `_config.yml` stamps every post `lang: en` by default, and
   everything under `_posts/zh` `lang: zh` with `permalink: /zh/:year/:month/:day/:title.html`.
   A translation therefore sits at the English URL with `/zh` prepended.
2. **`ref` pairs the two files.** Give the translation the same filename and the same `ref`
   as its original. `_includes/head.html` looks up all documents sharing a `ref` and emits
   `hreflang` tags (`en`, `zh-Hant`, plus `x-default` pointing at the English one) only when
   more than one exists. Nothing renders in the page body — the pairing is for search engines.
3. **The English side filters `zh` out.** `index.html` and `feed.xml` both select
   `where: "lang", "en"`, so translations never appear in the English post list or the RSS
   feed. `/zh/index.html` is the mirror, selecting `lang: zh`.
4. **The nav can't collide.** `index.html` and `zh/index.html` both carry `nav_order: 1`;
   `_includes/header.html` skips any page whose `lang` differs from the current page's, so
   only one ever renders. Pages with no `lang` (e.g. `bookshelf.html`) are shared across both
   languages and get their label from `nav_title_<lang>`.
5. **UI strings live in `_data/i18n.yml`**, keyed by language — home URL, section headings,
   and date format. Templates read `site.data.i18n[lang]` instead of hardcoding English.
   Post and page bodies are translated as whole files, not through this table.

To translate a post: copy `_posts/<name>.md` to `_posts/zh/<name>.md`, keep `ref` and `date`
identical, translate `title`, `excerpt`, and the body, and point in-body links at the `/zh/`
counterparts where they exist.

---

## Adding a book note

The bookshelf is two halves joined on `isbn`:

- `_data/bookshelf.yml` — the shelves and their arrangement. Each book entry has `title`,
  `author`, `isbn`, `cover_id`, and an optional external `link`. This file alone is enough to
  put a cover on `/bookshelf/`.
- `_books/<slug>.md` — the note. Its presence is what turns that cover into a link.

`bookshelf.html` does `site.books | where: "isbn", book.isbn | first` for every cover. If a
`_books/` file matches, the cover links to `/books/<slug>/` and gets a "Notes →" affordance;
otherwise it falls back to the entry's external `link`; otherwise it stays a plain,
unclickable cover. **The shelf entry's `isbn` and the note's `isbn` must match exactly, or
the link silently never appears.**

Front matter that `_layouts/book.html` reads:

```yaml
---
layout: book
title: "Deep Work"
author: "Cal Newport"
isbn: "9780349413686"          # join key to _data/bookshelf.yml
cover_id: "7988607"            # Open Library cover ID
one_liner: "The ability to perform deep work is becoming increasingly rare at exactly the same time it is becoming increasingly valuable."
read: 2020                     # year, shown in the facts list
link: https://...              # optional "Find this book →"
---
```

The shelf name shown on a book page is *not* front matter — `_layouts/book.html` looks it up
by scanning `_data/bookshelf.yml` for the matching `isbn`, so arrangement lives in exactly one
place.

### Covers

`_includes/book-cover.html` pulls covers from Open Library and prefers `cover_id`
(`/b/id/<id>-L.jpg`) over `isbn` (`/b/isbn/<isbn>-L.jpg`), because an ISBN often has no scan
or resolves to a foreign-language or audiobook jacket. `?default=false` makes the API 404
instead of returning a blank pixel, which triggers the `onerror` handler and reveals a
typeset title/author fallback.

### Tooling

Notes are synced out of a Notion "Book Summary" database; Notion is the source of truth, so
edit there and re-sync rather than hand-editing `_books/*.md`. Each file records its
`notion_url` in front matter. The helpers, all with usage in their docstrings:

| Script | Job |
| --- | --- |
| `tools/resolve_isbn.py` | Search Open Library by title + author for an English ISBN-13 (identity) and a `cover_i` (cover). Fuzzy-matches, so always eyeball the result |
| `tools/fetch_notion_images.py` | Download a fetched Notion page's images into `images/books/<slug>-N.png`. Must run immediately after the fetch — Notion's S3 URLs are signed for 300 seconds |
| `tools/notion_to_md.py` | Convert the fetched Notion JSON into a `_books/*.md` file (heading shift, callouts, image rewrites) |

---

## Notes and rough edges

- **`_data/settings.yml` has `presentations: []`**, and `index.html` renders the Presentations
  section only when that list is non-empty. The two decks in `presentations/` are published
  but currently unlinked.
- **`_includes/google_analytics.html` uses a Universal Analytics `UA-` property**, which no
  longer collects data.
- **`_includes/comments.html` interpolates `site.disqusid`**, which is not defined in
  `_config.yml`, so it renders as empty in the Disqus identifier.
- **`jekyll-feed` is in the Gemfile** but the site ships a hand-written `feed.xml`; the
  hand-written one is what gets served.
- **`_config.yml` has no `exclude` list**, so `README.md` and `tools/*.py` are copied into the
  built site.
