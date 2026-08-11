#!/usr/bin/env python3
"""
Download the images out of a freshly-fetched Notion page.

Notion serves page images from S3 behind a signature that expires after 300
seconds. That is the whole difficulty: by the time a note has been read and
converted, every image URL in it is already dead (403). So this has to run
*immediately* after the MCP fetch that produced the file, while the signatures
are still good.

    python3 tools/fetch_notion_images.py fetched.json --slug atomic-habits

Images land in images/books/<slug>-N.png in document order, and a manifest is
written next to them so notion_to_md.py can rewrite the Nth image link to the
Nth file. Order is the join -- the URLs carry no stable identity.

Exit code is non-zero if any download failed, which almost always means the
signatures expired and the page needs re-fetching.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

IMG = re.compile(r"https://prod-files-secure\.s3[^\s\)\"]+")
OUT_DIR = os.path.join("images", "books")


def urls_from(path):
    raw = open(path, encoding="utf-8").read()
    try:
        raw = json.loads(raw).get("text", raw)
    except json.JSONDecodeError:
        pass
    # De-duplicate while preserving document order.
    seen, ordered = set(), []
    for u in IMG.findall(raw):
        u = u.replace("\\u0026", "&").replace("\\/", "/")
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


MAX_WIDTH = 1400


def shrink(path, original):
    """
    Cap width at MAX_WIDTH. Notion screenshots come off retina displays at
    ~2800px and 2-3MB each, far more than a 740px content column needs.

    Two traps, both hit in practice:
      - sips re-encodes even when no resize is needed, which inflated a 24KB
        screenshot to 80KB. So check the width first.
      - sips' PNG encoder is worse than Notion's, so even a real downscale can
        come out larger (100KB -> 144KB). So keep the result only if it
        actually saved bytes, and restore the original otherwise.
    """
    try:
        probe = subprocess.run(["sips", "-g", "pixelWidth", path],
                               check=True, capture_output=True, text=True)
        if int(probe.stdout.strip().split()[-1]) <= MAX_WIDTH:
            return os.path.getsize(path) // 1024
        subprocess.run(["sips", "-Z", str(MAX_WIDTH), path],
                       check=True, capture_output=True)
        if os.path.getsize(path) >= len(original):
            with open(path, "wb") as f:
                f.write(original)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return os.path.getsize(path) // 1024


def download(job):
    idx, url, slug = job
    dest = os.path.join(OUT_DIR, f"{slug}-{idx}.png")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            data = r.read()
        if len(data) < 500:
            return idx, None, f"suspiciously small ({len(data)}b)"
        with open(dest, "wb") as f:
            f.write(data)
        before = len(data) // 1024
        after = shrink(dest, data)
        note = f"{before}KB" if after == before else f"{before}KB -> {after}KB"
        return idx, dest, note
    except Exception as e:
        return idx, None, str(e)[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fetched")
    ap.add_argument("--slug", required=True)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    urls = urls_from(a.fetched)
    if not urls:
        print("no images in this page")
        return 0

    jobs = [(i + 1, u, a.slug) for i, u in enumerate(urls)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = sorted(pool.map(download, jobs))

    manifest, failed, network = {}, 0, False
    for idx, dest, note in results:
        if dest:
            manifest[idx] = "/" + dest.replace(os.sep, "/")
            print(f"  [{idx:2}] {dest}  {note}")
        else:
            failed += 1
            # Distinguish "the signature died" from "the machine is offline".
            # Both fail every download, and blaming expiry when the network is
            # down sends you off re-fetching pages for no reason.
            if "nodename" in note or "Name or service" in note or "resolve" in note:
                network = True
            print(f"  [{idx:2}] FAILED  {note}")

    with open(os.path.join(OUT_DIR, f"{a.slug}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"{len(manifest)}/{len(urls)} downloaded")
    if network:
        print("DNS/network failure -- the signatures may still be good. "
              "Check connectivity and rerun before rescuing the page.",
              file=sys.stderr)
    elif failed:
        print("Signatures likely expired -- re-fetch the page and rerun "
              "within 300 seconds.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
