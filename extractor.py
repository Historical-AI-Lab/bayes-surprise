import json
import re
import sys
import posixpath
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

# We read the EPUB directly as a zip rather than via ebooklib: ebooklib's
# navigation parser raises IndexError on EPUBs whose <nav> lacks an
# epub:type="toc" attribute, which is common in the books we analyze.

# A section is "body-like" (narrative, not paratext) if its TOC label,
# heading, or filename begins with one of these. Trade-throughs (Part/
# Chapter dividers, Prologue, Epilogue) all qualify; front/back matter
# (Cover, Praise, Acknowledgments, Copyright, Excerpt, ...) does not.
BODY_PREFIX_RE = re.compile(
    r"^\s*(prologue|epilogue|part\b|chapter\b)", re.IGNORECASE
)


def parse_epub(epub_path):
    """Return an ordered list of section dicts, one per spine document.

    Each section: {id, label, href, heading, word_count, text}. The
    caller assigns the body/front/back `kind`; this function only
    extracts and preserves the EPUB's own structure.
    """
    with zipfile.ZipFile(epub_path) as zf:
        names = zf.namelist()

        # Adobe ADEPT and other DRM schemes encrypt the content documents;
        # extracting them yields binary garbage, so fail loudly instead.
        if "META-INF/encryption.xml" in names:
            sys.exit(
                f"{epub_path}: EPUB is DRM-protected "
                "(META-INF/encryption.xml present); cannot extract text. "
                "Use a DRM-free copy (e.g. from the Calibre library)."
            )

        # container.xml points to the OPF (package) file.
        container = BeautifulSoup(zf.read("META-INF/container.xml"), "xml")
        opf_path = container.find("rootfile")["full-path"]
        opf_dir = posixpath.dirname(opf_path)
        opf = BeautifulSoup(zf.read(opf_path), "xml")

        # manifest: id -> href; spine: ordered idrefs (reading order).
        manifest = {it["id"]: it["href"] for it in opf.find_all("item")}
        spine = opf.find("spine")

        # NCX TOC maps href -> human label ("Chapter One", "Prologue").
        # The narrative body is best identified by these labels.
        ncx_label = {}
        ncx_id = spine.get("toc")
        if ncx_id and ncx_id in manifest:
            ncx_path = posixpath.normpath(
                posixpath.join(opf_dir, manifest[ncx_id])
            )
            ncx = BeautifulSoup(zf.read(ncx_path), "xml")
            for nav in ncx.find_all("navPoint"):
                src = nav.find("content")["src"].split("#")[0]
                txt = nav.find("text")
                if txt:
                    ncx_label.setdefault(src, txt.get_text(strip=True))

        sections = []
        for itemref in spine.find_all("itemref"):
            href = manifest[itemref["idref"]]
            doc_path = posixpath.normpath(posixpath.join(opf_dir, href))
            soup = BeautifulSoup(zf.read(doc_path), "html.parser")
            head = soup.find(["h1", "h2", "h3"])
            text = soup.get_text()
            sections.append(
                {
                    "id": itemref["idref"],
                    "label": ncx_label.get(href),
                    "href": href,
                    "heading": head.get_text(strip=True) if head else None,
                    "word_count": len(text.split()),
                    "text": text,
                }
            )
    return sections


def classify(sections):
    """Tag each section kind = front | body | back, in place.

    Body = the contiguous span from the first body-like section through
    the last; everything before is front matter, after is back matter.
    Using a span (not a per-section test) keeps any unlabeled interior
    section, e.g. a stray divider, inside the narrative.
    """

    def body_like(s):
        for candidate in (s["label"], s["heading"], s["id"]):
            if candidate and BODY_PREFIX_RE.match(candidate):
                return True
        return False

    flags = [body_like(s) for s in sections]
    if any(flags):
        first = flags.index(True)
        last = len(flags) - 1 - flags[::-1].index(True)
    else:
        # No recognizable structure: treat everything as body rather
        # than silently discarding the whole book.
        first, last = 0, len(sections) - 1

    for i, s in enumerate(sections):
        s["kind"] = "body" if first <= i <= last else (
            "front" if i < first else "back"
        )


if len(sys.argv) < 2:
    sys.exit("usage: python extractor.py <book.epub> [output_basename]")

epub_path = Path(sys.argv[1]).expanduser()

if len(sys.argv) >= 3:
    base = Path(sys.argv[2]).expanduser()
else:
    # Plan-style names: strip everything but alphanumerics (e.g.
    # "She's Not Sorry - Mary Kubica" -> "ShesNotSorryMaryKubica").
    stem = re.sub(r"[^A-Za-z0-9]", "", epub_path.stem)
    base = Path("rawtexts") / stem

json_path = base.with_suffix(".json")
txt_path = base.with_suffix(".txt")

sections = parse_epub(str(epub_path))
classify(sections)

base.parent.mkdir(parents=True, exist_ok=True)

# Structured output: every section, with its kind, for the pipeline.
with open(json_path, "w") as f:
    json.dump(sections, f, ensure_ascii=False, indent=1)

# Flat body-only text for quick eyeballing (paratext omitted).
body = [s for s in sections if s["kind"] == "body"]
with open(txt_path, "w") as f:
    f.write("\n".join(s["text"] for s in body))

body_words = sum(s["word_count"] for s in body)
print(f"wrote {json_path} ({len(sections)} sections)")
print(f"wrote {txt_path} ({len(body)} body sections, {body_words} words)")
front = [s["id"] for s in sections if s["kind"] == "front"]
back = [s["id"] for s in sections if s["kind"] == "back"]
print(f"  front: {front}")
print(f"  back:  {back}")
