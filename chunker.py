import json
import re
import sys
from pathlib import Path

import nltk

# Re-segment a rawtexts/<Book>.json (spine-ordered sections from
# extractor.py) into ~1000-word chunks for the Bayesian-surprise pipeline.
# Spec: rawtexts/chunker_spec.md.

TARGET = 1000
MIN_WORDS = 800
MAX_WORDS = 1200

# A part-divider stub: "Part One"/"Part Two" structural marker, ~3 words,
# not prose. Matched on id/label/heading AND guarded by word count so a
# real section that merely starts with "Part" isn't discarded.
PART_RE = re.compile(r"^\s*part\b", re.IGNORECASE)
PART_STUB_MAX_WORDS = 5


def sent_tokenize(text):
    """nltk Punkt sentence split, downloading the model once if absent.

    After the one-time download the data is cached locally, so later runs
    work offline.
    """
    try:
        return nltk.tokenize.sent_tokenize(text)
    except LookupError:
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        return nltk.tokenize.sent_tokenize(text)


def is_part_stub(s):
    text_fields = (s.get("id"), s.get("label"), s.get("heading"))
    if s.get("word_count", 0) > PART_STUB_MAX_WORDS:
        return False
    return any(f and PART_RE.match(f) for f in text_fields)


def strip_heading(text, section):
    """Drop the leading heading echo from a section's flat text.

    extractor.py's get_text() prepends the chapter heading (often twice,
    e.g. "\\n\\nOne\\n\\n\\n\\n\\n\\nOne\\n"). Split on newlines and discard
    leading lines that are blank or equal (case-insensitively) the
    section's heading / label / id, then return the remaining paragraphs.
    """
    norm = lambda x: re.sub(r"[_\-]", " ", x).strip().casefold() if x else None
    headers = {norm(section.get(k)) for k in ("heading", "label", "id")}
    headers.discard(None)

    paras = [p.strip() for p in text.split("\n")]
    i = 0
    while i < len(paras) and (not paras[i] or paras[i].casefold() in headers):
        i += 1
    return [p for p in paras[i:] if p]


def build_units(sections):
    """Flatten selected body sections into an ordered sentence stream.

    Each unit: {text, words, chapter_index, chapter_id, para_end,
    chapter_end}. Operating on one ordered stream (section boundaries as
    hints, not hard one-section-per-chapter assumptions) keeps the chunker
    correct on EPUBs that ship the whole novel as one giant document; there
    chapter metadata is just coarser.
    """
    body = [s for s in sections if s.get("kind") == "body"]
    chapters = [s for s in body if not is_part_stub(s)]

    units = []
    chapter_meta = []
    for ci, s in enumerate(chapters):
        chapter_meta.append((ci, s["id"]))
        paras = strip_heading(s["text"], s)
        sec_units = []
        for p in paras:
            sents = sent_tokenize(p)
            for j, sent in enumerate(sents):
                sec_units.append(
                    {
                        "text": sent,
                        "words": len(sent.split()),
                        "chapter_index": ci,
                        "chapter_id": s["id"],
                        "para_end": j == len(sents) - 1,
                        "chapter_end": False,
                    }
                )
        if sec_units:
            sec_units[-1]["chapter_end"] = True
            sec_units[-1]["para_end"] = True
        units.extend(sec_units)
    return units, chapter_meta


def words_to_next_boundary(units, start):
    """Words from `start` up to and including the next chapter_end."""
    total = 0
    for u in units[start:]:
        total += u["words"]
        if u["chapter_end"]:
            break
    return total


def chunk(units):
    """Greedy fill with chapter-boundary steering (spec section 4)."""
    chunks = []
    i = 0
    n = len(units)
    while i < n:
        # Steering: when starting a chunk at a chapter boundary, shrink the
        # provisional target so an integer number of chunks lands exactly on
        # the next chapter end (spec's "400 + 4400 -> ~960" trick).
        target = TARGET
        remaining = words_to_next_boundary(units, i)
        k = max(1, round(remaining / TARGET))
        if k:
            cand = remaining / k
            if MIN_WORDS <= cand <= MAX_WORDS:
                target = cand

        cur_words = 0
        start = i
        while i < n:
            u = units[i]
            # Cut before overflowing the ceiling, once the floor is met.
            # (When the chunk is still empty, a lone >MAX sentence is added
            # anyway and flagged below — never split mid-sentence.)
            if cur_words >= MIN_WORDS and cur_words + u["words"] > MAX_WORDS:
                break
            cur_words += u["words"]
            i += 1

            if i >= n:
                break  # end of book: emit whatever is left
            if cur_words < MIN_WORDS:
                continue

            nxt = units[i]["words"]
            if u["chapter_end"]:
                break  # good-sized chapter end: use it (criterion 4)
            if not u["para_end"]:
                continue  # only cut on paragraph boundaries when we can
            if cur_words >= target:
                break
            if abs(cur_words + nxt - target) > abs(cur_words - target):
                break  # next unit moves us away from target

        seg = units[start:i]
        if seg[-1]["words"] > MAX_WORDS and len(seg) == 1:
            print(
                f"warning: chunk {len(chunks)} is a single "
                f"{seg[-1]['words']}-word unit (unsplittable sentence)",
                file=sys.stderr,
            )

        idxs, ids = [], []
        for u in seg:
            if u["chapter_index"] not in idxs:
                idxs.append(u["chapter_index"])
                ids.append(u["chapter_id"])
        text = " ".join(u["text"] for u in seg)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "chapter_indices": idxs,
                "chapter_ids": ids,
                "is_surprising": False,
                "word_count": len(text.split()),
                "text": text,
            }
        )
    return chunks


WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def chapter_ordinal(chapter_id):
    """Rough ordinal for a chapter id, for the monotonicity sanity check."""
    cid = chapter_id.strip().casefold()
    if cid.startswith("prologue"):
        return -1.0
    if cid.startswith("epilogue"):
        return float("inf")
    total = 0
    found = False
    for part in re.split(r"[\s_\-]+", cid):
        if part in WORD_NUM:
            total += WORD_NUM[part]
            found = True
    return float(total) if found else None


def sanity_check_order(chapter_meta):
    prev = None
    prev_id = None
    for _, cid in chapter_meta:
        ordn = chapter_ordinal(cid)
        if ordn is None or prev is None:
            prev, prev_id = ordn, cid
            continue
        if ordn < prev:
            print(
                f"warning: chapter order looks wrong: '{cid}' "
                f"follows '{prev_id}' (spine alphabetized?)",
                file=sys.stderr,
            )
        prev, prev_id = ordn, cid


if len(sys.argv) < 2:
    sys.exit("usage: python chunker.py <rawtexts/Book.json> [output_basename]")

in_path = Path(sys.argv[1]).expanduser()
if len(sys.argv) >= 3:
    out_path = Path(sys.argv[2]).expanduser().with_suffix(".json")
else:
    out_path = Path("chunkedtexts") / (in_path.stem + ".json")

with open(in_path) as f:
    sections = json.load(f)

units, chapter_meta = build_units(sections)
if not units:
    sys.exit(f"{in_path}: no body text found")
sanity_check_order(chapter_meta)
chunks = chunk(units)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=1)

wcs = [c["word_count"] for c in chunks]
out_of_band = sum(
    1 for c in chunks[:-1] if not (MIN_WORDS <= c["word_count"] <= MAX_WORDS)
)
print(f"read  {in_path}")
print(f"wrote {out_path} ({len(chunks)} chunks)")
print(
    f"  words: min {min(wcs)}, mean {sum(wcs) // len(wcs)}, max {max(wcs)}; "
    f"{out_of_band} non-final chunks outside [{MIN_WORDS}, {MAX_WORDS}]"
)
print(f"  chapters covered: {len(chapter_meta)}")
