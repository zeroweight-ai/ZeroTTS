"""Punctuation normalization + sentence segmentation/chunking for long-form
text-to-speech input.

No sentence-boundary-detection library is a project dependency (no nltk/
spacy/pysbd), so this uses a regex-based splitter, then greedily packs
sentences into chunks bounded by an estimated speaking duration.

Splitting falls back through a hierarchy so no single chunk ever exceeds
the character budget, however the user punctuates (or doesn't):
    1. sentence-ending punctuation (. ! ? …)
    2. commas, if a "sentence" is still too long
    3. word boundaries, if a comma-separated piece is still too long
    4. raw character slicing, as a last resort (e.g. one giant run-on word)
"""

from __future__ import annotations

import re

# Split after ., !, ?, or … followed by whitespace (or end of string), but
# not on a run of digits/abbreviation-like single letters (e.g. "3.5" or
# "Mr.") immediately before the dot — kept intentionally simple/heuristic.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")

# Split after a comma followed by whitespace, keeping the comma with the
# preceding text (used only when a whole "sentence" is over budget).
_COMMA_RE = re.compile(r"(?<=,)\s+")

# Rough average speaking rate used to size chunks by estimated duration
# rather than raw character count (TTS speaking rate varies by language,
# so this is a coarse heuristic, not a precise duration predictor).
_CHARS_PER_SEC = 15.0

# ── punctuation normalization ────────────────────────────────────────────────
# Written text carries breaks the model was never trained to voice. The BPE
# vocab has a token for ';', but the training corpus is transcribed speech,
# where a semicolon is rare — the model has no reliable prosody for it, and at
# inference it tends to read as a hard stop or as nothing at all. A comma is
# the break it actually stands for out loud, so it is rewritten to one. ':' is
# left as-is (kept, not folded to a comma). Newlines are worse: the BPE
# normalizer collapses all whitespace to a single space
# (zerotts.tokenizer.normalize_text), so a line break vanishes entirely
# and two unrelated lines run together as one breath. Turning it into a
# sentence stop preserves the pause the layout meant — unless the line already
# ends in punctuation, which already carries it.

_PAUSE_PUNCT_RE = re.compile(r"[;]")

# A newline (plus any surrounding horizontal whitespace, and any run of blank
# lines) with what immediately precedes it captured, so the replacement can look
# at whether that character is already punctuation.
_NEWLINE_RE = re.compile(r"(.?)[ \t]*(?:\r?\n[ \t]*)+")

# Characters that already terminate or pause a clause — a line ending in one of
# these needs no added '.'.
_EXISTING_PUNCT = set(".!?…,;:—–-\"')]}")


def normalize_punctuation(text: str) -> str:
    """Rewrite written-only punctuation into the breaks the model can voice.

        ';' -> ','
        newline -> '. ' when the line does not already end in punctuation,
                   otherwise just a space.

    Semicolons are converted FIRST, so a line ending in one counts as
    already-punctuated when the newline rule runs (it ends in ',' by then)
    and does not also collect a '.'. ':' is untouched by this step, but is
    already in _EXISTING_PUNCT so a line ending in one is also treated as
    already-punctuated.
    """
    if not text:
        return text

    text = _PAUSE_PUNCT_RE.sub(",", text)

    def _repl(m: re.Match) -> str:
        prev = m.group(1)
        if not prev:  # leading newline(s) — nothing before them to punctuate
            return ""
        if prev in _EXISTING_PUNCT:
            return f"{prev} "
        return f"{prev}. "

    text = _NEWLINE_RE.sub(_repl, text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Split paragraphs first so blank lines always force a break, then
    # sentence-split within each paragraph.
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE_END_RE.split(para):
            sent = sent.strip()
            if sent:
                sentences.append(sent)
    return sentences


def _pack_pieces(pieces: list[str], max_chars: int) -> list[str]:
    """Greedily join `pieces` (each already <= max_chars) with a single
    space, filling each output chunk as close to max_chars as possible
    without exceeding it."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        piece_len = len(piece) + (1 if current else 0)
        if current and current_len + piece_len > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
            piece_len = len(piece)
        current.append(piece)
        current_len += piece_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_by_chars(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _split_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    pieces = _pack_pieces(words, max_chars)
    # A single word longer than max_chars (e.g. a spammed run of characters
    # with no spaces) still needs a hard character split.
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            out.append(piece)
        else:
            out.extend(_split_by_chars(piece, max_chars))
    return out


def _atomize(text: str, max_chars: int) -> list[str]:
    """Break `text` down until every returned piece is <= max_chars,
    preferring the least disruptive split available (comma > word >
    character)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    comma_parts = [p.strip() for p in _COMMA_RE.split(text) if p.strip()]
    if len(comma_parts) > 1:
        out: list[str] = []
        for part in comma_parts:
            out.extend(_atomize(part, max_chars))
        return out

    if " " in text.strip():
        return _split_by_words(text, max_chars)

    return _split_by_chars(text, max_chars)


def chunk_text(text: str, max_chunk_sec: float = 15.0) -> list[str]:
    """Split `text` into speakable chunks whose estimated duration stays
    near `max_chunk_sec`, preferring to break on sentence-ending
    punctuation, then commas, then words, then raw characters as a last
    resort — so a single chunk never balloons past the budget no matter how
    the input is (or isn't) punctuated.
    """
    max_chars = max(1, int(max_chunk_sec * _CHARS_PER_SEC))
    sentences = split_sentences(text)

    atoms: list[str] = []
    for sent in sentences:
        atoms.extend(_atomize(sent, max_chars))

    return _pack_pieces(atoms, max_chars)


# ── per-segment punctuation cleanup ──────────────────────────────────────────
# Chunking (above) can cut a sentence mid-flow, leaving a segment that ends on
# a comma or nothing at all, and/or a stray '.'/'!'/'?'/other symbol stranded
# in the middle of what is now a shorter, standalone utterance. Interior
# punctuation reads as a hard stop (or gets silently dropped by the tokenizer)
# where only a pause belongs, so every mid-segment punctuation mark other than
# a hyphen — word-joining ('twenty-one') or otherwise —, a slash, or a dot is
# downgraded to a comma-pause. The segment's own end is where a real stop belongs: whatever
# terminal mark it already had is kept if it's one of '.', '!', '?' (that's
# real, meaningful prosody), and anything else trailing there — comma, dash,
# stray symbol, nothing — is normalized to a single '.'.

# Trailing run of whitespace/punctuation, i.e. everything after the last
# alphanumeric character in the segment.
_TRAILING_RE = re.compile(r"[^\w]+$", re.UNICODE)

# Any mid-segment character that is not a word char, whitespace, hyphen,
# slash, dot, colon, question mark, exclamation mark, quote (single or
# double), percent sign, or comma itself gets folded into a comma-pause.
_MID_PUNCT_RE = re.compile(r"[^\w\s\-/.,:?@!\"'%]", re.UNICODE)

# Collapse runs of commas (possibly separated by spaces) into a single
# ', ' — also fixes the missing space left when a punctuation mark abutting
# a word (e.g. the '"' in 'Hello"world') is folded straight into a comma.
_REPEAT_COMMA_RE = re.compile(r"\s*(?:,\s*)+")

_END_PUNCT = {".", "!", "?"}


def clean_segment_punctuation(text: str) -> str:
    """Normalize a single chunked segment's punctuation for TTS:

        1. every mid-segment punctuation mark other than
           '-', '/', '.', ':', '?', '!', '"', "'", '%' becomes a ',' pause
        2. terminal mark: kept as-is if it's '.', '!', or '?', otherwise
           whatever trails (comma, dash, stray symbol, nothing) -> '.'
    """
    text = text.strip()
    if not text:
        return text

    m = _TRAILING_RE.search(text)
    core = text[:m.start()] if m else text
    trailing = text[m.start():].rstrip() if m else ""

    end_punct = trailing[-1] if trailing and trailing[-1] in _END_PUNCT else "."

    core = _MID_PUNCT_RE.sub(",", core)
    core = _REPEAT_COMMA_RE.sub(", ", core).strip(" ,")

    if not core:
        return ""
    return core + end_punct


# ── sample text file loading ─────────────────────────────────────────────────
# Used by the Gradio demo's "Sample texts" picker.

def load_text_samples(path) -> dict[str, str]:
    """Parse a "### name" delimited sample-text file (see webui/test_samples.txt's
    header for the format) into {name: text}. Missing file -> {}."""
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return {}

    samples: dict[str, str] = {}
    name = None
    lines: list[str] = []

    def flush():
        if name is not None:
            samples[name] = "\n".join(lines).strip()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("### "):
            flush()
            name = raw_line[4:].strip()
            lines = []
        elif raw_line.startswith("#"):
            continue
        else:
            lines.append(raw_line)
    flush()
    return samples
