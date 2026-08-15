"""Vietnamese text normalization for TTS input: rewrites the written forms the
model was never trained to voice (digits, dates, clock times, version strings,
fractions, acronyms) into the words a speaker would actually say.

    "Ngày 23/8/2024 lúc 15h30, giá 1.250.000 tăng 12,5"
        -> "Ngày hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư lúc
            mười lăm giờ ba mươi phút, giá một triệu hai trăm năm mươi nghìn
            tăng mười hai phẩy năm"

Provenance
──────────
The expansion rules (Vietnamese number chunking, the mười/lăm/mốt/tư sandhi,
the time/date/version/fraction shapes) and ``data/abbreviations.txt`` are
adapted from soe-vinorm (https://github.com/vinhdq842/soe-vinorm), MIT —
see data/LICENSE.soe-vinorm. What is NOT carried over is its machinery: that
project tags every token with a CRF (sklearn-crfsuite + a pickled model) and
disambiguates acronyms with an ONNX scorer, both downloaded from the Hub on
first use. Neither runs here — this module is pure stdlib regex, so the webui's
ONNX-only, no-torch-anywhere backend stays that way and startup stays instant.

Scope — the deliberately small set of cases this covers
──────────────────────────────────────────────────────
    date       23/8/2024, 23-8-2024, 23.8.2024, 8/2024, "ngày 23/8"
    time       15h30, 15h, 15:30, 9g45, 15:30:20
    version    v1.2, 1.2.3   (3+ dot-separated groups, or a v-prefix)
    fraction   3/4           (when it isn't a date)
    percent    25%, 12,5 %   -> "... phần trăm"
    at         @             -> "a còng" (never inside an email — those are protected)
    abbrev     TP.HCM, ATM, UBND  -> dictionary lookup, uppercase forms only
    number     1.250.000, 12,5, -7, +3, 2+3, 5 * 4, 2^10
                 (integer, decimal, sign, thousand separators, arithmetic ops)

Everything else soe-vinorm handles — money, measurement units, Roman numerals,
ranges, scores, quarters, URLs/emails, letter-by-letter spelling of unknown
sequences — is intentionally left alone: those either need
the CRF's context to tag safely or were out of scope for this integration.
URLs and emails are actively *protected* (matched and skipped) so a normalizer
pass can never mangle one.

Ambiguity, without a tagger to resolve it
─────────────────────────────────────────
A CRF decides "3/4" by context; regexes cannot, so the choices are fixed and
documented at each pattern below. The two that matter:

  - ``D/M`` is a date only after a date cue word (ngày, mùng, mồng, hôm,
    sáng, ...); otherwise it is read as a fraction. So "ngày 3/4" -> "ngày ba
    tháng tư" but a bare "3/4" -> "ba trên bốn".
  - ``M/YYYY`` (4-digit second group in 1000-2999) is a month-year. Unlike
    soe-vinorm this emits the word "tháng" itself unless the text already has
    it, since here no tagger guarantees the cue word is there.

An acronym is only expanded when written in uppercase (``ATM``, ``TP.HCM``) and
at least two letters long — a lowercase dictionary key like "ca" or "tư" is an
ordinary Vietnamese word far more often than an acronym, and single letters are
pure noise. When the dictionary holds several readings for one acronym (BCS =
"ban cán sự" / "bao cao su"), soe-vinorm picks with its ONNX likelihood scorer;
here the first listed reading wins.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

__all__ = ["normalize_vi_text", "expand_number", "load_abbreviations"]

_DATA_DIR = Path(__file__).parent / "data"

# ── Vietnamese number words ──────────────────────────────────────────────────

_DIGIT = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
    ",": "phẩy",
}

# Position inside a 3-digit chunk: units, tens, hundreds.
_UNIT_SINGLE = ["", "mươi", "trăm"]

# Scale of each 3-digit chunk, from the least significant one up. A number with
# more chunks than this raises IndexError, which expand_number catches and falls
# back to reading the digits one by one.
_UNIT_TRIPLE = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ"]

# Arithmetic operators spoken inside a numeric expression ("2+3" -> "hai cộng
# ba"). '=' is deliberately absent: soe-vinorm does not speak it either.
_OP_WORDS = str.maketrans({"+": "cộng", "-": "trừ", "*": "nhân", "/": "chia", "^": "mũ"})


def _apply_sandhi(text: str) -> str:
    """Vietnamese number-pronunciation rules: 15 is "mười lăm" not "mười năm",
    21 "hai mươi mốt" not "hai mươi một", 24 "hai mươi tư", 104 "một trăm linh
    tư"."""
    return (
        text.replace("mười năm", "mười lăm")
        .replace("mươi năm", "mươi lăm")
        .replace("mươi bốn", "mươi tư")
        .replace("mươi một", "mươi mốt")
        .replace("linh bốn", "linh tư")
    )


def expand_digit(digits: str) -> str:
    """Read a run of characters one symbol at a time ("2024" -> "hai không hai
    bốn"). Used for the decimal tail and as the number fallback."""
    return " ".join(_DIGIT.get(c, c) for c in digits.replace(" ", ""))


def _split_chunks(number: str) -> list[str]:
    """Split a digit string into 3-digit chunks, most significant first, with a
    short leading chunk when the length isn't a multiple of 3."""
    chunks = [number[i:i + 3] for i in range(len(number) - 3, -1, -3)][::-1]
    if len(number) % 3:
        chunks = [number[: len(number) % 3]] + chunks
    return chunks


def _speak_chunk(chunk: str, scale_index: int) -> str:
    """One 3-digit chunk plus its scale word ("250" at scale 1 -> "hai trăm năm
    mươi nghìn"). An all-zero chunk is silent."""
    if chunk == "000":
        return ""

    result = ""
    pos = len(chunk) - 1
    while pos >= 0:
        if pos == len(chunk) - 1 and chunk[pos] == "0" and len(chunk) > 1:
            pass  # trailing zero: "hai mươi", not "hai mươi không"
        elif pos == len(chunk) - 2 and chunk[pos] in ("1", "0"):
            # Tens digit 1 -> "mười" (not "một mươi"); tens digit 0 -> "linh".
            if pos == 0 and chunk[pos] == "0":
                pass
            elif chunk[pos] == "1":
                result = (
                    f"mười {_DIGIT[chunk[pos + 1]]}" if chunk[pos + 1] != "0" else "mười"
                )
            else:
                result = "linh " + _DIGIT[chunk[pos + 1]] if chunk[pos + 1] != "0" else ""
        else:
            result = (
                _DIGIT[chunk[pos]]
                + " "
                + _UNIT_SINGLE[len(chunk) - pos - 1]
                + (" " + result if result else "")
            )
        pos -= 1

    if scale_index >= len(_UNIT_TRIPLE):
        raise IndexError("number is too large to speak")

    return " ".join([result.strip(), _UNIT_TRIPLE[scale_index]]).strip()


def expand_number(number: str) -> str:
    """Speak an integer, a decimal, or a small arithmetic expression.

    Handles the sign ("-7" -> "trừ bảy"), '.' as a thousands separator
    ("1.250.000"), ',' as the Vietnamese decimal comma ("12,5" -> "mười hai
    phẩy năm"), '.' as a decimal point when at most two digits follow it
    ("3.14" -> "ba chấm một bốn"), and operators between numbers ("2+3" ->
    "hai cộng ba"). Leading zeros are dropped; a number too large for the
    scale table falls back to digit-by-digit reading.
    """
    try:
        sign = ""
        if number[0] in ("-", "+"):
            sign = {"+": "cộng", "-": "trừ"}[number[0]]
            number = number[1:]

        while len(number) > 1 and number[0] == "0" and number[1].isdigit():
            number = number[1:]

        number = number.strip()

        # More than one number in the string -> an expression: speak each
        # number, then turn the operators between them into words.
        matches = re.findall(r"[-+]?[0-9.,]+", number)
        if len(matches) > 1 or (matches and matches[0] != number):
            return (
                re.sub(
                    r"\s*([-+]?[0-9.,]+)\s*",
                    lambda m: f" {expand_number(m.group(1))} ",
                    number,
                )
                .strip()
                .translate(_OP_WORDS)
            )

        number = re.sub(r"[^0-9.,]", "", number)

        decimal_part = ""
        if number.count(",") == 1:
            number = number.replace(".", "")
            decimal_part = f"phẩy {expand_digit(number.split(',')[-1])}"
            number = "".join(number.split(",")[:-1])
        elif number.count(".") == 1 and len(number[number.index("."):]) <= 3:
            number = number.replace(",", "")
            decimal_part = f"chấm {expand_digit(number.split('.')[-1])}"
            number = "".join(number.split(".")[:-1])
        else:
            number = number.replace(".", "")

        chunks = _split_chunks(number)
        parts = []
        for i, chunk in enumerate(chunks):
            spoken = _speak_chunk(chunk, len(chunks) - i - 1)
            if spoken:
                parts.append(spoken)

        return f"{sign} {_apply_sandhi(' '.join(parts))} {decimal_part}".strip()

    except IndexError:
        return expand_digit(number)


def _num(value: str) -> str:
    """expand_number for a plain integer group inside a date/time/version."""
    return expand_number(value)


def _month(value: str) -> str:
    """Month name. The fourth month is "tháng tư", never "tháng bốn" — the one
    place this module knowingly departs from soe-vinorm, which runs months
    through the plain cardinal reader."""
    return "tư" if value.lstrip("0") == "4" else expand_number(value)


# ── abbreviations ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_abbreviations() -> dict[str, list[str]]:
    """Parse data/abbreviations.txt ("ABBR:reading[,reading...]" per line) into
    {abbreviation: [reading, ...]}. Both the uppercase and lowercase spellings
    of each acronym are keys in the file; only the uppercase ones are ever used
    (see the module docstring)."""
    path = _DATA_DIR / "abbreviations.txt"
    table: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            abbr, readings = line.split(":", 1)
            table.setdefault(abbr, []).extend(readings.split(","))
    return table


def _expand_abbreviation(token: str) -> str | None:
    """Dictionary reading for an uppercase acronym, or None to leave it alone.

    Tries the token as written, then with dots stripped ("TP.HCM" -> "TPHCM"),
    then splits on dots and expands each part when every part is known
    ("T.Ư" -> "trung ương" only if both halves resolve). Unknown acronyms come
    back None — soe-vinorm would spell them out letter by letter, which this
    module does not do.
    """
    table = load_abbreviations()

    for key in (token, token.replace(".", "").replace("-", "")):
        if key in table:
            return table[key][0]

    parts = [p for p in re.split(r"[.\-]", token) if p]
    if len(parts) > 1 and all(len(p) >= 2 and p in table for p in parts):
        return " ".join(table[p][0] for p in parts)

    return None


# ── pattern scanner ──────────────────────────────────────────────────────────
# One ordered alternation over the whole text: the first alternative that
# matches at a position wins, so more specific shapes must come first (a full
# date before a month-year before a fraction). Each alternative uses distinctly
# named groups so a single callback can dispatch on whichever one fired.

_VN_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠ-Ỹ"
_VN_LOWER = "a-zàáâãèéêìíòóôõùúýăđĩũơưạ-ỹ"

# Cue words that force "D/M" to be read as a day-and-month rather than a
# fraction. Matched case-insensitively, immediately before the number.
_DATE_CUES = r"ngày|mùng|mồng|hôm|sáng|trưa|chiều|tối|đêm|từ|đến"

# Anything inside a URL or an email address is left untouched — normalizing
# there produces nonsense, and this module has no URL reader.
_PROTECTED_RE = re.compile(
    r"""(?:(?:https?|ftp)://\S+
        |www\.\S+
        |[\w.+-]+@[\w-]+(?:\.[\w-]+)+
        |\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|vn|io|edu|gov|info|dev|ai)\b(?:/\S*)?
        )""",
    re.VERBOSE | re.IGNORECASE,
)

_SCANNER = re.compile(
    rf"""
    # ── time: HH:MM:SS / HHhMMmSS ────────────────────────────────────────────
      (?<![\d:])(?P<t_h>[01]?\d|2[0-3])[:hg](?P<t_m>[0-5]?\d)[:mp](?P<t_s>[0-5]?\d)(?![\d:])
    # ── date: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY ─────────────────────────────
    | (?<![\d/.\-])(?P<d_d>0?[1-9]|[12]\d|3[01])(?P<d_sep>[/.\-])(?P<d_m>0?[1-9]|1[0-2])(?P=d_sep)(?P<d_y>[12]\d{{3}})(?![\d/.\-])
    # ── month-year: MM/YYYY, MM-YYYY ─────────────────────────────────────────
    | (?<![\d/.\-])(?P<my_m>0?[1-9]|1[0-2])[/\-](?P<my_y>1\d{{3}}|20\d{{2}}|21\d{{2}})(?![\d/.\-])
    # ── day-month: DD/MM after a date cue word only (else it's a fraction) ───
    | (?<=\b)(?P<dm_cue>(?i:{_DATE_CUES}))(?P<dm_gap>\s+)(?P<dm_d>0?[1-9]|[12]\d|3[01])[/\-](?P<dm_m>0?[1-9]|1[0-2])(?![\d/.\-])
    # ── time: HH:MM (2-digit minutes), HHhMM, HHgMM ──────────────────────────
    | (?<![\d:])(?P<hm_h>[01]?\d|2[0-3]):(?P<hm_m>[0-5]\d)(?![\d:])
    | (?<![\d:])(?P<hg_h>[01]?\d|2[0-3])[hg](?P<hg_m>[0-5]\d)(?![\dhg])
    # ── time: HH h  ("15h", "9g") ────────────────────────────────────────────
    | (?<![\d:])(?P<h_h>[01]?\d|2[0-3])[hg](?![\w])
    # ── version: v1.2 / V1.2.3, or a bare 3+ group dotted number ─────────────
    | (?<![\w.])(?P<vp>[vV])(?P<v_num>\d+(?:\.\d+)+)(?!\w|\.\d|,\d)
    | (?<![\w.,])(?P<v_bare>\d+(?:\.\d+){{2,}})(?!\w|\.\d|,\d)
    # ── fraction: a/b (dates already claimed above) ──────────────────────────
    | (?<![\w/.,])(?P<f_a>\d+)\s*/\s*(?P<f_b>\d+)(?![\w/.,])
    # ── percent: 25%, 12,5 % — BEFORE the number branch, or that claims the
    #    digits and leaves a bare '%' behind ───────────────────────────────────
    | (?<![\w.,])(?P<pct_num>[-+]?\d[\d.,]*?)\s*%(?!\w)
    # ── number: integer / decimal / thousand-separated / signed / arithmetic ─
    | (?<![\w.,])(?P<n_num>[-+]?\d[\d.,]*(?:\s*[*^+]\s*[-+]?\d[\d.,]*|\s+[-/]\s+[-+]?\d[\d.,]*|[*^]\s*[-+]?\d[\d.,]*)*)(?![.,]?\d)
    # ── abbreviation: uppercase acronym, optionally dotted ───────────────────
    | (?<![\w.])(?P<abbr>[{_VN_UPPER}][{_VN_UPPER}\d]+(?:\.[{_VN_UPPER}][{_VN_UPPER}\d]*)*)(?![{_VN_LOWER}\d])
    # ── at sign: read as a word. Email addresses never reach here — they are
    #    matched by _PROTECTED_RE and skipped ─────────────────────────────────
    | (?P<at>@)
    """,
    re.VERBOSE,
)

# A dotted number whose every group after the first is exactly 3 digits is a
# thousands-separated integer ("1.250.000"), not a version string.
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:\.\d{3})+$")

# Trailing sign/operator junk that the number alternative may have swallowed
# ("5." at the end of a sentence).
_NUM_TRAIL_RE = re.compile(r"[.,\s]+$")


def _speak_time(h: str, m: str | None = None, s: str | None = None) -> str:
    out = f"{_num(h)} giờ"
    # "17:00" is "mười bảy giờ" out loud, not "mười bảy giờ không phút" —
    # a whole hour drops its minutes unless the text also spelled out seconds.
    if m is not None and (s is not None or m.strip("0")):
        out += f" {_num(m)} phút"
    if s is not None:
        out += f" {_num(s)} giây"
    return out


def _replace(match: re.Match) -> str:
    """re.sub callback: expand the match, then keep it a separate word.

    A written form can sit flush against a unit or a letter ("1.250.000đ",
    "5G"); the spoken form must not ("một triệu ...đ"), or the tokenizer sees
    one impossible word. A space is inserted on whichever side abutted a
    letter/digit, and only when the match actually changed."""
    expanded = _expand_match(match)
    if expanded == match.group(0):
        return expanded

    text = match.string
    before = text[match.start() - 1] if match.start() else ""
    after = text[match.end():match.end() + 1]
    if before.isalnum() and not expanded[:1].isspace():
        expanded = " " + expanded
    if after.isalnum() and not expanded[-1:].isspace():
        expanded = expanded + " "
    return expanded


def _expand_match(match: re.Match) -> str:
    g = match.groupdict()

    if g["t_h"] is not None:
        return _speak_time(g["t_h"], g["t_m"], g["t_s"])

    if g["d_d"] is not None:
        return f"{_num(g['d_d'])} tháng {_month(g['d_m'])} năm {_num(g['d_y'])}"

    if g["my_m"] is not None:
        # Only add "tháng" when the text doesn't already say it right before.
        before = match.string[max(0, match.start() - 8):match.start()].lower()
        lead = "" if re.search(r"tháng\s*$", before) else "tháng "
        return f"{lead}{_month(g['my_m'])} năm {_num(g['my_y'])}"

    if g["dm_d"] is not None:
        return f"{g['dm_cue']}{g['dm_gap']}{_num(g['dm_d'])} tháng {_month(g['dm_m'])}"

    if g["hm_h"] is not None:
        return _speak_time(g["hm_h"], g["hm_m"])

    if g["hg_h"] is not None:
        return _speak_time(g["hg_h"], g["hg_m"])

    if g["h_h"] is not None:
        return _speak_time(g["h_h"])

    if g["v_num"] is not None:
        return g["vp"] + " " + " chấm ".join(_num(p) for p in g["v_num"].split("."))

    if g["v_bare"] is not None:
        raw = g["v_bare"]
        if _THOUSANDS_RE.match(raw):  # 1.250.000 — a number, not a version
            return expand_number(raw)
        return " chấm ".join(_num(p) for p in raw.split("."))

    if g["f_a"] is not None:
        return f"{_num(g['f_a'])} trên {_num(g['f_b'])}"

    if g["pct_num"] is not None:
        return f"{expand_number(g['pct_num'].rstrip('.,'))} phần trăm"

    if g["n_num"] is not None:
        raw = g["n_num"]
        trail = _NUM_TRAIL_RE.search(raw)
        suffix = ""
        if trail and not raw[trail.start():].strip(" ").rstrip(".,") :
            suffix = raw[trail.start():]
            raw = raw[:trail.start()]
        if not raw:
            return match.group(0)
        # A long separator-less digit run is an identifier, not a quantity — a
        # phone number, an account number, a citizen id. Reading it as one huge
        # cardinal ("chín trăm tám mươi bảy triệu...") is worse than leaving it
        # alone, and reading it digit by digit is the NDIG case this module
        # deliberately doesn't cover. soe-vinorm caps its own untagged fallback
        # the same way, at 8 digits.
        if re.fullmatch(r"[-+]?\d+", raw) and len(raw.lstrip("-+")) > 8:
            return match.group(0)
        # digit-letter-digit ("1m65", "3G4") is a compound written form — a
        # height, a model number — not a quantity that happens to touch a unit.
        # Speaking only its first half ("một m65") is worse than leaving it.
        if re.match(r"^[^\W\d_]\d", match.string[match.end():match.end() + 2]):
            return match.group(0)
        return expand_number(raw) + suffix

    if g["abbr"] is not None:
        return _expand_abbreviation(g["abbr"]) or g["abbr"]

    if g["at"] is not None:
        return "a còng"

    return match.group(0)


def normalize_vi_text(text: str) -> str:
    """Expand the supported non-standard forms in `text` into spoken Vietnamese.

    Punctuation, casing and every unsupported form are preserved verbatim, and
    URLs/emails are skipped wholesale, so this is safe to run over text that
    has already been punctuation-normalized (webui/text_chunking.py) and over
    text that contains none of these forms at all — it is then a no-op beyond
    Unicode NFC normalization.
    """
    if not text or not text.strip():
        return text

    # NFC first: the dictionary keys and the letter classes in the patterns are
    # precomposed, so decomposed input ("Ð" + combining marks) would miss.
    text = unicodedata.normalize("NFC", text)

    out: list[str] = []
    last = 0
    for protected in _PROTECTED_RE.finditer(text):
        out.append(_SCANNER.sub(_replace, text[last:protected.start()]))
        out.append(protected.group(0))
        last = protected.end()
    out.append(_SCANNER.sub(_replace, text[last:]))

    # Expansions leave double spaces where a compact form used to be.
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


if __name__ == "__main__":  # quick manual check: python -m text_norm.vi_normalizer
    import sys

    if len(sys.argv) > 1:
        print(normalize_vi_text(" ".join(sys.argv[1:])))
    else:
        for sample in [
            "Ngày 23/8/2024 lúc 15h30, giá là 1.250.000 đồng, tăng 12,5 điểm.",
            "Phiên bản v1.2.3 phát hành tháng 8/2024, còn 3/4 số máy chạy 1.2.3.",
            "UBND TP.HCM và ATM của NHNN, gửi mail tới abc@gmail.com nhé.",
            "Tính 2+3, rồi 10 - 4, rồi 2^10 và 6 * 7 = 42.",
        ]:
            print(f"{sample}\n  -> {normalize_vi_text(sample)}\n")
