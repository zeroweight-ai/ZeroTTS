"""Case-by-case checks for text_norm/vi_normalizer.py — one section per
supported form, plus a section for the forms that must be left ALONE (money,
units, percentages, Roman numerals, ranges, URLs, English text), which is where
a regex normalizer with no tagger is most likely to do damage.

Run:  pytest tests/test_vi_normalizer.py   (or run this file directly)
"""

from __future__ import annotations

import pytest

from zerotts.text_norm import normalize_vi_text as N

# (input, expected output)
CASES: list[tuple[str, str, str]] = [
    # ── integers ─────────────────────────────────────────────────────────────
    ("number", "5", "năm"),
    ("number", "15", "mười lăm"),
    ("number", "21", "hai mươi mốt"),
    ("number", "24", "hai mươi tư"),
    ("number", "104", "một trăm linh tư"),
    ("number", "2024", "hai nghìn không trăm hai mươi tư"),
    ("number", "1000000", "một triệu"),
    # thousands separators
    ("number", "1.250.000", "một triệu hai trăm năm mươi nghìn"),
    ("number", "12.000", "mười hai nghìn"),
    # decimals: ',' is the Vietnamese decimal comma, '.' reads as "chấm"
    ("number", "12,5", "mười hai phẩy năm"),
    ("number", "3.14", "ba chấm một bốn"),
    # sign
    ("number", "-7", "trừ bảy"),
    ("number", "+3", "cộng ba"),
    # arithmetic operators
    ("number", "2+3", "hai cộng ba"),
    ("number", "2^10", "hai mũ mười"),
    ("number", "6 * 7", "sáu nhân bảy"),
    ("number", "10 - 4", "mười trừ bốn"),
    # long digit runs (ids/phones) stay put — NDIG is out of scope
    ("number", "0987654321", "0987654321"),

    # ── dates ────────────────────────────────────────────────────────────────
    ("date", "23/8/2024", "hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư"),
    ("date", "01-09-1945", "một tháng chín năm một nghìn chín trăm bốn mươi lăm"),
    ("date", "2.9.1945", "hai tháng chín năm một nghìn chín trăm bốn mươi lăm"),
    ("date", "tháng 8/2024", "tháng tám năm hai nghìn không trăm hai mươi tư"),
    ("date", "8/2024", "tháng tám năm hai nghìn không trăm hai mươi tư"),
    ("date", "ngày 3/4", "ngày ba tháng tư"),

    # ── times ────────────────────────────────────────────────────────────────
    ("time", "15h30", "mười lăm giờ ba mươi phút"),
    ("time", "15h", "mười lăm giờ"),
    ("time", "9g45", "chín giờ bốn mươi lăm phút"),
    ("time", "15:30", "mười lăm giờ ba mươi phút"),
    ("time", "15:30:20", "mười lăm giờ ba mươi phút hai mươi giây"),
    # a whole hour drops its minutes
    ("time", "17:00", "mười bảy giờ"),
    ("time", "17h00", "mười bảy giờ"),

    # ── versions ─────────────────────────────────────────────────────────────
    ("version", "1.2.3", "một chấm hai chấm ba"),
    ("version", "v1.2", "v một chấm hai"),
    ("version", "phiên bản 2.10.1.", "phiên bản hai chấm mười chấm một."),

    # ── fractions ────────────────────────────────────────────────────────────
    ("fraction", "3/4", "ba trên bốn"),
    ("fraction", "1/2", "một trên hai"),

    # ── abbreviations ────────────────────────────────────────────────────────
    ("abbrev", "ATM", "máy rút tiền tự động"),
    ("abbrev", "TP.HCM", "Thành phố Hồ Chí Minh"),
    ("abbrev", "UBND", "Ủy ban Nhân dân"),
    # lowercase dictionary keys are ordinary words far more often than acronyms
    ("abbrev", "ca", "ca"),
    ("abbrev", "tư", "tư"),
    # single uppercase letters are never expanded
    ("abbrev", "Q", "Q"),
    # unknown acronym: left as written, never spelled out letter by letter
    ("abbrev", "ZZZQ", "ZZZQ"),

    # ── out of scope: must pass through untouched ────────────────────────────
    ("untouched", "https://vnexpress.net/tin-tuc/12/3", "https://vnexpress.net/tin-tuc/12/3"),
    ("untouched", "abc.dev@gmail.com", "abc.dev@gmail.com"),
    ("untouched", "thế kỷ XXI", "thế kỷ XXI"),          # Roman numerals
    ("untouched", "Xin chào các bạn.", "Xin chào các bạn."),
    ("untouched", "Hello world!", "Hello world!"),

    # ── out of scope but containing an in-scope number ───────────────────────
    # Money and units are NOT expanded; the bare number inside them is.
    # (Percent used to live here; it is now handled in full — see "percent".)
    ("partial", "5 km/h", "năm km/h"),
    ("partial", "1.250.000đ", "một triệu hai trăm năm mươi nghìn đ"),
    ("partial", "5G", "năm G"),
    # digit-letter-digit compounds (a height, a model number) stay whole
    ("partial", "cao 1m65", "cao 1m65"),

    # ── sentences ────────────────────────────────────────────────────────────
    (
        "sentence",
        "Ngày 23/8/2024 lúc 15h30, giá là 1.250.000 đồng, tăng 12,5 điểm.",
        "Ngày hai mươi ba tháng tám năm hai nghìn không trăm hai mươi tư lúc "
        "mười lăm giờ ba mươi phút, giá là một triệu hai trăm năm mươi nghìn "
        "đồng, tăng mười hai phẩy năm điểm.",
    ),
    (
        "sentence",
        "UBND TP.HCM họp lúc 9h, xem tại https://hcmcpv.org.vn nhé.",
        "Ủy ban Nhân dân Thành phố Hồ Chí Minh họp lúc chín giờ, xem tại "
        "https://hcmcpv.org.vn nhé.",
    ),
    # ── percent ──────────────────────────────────────────────────────────────
    ("percent", "giảm 25%", "giảm hai mươi lăm phần trăm"),
    ("percent", "lãi suất 12,5% một năm", "lãi suất mười hai phẩy năm phần trăm một năm"),
    ("percent", "tăng 100%", "tăng một trăm phần trăm"),
    ("percent", "chiếm 7,5 % tổng số", "chiếm bảy phẩy năm phần trăm tổng số"),
    # ── at sign ──────────────────────────────────────────────────────────────
    ("at", "liên hệ @peter_shop nhé", "liên hệ a còng peter_shop nhé"),
    ("at", "@", "a còng"),
    # An email is protected wholesale, so its @ must NOT be spoken.
    ("untouched", "email abc@gmail.com nhé", "email abc@gmail.com nhé"),
    # %20 inside a URL is percent-encoding, not a percentage.
    ("untouched", "xem tại https://a.vn/x?q=1%20b", "xem tại https://a.vn/x?q=1%20b"),
]


@pytest.mark.parametrize(
    ("case", "text", "expected"),
    CASES,
    ids=[f"{case}-{i}" for i, (case, _, _) in enumerate(CASES)],
)
def test_normalize(case, text, expected):
    assert N(text) == expected


def test_every_supported_form_is_covered():
    """A form with no case is a form nobody notices breaking."""
    covered = {c for c, _, _ in CASES}
    assert covered >= {
        "number", "date", "time", "version", "fraction", "abbrev", "untouched",
        "percent", "at",
    }, f"missing coverage for {covered}"


# ── regressions ───────────────────────────────────────────────────────────────
# Each case below was a real bug found by scoring the normalizer's output
# against ZeroBench-TTS's curated readings. They are grouped here so a future
# scanner edit that reintroduces one fails loudly.

@pytest.mark.parametrize("text,expected", [
    # A sentence-final period made the trailing guard `(?![\d/.\-])` reject the
    # date, so it fell through to the bare-number branch and kept its slashes:
    # "ngày ba mươi mốt/mười hai/hai nghìn...". The guard must exclude a LONGER
    # number, not any dot.
    ("Hạn cuối là ngày 31/12/2025.",
     "Hạn cuối là ngày ba mươi mốt tháng mười hai năm hai nghìn không trăm hai mươi lăm."),
    ("ngày 15/8.", "ngày mười lăm tháng tám."),
    ("Ngày 2/9/1945, Chủ tịch",
     "Ngày hai tháng chín năm một nghìn chín trăm bốn mươi lăm, Chủ tịch"),
])
def test_date_survives_sentence_final_period(text, expected):
    assert N(text) == expected


@pytest.mark.parametrize("text,expected", [
    # Alphanumeric identifiers are spaced out, not read as quantities:
    # "AB-1234" was becoming "AB-một nghìn hai trăm ba mươi tư". Letters stay
    # capitals — the model spells them; a Vietnamese letter-name table would be
    # a second thing to get wrong.
    ("Mã đơn hàng là AB-1234.", "Mã đơn hàng là A B 1 2 3 4."),
    # ... and the letter half must not go through the abbreviation table, which
    # turned the flight code VN-215 into "Việt Nam-hai trăm mười lăm".
    ("Chuyến bay VN-215 khởi hành.", "Chuyến bay V N 2 1 5 khởi hành."),
])
def test_alphanumeric_codes_are_spelled_not_counted(text, expected):
    assert N(text) == expected


def test_degree_sign_is_spoken():
    # The bare-number branch claimed the digits and stranded "°C".
    assert N("Hôm nay 38°C.") == "Hôm nay ba mươi tám độ xê."


def test_roman_numeral_after_ordinal_cue():
    # "quý III" is a quarter, not the acronym III.
    assert N("GDP quý III tăng.") == "GDP quý ba tăng."
    # Without a cue it stays an acronym.
    assert "ba" not in N("Nhóm III họp.").split()[1]


def test_acronym_pair_over_slash_is_left_verbatim():
    # Each side used to expand separately ("đô la mỹ/việt nam đồng"), inventing
    # a reading worse than the source. The model says "USD/VND" fine as written.
    assert N("Tỷ giá USD/VND cao.") == "Tỷ giá USD/VND cao."
    # A one-letter side counts too, or "KM" expands to "khuyến mại".
    assert N("Tốc độ 60 KM/H.") == "Tốc độ sáu mươi KM/H."
    # A lone acronym still expands — only the paired form is held back.
    assert N("Giá 100 USD.") == "Giá một trăm đô la mỹ."


def test_prefix_abbreviation_keeps_its_dot_out_of_the_sentence():
    assert N("Hà Nội, TP. HCM và Đà Nẵng.") == "Hà Nội, thành phố Hồ Chí Minh và Đà Nẵng."
    # ... but a genuine sentence break must survive.
    assert N("Anh ấy làm ở FPT. Sau đó nghỉ.") == "Anh ấy làm ở FPT. Sau đó nghỉ."


def test_coordinated_day_shares_the_month():
    assert N("hai ngày 26 và 27/6") == "hai ngày hai mươi sáu và hai mươi bảy tháng sáu"
    # "và" is only a date cue near a real one — this stays arithmetic.
    assert N("Kết quả là 3 và 4/5 của tổng.") == "Kết quả là ba và 4/5 của tổng."


def test_parenthesized_acronym_after_its_own_expansion_is_spelled():
    assert N("Tổ chức Thương mại Thế giới (WTO) dự báo.") == (
        "Tổ chức Thương mại Thế giới (W T O) dự báo.")
