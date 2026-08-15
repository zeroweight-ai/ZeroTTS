"""Tokenizer tests.

These guard the one thing that fails silently: ``normalize_text``. A drift there
does not raise — it changes which BPE pieces a string produces, so the model gets
a segmentation it never saw in training and simply sounds worse.
"""

import numpy as np
import pytest

from zerotts.tokenizer import SPECIAL_TOKENS, normalize_text


def test_special_token_ids_are_pinned():
    # These ids are baked into the exported graphs and the tied embedding table.
    assert SPECIAL_TOKENS == {
        "<pad>": 0, "<bos>": 1, "<eot>": 2, "<soa>": 3,
        "<slot>": 4, "<eoa>": 5, "<en>": 6, "<vi>": 7,
    }
    assert len(SPECIAL_TOKENS) == 8


def test_nfc_composes_vietnamese():
    decomposed = "ệ"          # e + dot below + circumflex
    assert normalize_text(decomposed) == "ệ"  # ệ
    assert len(normalize_text(decomposed)) == 1


def test_whitespace_runs_collapse():
    # Tabs/newlines have no token of their own; left alone they would encode as
    # <unk> and delete the word boundary they stand for.
    assert normalize_text("a\t\tb") == "a b"
    assert normalize_text("a\n\nb") == "a b"
    assert normalize_text("a   b") == "a b"
    assert normalize_text("a \r\n b") == "a b"


def test_case_and_punctuation_survive():
    text = "Xin chào, ZeroTTS 3,2% — OK?"
    assert normalize_text(text) == text


@pytest.mark.model
def test_roundtrip_and_wrapping(tts):
    ids = tts.tokenizer("Xin chào các bạn.")
    assert ids.dtype == np.int32
    assert ids[0] == SPECIAL_TOKENS["<bos>"]
    assert ids[-1] == SPECIAL_TOKENS["<eot>"]
    assert tts.tokenizer.decode(ids).strip() == "Xin chào các bạn."


@pytest.mark.model
def test_max_length_truncates_body_not_wrapper(tts):
    ids = tts.tokenizer("một hai ba bốn năm sáu bảy tám chín mười", max_length=3)
    assert len(ids) == 5  # bos + 3 + eot
    assert ids[0] == SPECIAL_TOKENS["<bos>"]
    assert ids[-1] == SPECIAL_TOKENS["<eot>"]
