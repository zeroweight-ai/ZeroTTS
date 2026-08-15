"""BPE text encoding for ZeroTTS — numpy only, no torch.

Port of the training-side tokenizer with the training-only parts removed: BPE
dropout is gone (inference is always dropout=0.0) and ids come back as
``np.ndarray`` instead of ``torch.Tensor``. Everything that affects *which ids a
string produces* is byte-identical to the training tokenizer; that is the whole
contract of this file.

``normalize_text`` in particular must not drift. It is NFC + whitespace collapse:

  * NFC matters most for Vietnamese — decomposed input ('e' + combining marks)
    tokenizes into different pieces than the precomposed 'ệ' the vocab was
    trained on, scattering the tone across tokens.
  * Whitespace runs collapse to a single space because whitespace is carried as
    its OWN token: a tab or newline has no token of its own, so it would encode
    as <unk> and silently delete the word boundary it stands for.

A divergence here does not raise — it just makes the model sound worse on input
that looks fine, which is why it gets a docstring instead of a comment.
"""

from __future__ import annotations

import re
from pathlib import Path
from unicodedata import normalize

import numpy as np

# Pinned model-wide id convention. The exported graphs hardcode <soa>/<slot>/<eoa>
# and the text embedding is tied to the LM head, so these ids are part of the
# weights, not a preference.
SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eot>": 2,
    "<soa>": 3,
    "<slot>": 4,
    "<eoa>": 5,
    "<en>": 6,
    "<vi>": 7,
}

UNK_TOKEN = "<unk>"
_WS_RUN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC + whitespace collapse. Case and punctuation preserved verbatim."""
    return _WS_RUN.sub(" ", normalize("NFC", text))


class BPEProcessor:
    """Args:
        tokenizer: a tokenizers JSON path, the JSON string itself, or an
            already-built ``tokenizers.Tokenizer``.
    """

    def __init__(self, tokenizer):
        from tokenizers import Tokenizer

        if tokenizer is None:
            raise ValueError("BPEProcessor requires a tokenizer (JSON path, JSON string, "
                             "or Tokenizer instance).")

        if isinstance(tokenizer, Tokenizer):
            self._tok = tokenizer
        else:
            text = str(tokenizer)
            if text.lstrip().startswith("{"):
                self._tok = Tokenizer.from_str(text)
            else:
                self._tok = Tokenizer.from_file(str(tokenizer))

        vocab = self._tok.get_vocab()
        self._vocab_size = len(vocab)
        self.unk_id = vocab.get(UNK_TOKEN)
        for name, want in SPECIAL_TOKENS.items():
            got = vocab.get(name)
            if got != want:
                raise ValueError(
                    f"tokenizer special id mismatch: {name} is {got}, must be {want}. "
                    f"The exported graphs hardcode these ids — this tokenizer does not "
                    f"belong to these weights.")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def to_str(self) -> str:
        return self._tok.to_str()

    def encode_body(self, text: str) -> list:
        """NFC + whitespace-collapse -> BPE ids, no BOS/EOT."""
        return self._tok.encode(normalize_text(text)).ids

    def decode(self, ids) -> str:
        """Ids back to text, skipping the reserved specials (0-7)."""
        keep = [int(i) for i in np.asarray(ids).reshape(-1) if int(i) >= len(SPECIAL_TOKENS)]
        return self._tok.decode(keep)

    def wrap_ids(self, body_ids, max_length: int = 512) -> np.ndarray:
        """[BOS | body | EOT] as int32, body truncated to ``max_length``."""
        body = [int(i) for i in body_ids][:max_length]
        ids = [SPECIAL_TOKENS["<bos>"]] + body + [SPECIAL_TOKENS["<eot>"]]
        return np.asarray(ids, dtype=np.int32)

    def __call__(self, text: str, max_length: int = 512) -> np.ndarray:
        return self.wrap_ids(self.encode_body(text), max_length)


def load_tokenizer(config: dict, model_dir: str | Path | None = None) -> BPEProcessor:
    """Build the tokenizer described by a model ``config.json``.

    Accepts either an inline ``bpe_tokenizer`` JSON string (how the private
    export embeds it) or a ``tokenizer.json`` sitting next to the config (how
    the published repo lays it out).

    Char-tokenizer exports are refused rather than silently mis-encoded: the
    published model is BPE and nothing here can encode for a char vocab.
    """
    fmt = str(config.get("text_format", "bpe"))
    if fmt != "bpe":
        raise ValueError(
            f"text_format={fmt!r} is not supported — this package ships BPE only. "
            f"A char-vocab export needs a different tokenizer than the one here.")

    inline = config.get("bpe_tokenizer")
    if inline:
        return BPEProcessor(str(inline))

    if model_dir is not None:
        path = Path(model_dir) / "tokenizer.json"
        if path.exists():
            return BPEProcessor(path)

    raise FileNotFoundError(
        "no tokenizer found: config.json has no inline 'bpe_tokenizer' and no "
        "tokenizer.json sits beside it.")
