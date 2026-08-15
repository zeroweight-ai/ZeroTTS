"""Emit the JS parity corpus from the Python normalizer (curated + fuzz).

    python js/test/gen_cases.py > js/test/cases.json

Regenerate whenever the Python rules change; the JS port is then checked
against the new expectations by js/test/parity.mjs.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from zerotts.text_norm import normalize_vi_text  # noqa: E402

FRAGMENTS = [
    "{n}", "{n}%", "{n},{d}%", "ngày {dd}/{mm}", "{dd}/{mm}/{yyyy}", "{mm}/{yyyy}",
    "{hh}h{mi}", "{hh}:{mi}", "{hh}h", "{hh}:{mi}:{ss}", "v{n}.{d}", "{n}.{d}.{d}",
    "{n}/{n2}", "@", "@shop", "ATM", "UBND", "TP.HCM", "{n}.{ddd}.{ddd}",
    "{n} + {n2}", "{n}^{d}", "-{n}", "+{n}", "{n},{d}", "giá {n}đ", "{n} km/h",
    "abc@gmail.com", "https://a.vn/x", "www.b.com", "{n}m{d}",
]
WORDS = ["giá", "lúc", "còn", "tăng", "giảm", "hôm nay", "nhé", "và", "của", "khoảng"]


def main() -> None:
    cases = []
    # 1. the curated cases, so the corpus covers every documented rule
    try:
        from test_vi_normalizer import CASES
        cases += [{"in": text, "expected": expected} for _c, text, expected in CASES]
    except ImportError:
        pass

    # 2. fuzz, seeded so the corpus is reproducible
    rng = random.Random(1234)
    for _ in range(1200):
        parts = []
        for _ in range(rng.randint(1, 5)):
            if rng.random() < 0.45:
                parts.append(rng.choice(WORDS))
            else:
                parts.append(rng.choice(FRAGMENTS).format(
                    n=rng.randint(0, 99999), n2=rng.randint(1, 99), d=rng.randint(0, 9),
                    dd=rng.randint(1, 31), mm=rng.randint(1, 12),
                    yyyy=rng.randint(1900, 2100), hh=rng.randint(0, 23),
                    mi=rng.randint(0, 59), ss=rng.randint(0, 59),
                    ddd=f"{rng.randint(0, 999):03d}"))
        text = " ".join(parts)
        cases.append({"in": text, "expected": normalize_vi_text(text)})

    json.dump(cases, sys.stdout, ensure_ascii=False, separators=(',', ':'))


if __name__ == "__main__":
    main()
