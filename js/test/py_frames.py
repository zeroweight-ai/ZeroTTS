"""Replay the JS draw sequence through the Python package and compare frames.

    node --experimental-strip-types test/frames.mjs /path/to/model
    python test/py_frames.py /path/to/model

Exits non-zero if the two runtimes disagree on any frame.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from zerotts import ZeroTTS  # noqa: E402

model_dir = sys.argv[1] if len(sys.argv) > 1 else "zeroweight-ai/ZeroTTS"
text = sys.argv[2] if len(sys.argv) > 2 else "Xin chào các bạn."
voice = sys.argv[3] if len(sys.argv) > 3 else "arya"

draws = json.load(open("js_draws.json"))
cursor = {"i": 0}


def replay(size=None):
    """Stand in for np.random.random, handing back the JS draws in order."""
    shape = () if size is None else (size if isinstance(size, tuple) else (size,))
    n = int(np.prod(shape)) if shape else 1
    out = np.asarray(draws[cursor["i"]:cursor["i"] + n], dtype=np.float32)
    cursor["i"] += n
    return out.reshape(shape) if shape else out[0]


tts = ZeroTTS.from_pretrained(model_dir, intra_op_num_threads=8)
np.random.random = replay  # only after warmup, which consumes draws of its own

emb = np.fromfile(Path(model_dir) / "voices" / voice / "voice.bin",
                  dtype="<f4").reshape(1, tts.n_voice_queries, tts.d_model)
frames = [f[0].tolist() for f in tts._generate_frames(text, 4, 1500, voice_emb=emb)]
json.dump(frames, open("py_frames.json", "w"))

js = json.load(open("js_frames.json"))
if js == frames:
    print(f"{len(frames)} frames — BIT-IDENTICAL across Python and JS")
    sys.exit(0)

print(f"DIVERGE: js {len(js)} frames, python {len(frames)}")
for i, (a, b) in enumerate(zip(js, frames)):
    if a != b:
        print(f"  first difference at frame {i}\n    js: {a}\n    py: {b}")
        break
sys.exit(1)
