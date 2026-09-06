"""Convert a ZeroTTS training checkpoint into a GGUF file for the ggml runtime.

The ggml port (cpp/) reimplements the three hot graphs — text encoder, global
decoder step, local frame decode — directly on ggml so the weights can live in
memory *quantized*. That is the whole point: at batch 1 every matmul here is a
matvec, so the runtime is bound by how many bytes of weights it reads per
frame, not by flops. See cpp/README.md for the measurement.

Tensor naming is this port's own (the torch module paths are not stable across
refactors and are longer than GGUF's 64-char limit in places). The C++ side
looks tensors up by these exact names; keep the two in step.

    conda run -n tts python cpp/convert/convert_zerotts_gguf.py \
        /path/to/checkpoint.pt -o zerotts-q8_0.gguf --quant q8_0

**Raw weights, not the EMA shadow, by default.** export_compact.py in the
research repo defaults the other way, but the ONNX graphs this port has to agree
with were exported from the raw weights — checked by matching their LayerNorm
initializers against both candidates, where raw is an exact match and EMA is off
by ~1e-4. Converting with --ema instead produces a model that is bit-exact
against nothing and drifts ~5% in the global hidden state, which shows up as a
demo whose two backends quietly say different things.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import gguf

# ── quantization policy ─────────────────────────────────────────────────────
# ggml block-quantized types need the *contiguous* dimension (ne0 = a matmul's
# in-features, or an embedding row's width) to be a multiple of the block size.
# Every such dim here is 768, 2048 or 3072, so all of them qualify.
QUANT = {
    "f32": gguf.GGMLQuantizationType.F32,
    "f16": gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
}


def merge_ema(ckpt: dict) -> dict:
    """Fold the EMA shadow into the raw weights, as export_compact.py does."""
    sd = dict(ckpt["model"])
    ema = ckpt.get("ema")
    if not ema:
        print("  no 'ema' key — using raw weights")
        return sd
    shadow = ema.get("shadow", ema)
    n = 0
    for k, v in shadow.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
            n += 1
    print(f"  merged {n} EMA-shadow tensors")
    return sd


def build_tensors(sd: dict, hp: dict) -> dict[str, tuple[torch.Tensor, bool]]:
    """torch state dict -> {gguf name: (tensor, is_matmul_weight)}.

    ``is_matmul_weight`` drives quantization: True for the 2-D weights that are
    read on the hot path, False for norms, biases and small parameters, which
    stay f32 (they cost nothing and quantizing them is where accuracy goes).

    **Layout.** ggml's ``ggml_mul_mat(W, x)`` wants W with ne0 = in-features,
    which for a row-major buffer means shape ``(out, in)`` — exactly how
    ``nn.Linear.weight`` is already stored, so those pass through untouched.
    ``audio_heads`` is the exception: it is applied as ``h @ audio_heads[c]``
    with ``audio_heads[c]`` shaped ``(in, out)``, so each slice is transposed
    here to ``(out, in)`` and the C++ side treats it like any other linear.
    """
    t: dict[str, tuple[torch.Tensor, bool]] = {}
    B = "backbone."

    def put(name, tensor, mm=False):
        t[name] = (tensor.detach().to(torch.float32).contiguous(), mm)

    # ── shared embedding / control head (tied in the model: lm_head.weight IS
    #    text_embed.weight, so it is stored once and read twice) ──
    put("text_embed.weight", sd[B + "text_embed.weight"], mm=True)
    put("text_mask_embed", sd[B + "text_mask_embed"])
    put("text_ln_f.weight", sd[B + "text_ln_f.weight"])
    put("text_ln_f.bias", sd[B + "text_ln_f.bias"])
    put("ln_f.weight", sd[B + "ln_f.weight"])
    put("ln_f.bias", sd[B + "ln_f.bias"])

    def norm(dst, src):
        put(f"{dst}.weight", sd[f"{src}.weight"])
        put(f"{dst}.bias", sd[f"{src}.bias"])

    def attn(dst, src, prefix="attn"):
        for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")):
            put(f"{dst}.{prefix}_{a}.weight", sd[f"{src}.{b}.weight"], mm=True)

    def ffn(dst, src):
        put(f"{dst}.ffn_up.weight", sd[f"{src}.net.0.weight"], mm=True)
        put(f"{dst}.ffn_up.bias", sd[f"{src}.net.0.bias"])
        put(f"{dst}.ffn_down.weight", sd[f"{src}.net.3.weight"], mm=True)
        put(f"{dst}.ffn_down.bias", sd[f"{src}.net.3.bias"])

    # ── bidirectional text encoder ──
    for i in range(hp["n_layers"]):
        s = f"{B}text_blocks.{i}"
        norm(f"txt.{i}.norm_attn", f"{s}.norm_attn")
        attn(f"txt.{i}", f"{s}.attn")
        norm(f"txt.{i}.norm_ffn", f"{s}.norm_ffn")
        ffn(f"txt.{i}", f"{s}.ffn")

    # ── global causal decoder (self-attn + cross-attn to text + ffn) ──
    for i in range(hp["n_layers"]):
        s = f"{B}blocks.{i}"
        norm(f"dec.{i}.norm_self", f"{s}.norm_self")
        attn(f"dec.{i}", f"{s}.self_attn", prefix="self")
        norm(f"dec.{i}.norm_cross", f"{s}.norm_cross")
        norm(f"dec.{i}.norm_context", f"{s}.norm_context")
        attn(f"dec.{i}", f"{s}.cross_attn", prefix="cross")
        norm(f"dec.{i}.norm_ffn", f"{s}.norm_ffn")
        ffn(f"dec.{i}", f"{s}.ffn")

    # ── local (depth) transformer over a frame's 17 channel positions ──
    put("local.depth_embed", sd[B + "local_transformer.depth_embed"])
    norm("local.ln_f", B + "local_transformer.ln_f")
    for i in range(hp["local_n_layers"]):
        s = f"{B}local_transformer.blocks.{i}"
        norm(f"loc.{i}.norm_attn", f"{s}.norm_attn")
        attn(f"loc.{i}", f"{s}.attn")
        norm(f"loc.{i}.norm_ffn", f"{s}.norm_ffn")
        ffn(f"loc.{i}", f"{s}.ffn")

    # ── per-codebook audio embeddings and heads ──
    heads = sd[B + "audio_heads"]                       # (K, d_model, codebook)
    for c in range(hp["num_codebooks"]):
        put(f"audio_embed.{c}.weight", sd[f"{B}audio_embeddings.{c}.weight"], mm=True)
        # (d_model, codebook) -> (codebook, d_model): out-major, like nn.Linear.
        put(f"audio_head.{c}.weight", heads[c].t(), mm=True)

    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--quant", default="q8_0", choices=sorted(QUANT),
                    help="type for the hot 2-D weights (default: q8_0)")
    ap.add_argument("--embed-quant", default=None, choices=sorted(QUANT),
                    help="type for embedding tables and output heads; defaults "
                         "to --quant, but q8_0 is a good floor here — these are "
                         "read once per depth step, so making them cheaper buys "
                         "little bandwidth and costs accuracy directly")
    ap.add_argument("--ema", dest="ema", action="store_true", default=False,
                    help="merge the EMA shadow. Off by default: the shipped ONNX "
                         "export used the raw weights, and this port is validated "
                         "bit-for-bit against it")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = merge_ema(ckpt) if args.ema else dict(ckpt["model"])
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.blocks."))
    local_n_layers = 1 + max(int(k.split(".")[3]) for k in sd
                             if k.startswith("backbone.local_transformer.blocks."))
    hp = {
        "d_model": sd["backbone.ln_f.weight"].shape[0],
        "n_heads": 12,
        "n_layers": n_layers,
        "d_ff": sd["backbone.blocks.0.ffn.net.0.weight"].shape[0],
        "local_n_layers": local_n_layers,
        "local_n_heads": 8,
        "local_d_ff": sd["backbone.local_transformer.blocks.0.ffn.net.0.weight"].shape[0],
        "n_depth": sd["backbone.local_transformer.depth_embed"].shape[1],
        "num_codebooks": int(ckpt["num_codebooks"]),
        "codebook_size": int(ckpt["codebook_size"]),
        "vocab_size": int(ckpt["vocab_size"]),
        "n_voice_queries": int(ckpt["n_voice_queries"]),
        "cross_qk_norm": bool(ckpt.get("cross_qk_norm", False)),
    }
    # n_heads / local_n_heads are not recorded in the checkpoint (they do not
    # change any tensor shape), so they come from training/config.py's defaults
    # above. Assert what they do constrain, so a differently-configured
    # checkpoint fails here rather than producing silent garbage.
    assert hp["d_model"] % hp["n_heads"] == 0
    assert hp["d_model"] % hp["local_n_heads"] == 0
    assert hp["n_depth"] == hp["num_codebooks"] + 1

    print("  hparams: " + ", ".join(f"{k}={v}" for k, v in hp.items()))

    tensors = build_tensors(sd, hp)

    hot = QUANT[args.quant]
    embed = QUANT[args.embed_quant or args.quant]

    w = gguf.GGUFWriter(str(args.out), "zerotts")
    w.add_description("ZeroTTS — MOSS-TTS-Nano-style two-transformer AR codec TTS")
    for k, v in hp.items():
        if isinstance(v, bool):
            w.add_bool(f"zerotts.{k}", v)
        else:
            w.add_uint32(f"zerotts.{k}", int(v))
    w.add_float32("zerotts.rope_theta", 10000.0)       # transformer_block._apply_rope
    w.add_float32("zerotts.layer_norm_eps", 1e-5)      # nn.LayerNorm default
    w.add_float32("zerotts.qk_norm_eps", 1e-6)         # CrossAttention.qk_norm
    w.add_file_type(hot)

    total_raw = total_out = 0
    for name, (t, is_mm) in tensors.items():
        a = t.numpy()
        # Only 2-D weights get quantized, and only when the row length is a
        # whole number of blocks — everything here is, but a re-shaped model
        # should fall back rather than corrupt the file.
        qtype = gguf.GGMLQuantizationType.F32
        if is_mm and a.ndim == 2:
            want = embed if (".weight" in name and
                             ("embed" in name or "head" in name)) else hot
            blk = gguf.GGML_QUANT_SIZES[want][0]
            if a.shape[-1] % blk == 0:
                qtype = want
        data = a if qtype == gguf.GGMLQuantizationType.F32 else gguf.quants.quantize(a, qtype)
        # gguf derives the logical shape from the quantized byte array, so the
        # array passed here is the packed one and no raw_shape is given.
        w.add_tensor(name, data, raw_dtype=qtype)
        total_raw += a.nbytes
        total_out += data.nbytes
        print(f"  {name:38s} {str(tuple(a.shape)):16s} -> {qtype.name}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()

    print(f"\nwrote {args.out}  ({total_out/1e6:.1f} MB, from {total_raw/1e6:.1f} MB f32 "
          f"= {total_raw/total_out:.2f}x smaller)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
