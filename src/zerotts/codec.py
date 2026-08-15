"""onnxruntime-only MOSS-Audio-Tokenizer-Nano **decoder** — codes -> waveform.

Vendored, not downloaded. The decoder graphs ship inside the ZeroTTS weights
repo (``onnx/codec/``) so a ZeroTTS install has no runtime dependency on any
third-party model repo staying up or unchanged.

Decoder only. The upstream export also has an encoder graph (waveform -> codes),
used solely to turn reference audio into voice latents. That is voice cloning,
which this release does not do (see zerotts.voices), so the encoder is neither
shipped nor wrapped — it would be ~45MB of weights nothing here can call.

Conventions worth knowing before touching this:
  * native sample rate 48 kHz; the codec is stereo internally, the public
    interface is mono (decode averages the two channels).
  * this export uses (batch, T, K) — TIME-major, codebook-LAST — for
    ``audio_codes``. ZeroTTS's AR loop produces (B, K, T), so decode transposes.
  * every integer tensor here is **int32**, not int64. This is verified against
    the graphs; do not "fix" it to int64 out of PyTorch habit.

Credit: MOSS-Audio-Tokenizer-Nano by the OpenMOSS team, Apache-2.0. See
https://github.com/OpenMOSS/MOSS-Audio-Tokenizer and the NOTICE file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MossCodecDecoder:
    """Args:
        codec_dir: directory holding the decoder graphs and
            ``codec_browser_onnx_meta.json``.
        providers: onnxruntime execution providers.
        intra_op_num_threads: per-session thread count.
    """

    def __init__(
        self,
        codec_dir: str | Path,
        providers: list[str] | None = None,
        intra_op_num_threads: int = 4,
    ):
        import onnxruntime as ort

        codec_dir = Path(codec_dir)
        meta_path = codec_dir / "codec_browser_onnx_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{codec_dir} has no codec_browser_onnx_meta.json — the vendored "
                f"codec is missing from this model directory.")
        meta = json.loads(meta_path.read_text())
        self._meta = meta

        cfg = meta["codec_config"]
        self.sample_rate = int(cfg["sample_rate"])
        self.num_channels = int(cfg["channels"])
        self.frame_size = int(cfg["downsample_rate"])
        self.frame_rate = self.sample_rate / self.frame_size
        self.num_codebooks = int(cfg["num_quantizers"])

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = 1
        resolved = providers or ["CPUExecutionProvider"]

        def _session(key: str):
            return ort.InferenceSession(
                str(codec_dir / meta["files"][key]),
                sess_options=sess_options,
                providers=resolved,
            )

        self._decode_full_sess = _session("decode_full")
        self._decode_step_sess = _session("decode_step")

    def decode(self, codes_bkt: np.ndarray) -> np.ndarray:
        """(B, K, T) int codes -> (B, T_audio) float32 mono at sample_rate."""
        codes = np.asarray(codes_bkt)
        if codes.ndim == 2:  # (K, T) -> (1, K, T)
            codes = codes[None, :, :]
        codes_btk = codes.transpose(0, 2, 1).astype(np.int32)  # -> (B, T, K)
        lengths = np.array([codes_btk.shape[1]], dtype=np.int32)
        audio, audio_lengths = self._decode_full_sess.run(
            None, {"audio_codes": codes_btk, "audio_code_lengths": lengths}
        )
        n = int(audio_lengths.reshape(-1)[0])
        return audio[:, :, :n].mean(axis=1).astype(np.float32)

    def streaming_decoder(self) -> MossStreamingDecoder:
        """Open a stateful streaming decoder (keeps the causal decoder KV cache
        across chunks). Call decode_chunk per chunk, then close."""
        return MossStreamingDecoder(self)


class MossStreamingDecoder:
    """KV-cached streaming decode over decode_step.onnx, driven by the state
    layout ``codec_browser_onnx_meta.json``'s "streaming_decode" section
    describes: per-decoder transformer offsets plus per-layer attention caches
    (key/value/position ring buffers). Use via
    ``MossCodecDecoder.streaming_decoder()``."""

    def __init__(self, codec: MossCodecDecoder):
        self._codec = codec
        self._session = codec._decode_step_sess
        streaming = codec._meta.get("streaming_decode", {})
        self._transformer_specs = list(streaming.get("transformer_offsets", []))
        self._attention_specs = list(streaming.get("attention_caches", []))
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._state: dict = {}
        self._reset_state()

    def _reset_state(self) -> None:
        self._state = {}
        for spec in self._transformer_specs:
            self._state[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self._attention_specs:
            self._state[str(spec["offset_input_name"])] = np.zeros(
                tuple(spec["offset_shape"]), dtype=np.int32)
            self._state[str(spec["cached_keys_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32)
            self._state[str(spec["cached_values_input_name"])] = np.zeros(
                tuple(spec["cache_shape"]), dtype=np.float32)
            # -1, not 0: position 0 is a real position, so a zero-filled ring
            # buffer would read as "every slot holds frame 0".
            self._state[str(spec["cached_positions_input_name"])] = np.full(
                tuple(spec["positions_shape"]), -1, dtype=np.int32)

    def decode_chunk(self, codes_bkt: np.ndarray) -> np.ndarray:
        """(1, K, n) int codes -> (1, chunk_samples) float32 mono."""
        codes = np.asarray(codes_bkt)
        if codes.ndim == 2:
            codes = codes[None, :, :]
        codes_btk = codes.transpose(0, 2, 1).astype(np.int32)  # -> (1, n, K)
        feeds = {
            "audio_codes": codes_btk,
            "audio_code_lengths": np.array([codes_btk.shape[1]], dtype=np.int32),
            **self._state,
        }
        outputs = self._session.run(None, feeds)
        named = dict(zip(self._output_names, outputs))
        for spec in self._transformer_specs:
            self._state[str(spec["input_name"])] = named[str(spec["output_name"])]
        for spec in self._attention_specs:
            for key in ("offset", "cached_keys", "cached_values", "cached_positions"):
                self._state[str(spec[f"{key}_input_name"])] = named[str(spec[f"{key}_output_name"])]
        n = int(named["audio_lengths"].reshape(-1)[0])
        return named["audio"][:, :, :n].mean(axis=1).astype(np.float32)

    def close(self) -> None:
        self._reset_state()
