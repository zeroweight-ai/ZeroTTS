// ZeroTTS on ggml — a quantized CPU/WASM runtime for the two hot graphs.
//
// Why this exists: at batch 1 every matmul in this model is a mat-VEC, so the
// generation loop is bound by how many bytes of weights it reads per frame,
// not by flops. The fp32 ONNX path reads ~1.9 GB/frame, which needs ~24 GB/s
// sustained to hit the model's 12.5 frames/s realtime line — comfortable on
// Apple silicon, not on a typical laptop. Holding the weights quantized cuts
// that by 4x (q8_0) or ~7x (q4_0). See cpp/README.md.
//
// The call sequence mirrors docs/RUNTIME.md's ONNX contract 1:1, so the
// browser synthesizer can swap backends without restructuring its loop:
//
//     zerotts_begin(ctx, text_ids, n_text, voice_emb);   // once per segment
//     for (t = 0; ; t++) {
//         zerotts_frame(ctx, forbid_eoa, &sp, u_ctrl, u_audio, codes, &eoa);
//         ... keep/stop ...
//         zerotts_advance(ctx, codes, t);                // next global hidden
//     }
//
// `zerotts_frame` samples; `zerotts_advance` steps the global transformer.
// The random draws are inputs, exactly as in the ONNX graph, so a run is
// reproducible from a seed and can be compared bit-for-bit against the
// PyTorch and onnxruntime paths (see cpp/test/).

#ifndef ZEROTTS_H
#define ZEROTTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct zerotts_context;

struct zerotts_hparams {
    int32_t d_model;
    int32_t n_heads;
    int32_t n_layers;         // text encoder AND global decoder (same count)
    int32_t d_ff;
    int32_t local_n_layers;
    int32_t local_n_heads;
    int32_t local_d_ff;
    int32_t n_depth;          // num_codebooks + 1
    int32_t num_codebooks;
    int32_t codebook_size;
    int32_t vocab_size;
    int32_t n_voice_queries;
    int32_t cross_qk_norm;
    float   rope_theta;
    float   layer_norm_eps;
    float   qk_norm_eps;
};

// Per-frame sampling controls. Mirrors local_frame_decode.onnx's scalar inputs;
// the defaults are model/zero_tts.py's inference defaults.
struct zerotts_sampling {
    float   text_temperature;           // 1.0
    int32_t text_topk;                  // 50 (clamped to 2 — the control
                                        //     channel is a {slot, eoa} choice)
    float   audio_temperature;          // 0.8
    int32_t audio_topk;                 // 25
    float   audio_topp;                 // 0.95
    float   audio_repetition_penalty;   // 1.2
};

void zerotts_sampling_defaults(struct zerotts_sampling * sp);

// `n_threads <= 0` picks a default from the hardware concurrency.
struct zerotts_context * zerotts_init_from_file(const char * gguf_path, int n_threads);
// Takes ownership of neither pointer; the buffer must outlive the context.
struct zerotts_context * zerotts_init_from_buffer(const void * data, size_t size, int n_threads);
void zerotts_free(struct zerotts_context * ctx);

const struct zerotts_hparams * zerotts_hparams_of(const struct zerotts_context * ctx);

// Longest utterance the KV cache is sized for, in frames.
int zerotts_max_frames(const struct zerotts_context * ctx);

// Start a segment: encode the text, cache the per-layer cross-attention K/V,
// run the [voice | soa] prefix, and leave the global hidden state ready for
// frame 0. `voice_emb` is n_voice_queries * d_model floats, row-major.
// Resets the repetition-penalty history. Returns 0 on success.
int zerotts_begin(struct zerotts_context * ctx,
                  const int32_t * text_ids, int n_text,
                  const float * voice_emb);

// Decode one frame from the current global hidden: samples the control channel
// and all K codebooks. `audio_u` is num_codebooks uniforms in [0,1).
// Writes num_codebooks codes and sets *out_is_eoa. Returns 0 on success.
int zerotts_frame(struct zerotts_context * ctx,
                  bool forbid_eoa,
                  const struct zerotts_sampling * sp,
                  float ctrl_u, const float * audio_u,
                  int32_t * out_codes, int * out_is_eoa);

// Advance the global transformer by one frame, making the next hidden state
// current. `frame_index` is t (0-based); the token's RoPE position is
// n_voice_queries + 1 + t. Returns 0 on success.
int zerotts_advance(struct zerotts_context * ctx, const int32_t * codes, int frame_index);

// Cumulative wall time in ms spent inside each stage since the last reset —
// the same split the ONNX benchmark reports, so the two are comparable.
void zerotts_timings(const struct zerotts_context * ctx,
                     double * text_ms, double * frame_ms, double * advance_ms,
                     int * n_frames);
void zerotts_reset_timings(struct zerotts_context * ctx);

// Copies the current global hidden state (d_model floats) out. Test hook: it is
// the one value the ONNX path also exposes directly, so it isolates "is the
// backbone right" from "is the depth decoder right".
void zerotts_debug_hidden(const struct zerotts_context * ctx, float * out);

// Like zerotts_frame, but the depth decoder is fed `forced` instead of what it
// sampled, while `out_codes` still reports what it *would* have drawn.
//
// This is how quantization damage gets measured. Two models sampling
// independently diverge after the first differing token and every later frame is
// then a different utterance, so comparing their outputs says nothing. Forcing
// both along the same code sequence keeps the inputs identical at every depth of
// every frame, and the disagreement rate in `out_codes` is then attributable to
// the weights alone.
int zerotts_frame_forced(struct zerotts_context * ctx,
                         bool forbid_eoa,
                         const struct zerotts_sampling * sp,
                         float ctrl_u, const float * audio_u,
                         const int32_t * forced,
                         int32_t * out_codes, int * out_is_eoa);

#ifdef __cplusplus
}
#endif

#endif // ZEROTTS_H
