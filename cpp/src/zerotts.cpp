// ZeroTTS on ggml. See ../include/zerotts.h for the contract and why this
// exists; docs/RUNTIME.md in the repo root for the reference semantics this
// reproduces, and model/backbone.py / inference/export_onnx.py in the research
// repo for the originals.
//
// Layout conventions, once, because everything below assumes them:
//
//   * A weight is stored the way nn.Linear stores it, (out, in), which in ggml
//     is ne0=in, ne1=out — exactly what ggml_mul_mat wants on its left. No
//     transposes at run time; the converter did the one that was needed.
//   * Activations are (d_model, n_tokens). Heads split as (d_head, n_head,
//     n_tokens) and permute to (d_head, n_tokens, n_head) for the score matmul.
//   * The K cache is position-major — element (dh, h, p) at p*d_model +
//     h*d_head + dh. The V cache is stored transposed, (n_ctx, d_model), so the
//     attention-weight matmul reads it without a per-frame copy. This is
//     llama.cpp's layout and the views below mirror it.

#include "zerotts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// From config.json's special_tokens. The control channel only ever emits
// <slot> or <eoa>; <soa> opens the audio stream.
static const int32_t SOA_ID  = 3;
static const int32_t SLOT_ID = 4;
static const int32_t EOA_ID  = 5;

// KV cache depth. The Python runtime's max_frames ceiling is 1500 frames
// (120 s at 12.5 Hz); the cache holds the voice block and <soa> on top.
static const int ZT_MAX_FRAMES = 1500;
static const int ZT_MAX_TEXT   = 512;   // Backbone.max_text_len

struct zt_norm { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };
struct zt_attn { ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr; };
struct zt_ffn  { ggml_tensor * up_w = nullptr, * up_b = nullptr,
                             * down_w = nullptr, * down_b = nullptr; };

struct zt_txt_layer { zt_norm n_attn, n_ffn; zt_attn attn; zt_ffn ffn; };
struct zt_dec_layer { zt_norm n_self, n_cross, n_context, n_ffn;
                      zt_attn self_a, cross_a; zt_ffn ffn; };
struct zt_loc_layer { zt_norm n_attn, n_ffn; zt_attn attn; zt_ffn ffn; };

struct zerotts_context {
    zerotts_hparams hp{};

    ggml_backend_t         backend  = nullptr;
    ggml_context *         ctx_w    = nullptr;
    ggml_backend_buffer_t  buf_w    = nullptr;
    ggml_context *         ctx_kv   = nullptr;
    ggml_backend_buffer_t  buf_kv   = nullptr;
    ggml_gallocr_t         alloc    = nullptr;
    int                    n_threads = 4;

    // weights
    ggml_tensor * text_embed = nullptr;      // also the control head (tied)
    ggml_tensor * text_ln_f_w = nullptr, * text_ln_f_b = nullptr;
    ggml_tensor * ln_f_w = nullptr, * ln_f_b = nullptr;
    ggml_tensor * depth_embed = nullptr;
    ggml_tensor * local_ln_f_w = nullptr, * local_ln_f_b = nullptr;
    std::vector<zt_txt_layer> txt;
    std::vector<zt_dec_layer> dec;
    std::vector<zt_loc_layer> loc;
    std::vector<ggml_tensor *> audio_embed, audio_head;

    // caches
    int n_ctx = 0;                            // n_voice_queries + 1 + max frames
    std::vector<ggml_tensor *> kv_k, kv_v;    // global decoder self-attention
    std::vector<ggml_tensor *> cx_k, cx_v;    // cross-attention to text (per segment)
    std::vector<ggml_tensor *> lkv_k, lkv_v;  // local depth transformer

    int n_kv   = 0;   // decoder positions written so far
    int n_text = 0;   // text tokens in the current segment

    std::vector<float>   h;      // current global hidden, d_model
    std::vector<uint8_t> seen;   // repetition history, num_codebooks * codebook_size

    double t_text = 0, t_frame = 0, t_advance = 0;   // ms
    int    n_frames = 0;
};

// ── small helpers ───────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static ggml_tensor * zt_layer_norm(ggml_context * c, ggml_tensor * x,
                                   const zt_norm & n, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, n.w);
    return ggml_add(c, x, n.b);
}

// nn.Sequential(Linear, SiLU, Dropout, Linear) — dropout is inference-inert.
static ggml_tensor * zt_feed_forward(ggml_context * c, ggml_tensor * x, const zt_ffn & f) {
    x = ggml_add(c, ggml_mul_mat(c, f.up_w, x), f.up_b);
    x = ggml_silu(c, x);
    return ggml_add(c, ggml_mul_mat(c, f.down_w, x), f.down_b);
}

static ggml_tensor * zt_rope(ggml_context * c, ggml_tensor * x, ggml_tensor * pos,
                             int d_head, float theta) {
    // GGML_ROPE_TYPE_NORMAL rotates adjacent pairs (c s c s …), which is what
    // transformer_block._apply_rope does with its x.view(..., half, 2) split.
    // NEOX's half-and-half ordering would be silently wrong here.
    return ggml_rope_ext(c, x, pos, nullptr, d_head, GGML_ROPE_TYPE_NORMAL,
                         0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

// Attention with no KV cache: the whole sequence attends to itself. Used by the
// bidirectional text encoder, where batch 1 means no padding and therefore no
// mask at all.
static ggml_tensor * zt_attn_full(ggml_context * c, const zt_attn & w, ggml_tensor * cur,
                                  ggml_tensor * pos, int n_tok, int n_head,
                                  int d_model, float theta) {
    const int d_head = d_model / n_head;

    ggml_tensor * Q = ggml_reshape_3d(c, ggml_mul_mat(c, w.q, cur), d_head, n_head, n_tok);
    ggml_tensor * K = ggml_reshape_3d(c, ggml_mul_mat(c, w.k, cur), d_head, n_head, n_tok);
    ggml_tensor * V = ggml_reshape_3d(c, ggml_mul_mat(c, w.v, cur), d_head, n_head, n_tok);

    Q = zt_rope(c, Q, pos, d_head, theta);
    K = zt_rope(c, K, pos, d_head, theta);

    ggml_tensor * Qp = ggml_permute(c, Q, 0, 2, 1, 3);                // (dh, T, H)
    ggml_tensor * Kp = ggml_permute(c, K, 0, 2, 1, 3);                // (dh, T, H)
    ggml_tensor * kq = ggml_mul_mat(c, Kp, Qp);                       // (T, T, H)
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) d_head), 0.0f);

    ggml_tensor * Vt  = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3)); // (T, dh, H)
    ggml_tensor * kqv = ggml_mul_mat(c, Vt, kq);                      // (dh, T, H)
    ggml_tensor * out = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), d_model, n_tok);
    return ggml_mul_mat(c, w.o, out);
}

// Self-attention against a persistent KV cache. Appends this call's tokens at
// `kv_head` and attends over the first `n_kv` positions. No mask is needed:
// every cached position is real, and causal order is guaranteed by the order of
// the calls — see zerotts_begin for why the voice block gets away with this too.
static ggml_tensor * zt_attn_cached(ggml_context * c, ggml_cgraph * gf,
                                    const zt_attn & w, ggml_tensor * cur,
                                    ggml_tensor * k_cache, ggml_tensor * v_cache,
                                    ggml_tensor * pos, int n_tok, int kv_head, int n_kv,
                                    int n_head, int d_model, int n_ctx, float theta) {
    const int    d_head = d_model / n_head;
    const size_t es     = sizeof(float);

    ggml_tensor * Q = ggml_reshape_3d(c, ggml_mul_mat(c, w.q, cur), d_head, n_head, n_tok);
    ggml_tensor * K = ggml_reshape_3d(c, ggml_mul_mat(c, w.k, cur), d_head, n_head, n_tok);
    ggml_tensor * V = ggml_mul_mat(c, w.v, cur);                      // (d_model, n_tok)

    if (pos) {
        Q = zt_rope(c, Q, pos, d_head, theta);
        K = zt_rope(c, K, pos, d_head, theta);
    }

    // Append. K position-major; V transposed, so the second matmul reads it
    // directly instead of copying the whole cache every frame.
    ggml_tensor * k_dst = ggml_view_1d(c, k_cache, n_tok * d_model, es * d_model * kv_head);
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, K, d_model, n_tok), k_dst));

    ggml_tensor * v_dst = ggml_view_2d(c, v_cache, n_tok, d_model, es * n_ctx, es * kv_head);
    ggml_build_forward_expand(gf, ggml_cpy(c, ggml_transpose(c, V), v_dst));

    ggml_tensor * Kall = ggml_view_3d(c, k_cache, d_head, n_kv, n_head,
                                      es * d_model, es * d_head, 0);
    ggml_tensor * Vall = ggml_view_3d(c, v_cache, n_kv, d_head, n_head,
                                      es * n_ctx, es * n_ctx * d_head, 0);

    ggml_tensor * Qp = ggml_permute(c, Q, 0, 2, 1, 3);                // (dh, n_tok, H)
    ggml_tensor * kq = ggml_mul_mat(c, Kall, Qp);                     // (n_kv, n_tok, H)
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) d_head), 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(c, Vall, kq);                    // (dh, n_tok, H)
    ggml_tensor * out = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), d_model, n_tok);
    return ggml_mul_mat(c, w.o, out);
}

// Cross-attention onto the text, whose K/V were projected once per segment (see
// build_text_graph). That precompute is not just tidiness: the ONNX graph takes
// `text_states` as an input on *every* frame and so re-runs both projections
// over the whole text each time — 9 layers x 2 x L x 768 x 768, which at a
// typical L is the same order of work as the rest of the decoder step.
static ggml_tensor * zt_cross_attn(ggml_context * c, const zt_attn & w, ggml_tensor * cur,
                                   ggml_tensor * k_cache, ggml_tensor * v_cache,
                                   ggml_tensor * pos, int n_tok, int n_text,
                                   int n_head, int d_model, int max_text,
                                   bool qk_norm, float qk_eps, float theta) {
    const int    d_head = d_model / n_head;
    const size_t es     = sizeof(float);

    ggml_tensor * Q = ggml_reshape_3d(c, ggml_mul_mat(c, w.q, cur), d_head, n_head, n_tok);
    if (qk_norm) {
        // RMS over d_head, per head. RoPE is a rotation and preserves the norm,
        // so normalizing before it matches CrossAttention.qk_norm's "before".
        Q = ggml_rms_norm(c, Q, qk_eps);
    }
    Q = zt_rope(c, Q, pos, d_head, theta);

    ggml_tensor * Kall = ggml_view_3d(c, k_cache, d_head, n_text, n_head,
                                      es * d_model, es * d_head, 0);
    ggml_tensor * Vall = ggml_view_3d(c, v_cache, n_text, d_head, n_head,
                                      es * max_text, es * max_text * d_head, 0);

    ggml_tensor * Qp = ggml_permute(c, Q, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(c, Kall, Qp);                     // (n_text, n_tok, H)
    kq = ggml_soft_max_ext(c, kq, nullptr, 1.0f / sqrtf((float) d_head), 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(c, Vall, kq);
    ggml_tensor * out = ggml_cont_2d(c, ggml_permute(c, kqv, 0, 2, 1, 3), d_model, n_tok);
    return ggml_mul_mat(c, w.o, out);
}

// ── loading ─────────────────────────────────────────────────────────────────

namespace {

struct loader {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    zerotts_context * m = nullptr;
    bool ok = true;

    uint32_t u32(const char * k, uint32_t fallback, bool required = true) {
        const int64_t id = gguf_find_key(gguf, k);
        if (id < 0) {
            if (required) { fprintf(stderr, "zerotts: missing key '%s'\n", k); ok = false; }
            return fallback;
        }
        return gguf_get_val_u32(gguf, id);
    }
    float f32(const char * k, float fallback) {
        const int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? fallback : gguf_get_val_f32(gguf, id);
    }
    bool boolean(const char * k, bool fallback) {
        const int64_t id = gguf_find_key(gguf, k);
        return id < 0 ? fallback : gguf_get_val_bool(gguf, id);
    }

    // Tensors are created in the model context as duplicates of the metadata
    // ones; data is filled in afterwards from the file/buffer.
    ggml_tensor * get(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        if (!t) {
            fprintf(stderr, "zerotts: missing tensor '%s'\n", name.c_str());
            ok = false;
            return nullptr;
        }
        ggml_tensor * d = ggml_dup_tensor(m->ctx_w, t);
        ggml_set_name(d, name.c_str());
        return d;
    }

    void norm(zt_norm & n, const std::string & p) {
        n.w = get(p + ".weight");
        n.b = get(p + ".bias");
    }
    void attn(zt_attn & a, const std::string & p) {
        a.q = get(p + "_q.weight");
        a.k = get(p + "_k.weight");
        a.v = get(p + "_v.weight");
        a.o = get(p + "_o.weight");
    }
    void ffn(zt_ffn & f, const std::string & p) {
        f.up_w   = get(p + ".ffn_up.weight");
        f.up_b   = get(p + ".ffn_up.bias");
        f.down_w = get(p + ".ffn_down.weight");
        f.down_b = get(p + ".ffn_down.bias");
    }
};

} // namespace

static bool zt_load_common(zerotts_context * m, loader & L) {
    zerotts_hparams & hp = m->hp;
    hp.d_model         = (int) L.u32("zerotts.d_model", 768);
    hp.n_heads         = (int) L.u32("zerotts.n_heads", 12);
    hp.n_layers        = (int) L.u32("zerotts.n_layers", 9);
    hp.d_ff            = (int) L.u32("zerotts.d_ff", 3072);
    hp.local_n_layers  = (int) L.u32("zerotts.local_n_layers", 4);
    hp.local_n_heads   = (int) L.u32("zerotts.local_n_heads", 8);
    hp.local_d_ff      = (int) L.u32("zerotts.local_d_ff", 2048);
    hp.n_depth         = (int) L.u32("zerotts.n_depth", 17);
    hp.num_codebooks   = (int) L.u32("zerotts.num_codebooks", 16);
    hp.codebook_size   = (int) L.u32("zerotts.codebook_size", 1024);
    hp.vocab_size      = (int) L.u32("zerotts.vocab_size", 8192);
    hp.n_voice_queries = (int) L.u32("zerotts.n_voice_queries", 10);
    hp.cross_qk_norm   = L.boolean("zerotts.cross_qk_norm", false) ? 1 : 0;
    hp.rope_theta      = L.f32("zerotts.rope_theta", 10000.0f);
    hp.layer_norm_eps  = L.f32("zerotts.layer_norm_eps", 1e-5f);
    hp.qk_norm_eps     = L.f32("zerotts.qk_norm_eps", 1e-6f);
    if (!L.ok) return false;

    // Enough headroom for every tensor's metadata; ggml only stores structs here.
    const size_t n_meta = 4096;
    ggml_init_params ip = { ggml_tensor_overhead() * n_meta, nullptr, /*no_alloc*/ true };
    m->ctx_w = ggml_init(ip);
    if (!m->ctx_w) return false;

    m->text_embed   = L.get("text_embed.weight");
    m->text_ln_f_w  = L.get("text_ln_f.weight");
    m->text_ln_f_b  = L.get("text_ln_f.bias");
    m->ln_f_w       = L.get("ln_f.weight");
    m->ln_f_b       = L.get("ln_f.bias");
    m->depth_embed  = L.get("local.depth_embed");
    m->local_ln_f_w = L.get("local.ln_f.weight");
    m->local_ln_f_b = L.get("local.ln_f.bias");

    m->txt.resize(hp.n_layers);
    m->dec.resize(hp.n_layers);
    for (int i = 0; i < hp.n_layers; i++) {
        const std::string t = "txt." + std::to_string(i);
        L.norm(m->txt[i].n_attn, t + ".norm_attn");
        L.attn(m->txt[i].attn,   t + ".attn");
        L.norm(m->txt[i].n_ffn,  t + ".norm_ffn");
        L.ffn (m->txt[i].ffn,    t);

        const std::string d = "dec." + std::to_string(i);
        L.norm(m->dec[i].n_self,    d + ".norm_self");
        L.attn(m->dec[i].self_a,    d + ".self");
        L.norm(m->dec[i].n_cross,   d + ".norm_cross");
        L.norm(m->dec[i].n_context, d + ".norm_context");
        L.attn(m->dec[i].cross_a,   d + ".cross");
        L.norm(m->dec[i].n_ffn,     d + ".norm_ffn");
        L.ffn (m->dec[i].ffn,       d);
    }

    m->loc.resize(hp.local_n_layers);
    for (int i = 0; i < hp.local_n_layers; i++) {
        const std::string p = "loc." + std::to_string(i);
        L.norm(m->loc[i].n_attn, p + ".norm_attn");
        L.attn(m->loc[i].attn,   p + ".attn");
        L.norm(m->loc[i].n_ffn,  p + ".norm_ffn");
        L.ffn (m->loc[i].ffn,    p);
    }

    m->audio_embed.resize(hp.num_codebooks);
    m->audio_head.resize(hp.num_codebooks);
    for (int c = 0; c < hp.num_codebooks; c++) {
        m->audio_embed[c] = L.get("audio_embed." + std::to_string(c) + ".weight");
        m->audio_head[c]  = L.get("audio_head."  + std::to_string(c) + ".weight");
    }
    return L.ok;
}

// The depth-embedding table arrives as (1, n_depth, d_model) from torch, which
// in ggml's reversed order is ne = (d_model, n_depth, 1). zt_local_step indexes
// it as a plain (d_model, n_depth) table via nb[1], so check that shape here
// instead of discovering a silently-wrong depth embedding as degraded audio.
static bool zt_check_depth_embed(const zerotts_context * m) {
    const ggml_tensor * t = m->depth_embed;
    if (t && t->ne[0] == m->hp.d_model && t->ne[1] == m->hp.n_depth && t->ne[2] == 1) {
        return true;
    }
    fprintf(stderr, "zerotts: local.depth_embed has an unexpected shape\n");
    return false;
}

static bool zt_alloc_cache(zerotts_context * m) {
    const zerotts_hparams & hp = m->hp;
    m->n_ctx = hp.n_voice_queries + 1 + ZT_MAX_FRAMES;

    const size_t n_tensors = (size_t) hp.n_layers * 4 + (size_t) hp.local_n_layers * 2 + 8;
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc*/ true };
    m->ctx_kv = ggml_init(ip);
    if (!m->ctx_kv) return false;

    auto mk = [&](int64_t n) { return ggml_new_tensor_1d(m->ctx_kv, GGML_TYPE_F32, n); };

    m->kv_k.resize(hp.n_layers); m->kv_v.resize(hp.n_layers);
    m->cx_k.resize(hp.n_layers); m->cx_v.resize(hp.n_layers);
    for (int i = 0; i < hp.n_layers; i++) {
        m->kv_k[i] = mk((int64_t) hp.d_model * m->n_ctx);
        m->kv_v[i] = mk((int64_t) hp.d_model * m->n_ctx);
        m->cx_k[i] = mk((int64_t) hp.d_model * ZT_MAX_TEXT);
        m->cx_v[i] = mk((int64_t) hp.d_model * ZT_MAX_TEXT);
    }
    m->lkv_k.resize(hp.local_n_layers); m->lkv_v.resize(hp.local_n_layers);
    for (int i = 0; i < hp.local_n_layers; i++) {
        m->lkv_k[i] = mk((int64_t) hp.d_model * hp.n_depth);
        m->lkv_v[i] = mk((int64_t) hp.d_model * hp.n_depth);
    }

    m->buf_kv = ggml_backend_alloc_ctx_tensors(m->ctx_kv, m->backend);
    return m->buf_kv != nullptr;
}

static zerotts_context * zt_finish_init(zerotts_context * m, loader & L,
                                        const uint8_t * blob, FILE * f) {
    if (!zt_load_common(m, L)) { zerotts_free(m); return nullptr; }
    if (!zt_check_depth_embed(m)) { zerotts_free(m); return nullptr; }

    m->buf_w = ggml_backend_alloc_ctx_tensors(m->ctx_w, m->backend);
    if (!m->buf_w) { zerotts_free(m); return nullptr; }

    // Copy weight data in. Both sources are laid out the same way: tensor i's
    // bytes start at the file's data offset plus the tensor's own offset.
    const size_t data_off = gguf_get_data_offset(L.gguf);
    std::vector<uint8_t> tmp;
    for (int64_t i = 0; i < gguf_get_n_tensors(L.gguf); i++) {
        const char * name = gguf_get_tensor_name(L.gguf, i);
        ggml_tensor * dst = ggml_get_tensor(m->ctx_w, name);
        if (!dst) continue;   // not needed by this runtime (e.g. an unused head)
        const size_t off = data_off + gguf_get_tensor_offset(L.gguf, i);
        const size_t n   = ggml_nbytes(dst);
        if (blob) {
            ggml_backend_tensor_set(dst, blob + off, 0, n);
        } else {
            tmp.resize(n);
            if (fseek(f, (long) off, SEEK_SET) != 0 || fread(tmp.data(), 1, n, f) != n) {
                fprintf(stderr, "zerotts: short read on tensor '%s'\n", name);
                zerotts_free(m);
                return nullptr;
            }
            ggml_backend_tensor_set(dst, tmp.data(), 0, n);
        }
    }

    if (!zt_alloc_cache(m)) { zerotts_free(m); return nullptr; }

    m->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    m->h.assign(m->hp.d_model, 0.0f);
    m->seen.assign((size_t) m->hp.num_codebooks * m->hp.codebook_size, 0);
    return m;
}

static int zt_default_threads(int n) {
    if (n > 0) return n;
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? (int) std::min(hc, 8u) : 4;
}

zerotts_context * zerotts_init_from_file(const char * path, int n_threads) {
    auto * m = new zerotts_context();
    m->n_threads = zt_default_threads(n_threads);
    m->backend = ggml_backend_cpu_init();
    if (!m->backend) { delete m; return nullptr; }

    loader L;
    L.m = m;
    gguf_init_params gp = { /*no_alloc*/ true, &L.meta };
    L.gguf = gguf_init_from_file(path, gp);
    if (!L.gguf) { fprintf(stderr, "zerotts: cannot open %s\n", path); zerotts_free(m); return nullptr; }

    FILE * f = fopen(path, "rb");
    if (!f) { gguf_free(L.gguf); zerotts_free(m); return nullptr; }
    zerotts_context * out = zt_finish_init(m, L, nullptr, f);
    fclose(f);
    gguf_free(L.gguf);
    ggml_free(L.meta);
    return out;
}

zerotts_context * zerotts_init_from_buffer(const void * data, size_t size, int n_threads) {
    // gguf has no parse-from-memory entry point, so the buffer is handed to it
    // through an in-memory FILE. The browser build always takes this path: the
    // weights arrive as an ArrayBuffer, and writing 200+ MB to the Emscripten
    // filesystem first would double the peak memory for no reason.
    auto * m = new zerotts_context();
    m->n_threads = zt_default_threads(n_threads);
    m->backend = ggml_backend_cpu_init();
    if (!m->backend) { delete m; return nullptr; }

    FILE * f = fmemopen(const_cast<void *>(data), size, "rb");
    if (!f) { zerotts_free(m); return nullptr; }

    loader L;
    L.m = m;
    gguf_init_params gp = { /*no_alloc*/ true, &L.meta };
    L.gguf = gguf_init_from_file_ptr(f, gp);
    if (!L.gguf) { fclose(f); zerotts_free(m); return nullptr; }

    zerotts_context * out = zt_finish_init(m, L, (const uint8_t *) data, nullptr);
    fclose(f);
    gguf_free(L.gguf);
    ggml_free(L.meta);
    return out;
}

void zerotts_free(zerotts_context * m) {
    if (!m) return;
    if (m->alloc)   ggml_gallocr_free(m->alloc);
    if (m->buf_w)   ggml_backend_buffer_free(m->buf_w);
    if (m->buf_kv)  ggml_backend_buffer_free(m->buf_kv);
    if (m->ctx_w)   ggml_free(m->ctx_w);
    if (m->ctx_kv)  ggml_free(m->ctx_kv);
    if (m->backend) ggml_backend_free(m->backend);
    delete m;
}

const zerotts_hparams * zerotts_hparams_of(const zerotts_context * m) { return &m->hp; }
int zerotts_max_frames(const zerotts_context *) { return ZT_MAX_FRAMES; }

void zerotts_sampling_defaults(zerotts_sampling * sp) {
    sp->text_temperature         = 1.0f;
    sp->text_topk                = 50;
    sp->audio_temperature        = 0.8f;
    sp->audio_topk               = 25;
    sp->audio_topp               = 0.95f;
    sp->audio_repetition_penalty = 1.2f;
}

void zerotts_timings(const zerotts_context * m, double * text_ms, double * frame_ms,
                     double * advance_ms, int * n_frames) {
    if (text_ms)    *text_ms    = m->t_text;
    if (frame_ms)   *frame_ms   = m->t_frame;
    if (advance_ms) *advance_ms = m->t_advance;
    if (n_frames)   *n_frames   = m->n_frames;
}

void zerotts_debug_hidden(const zerotts_context * m, float * out) {
    memcpy(out, m->h.data(), (size_t) m->hp.d_model * sizeof(float));
}

void zerotts_reset_timings(zerotts_context * m) {
    m->t_text = m->t_frame = m->t_advance = 0;
    m->n_frames = 0;
}

// ── graph scaffolding ───────────────────────────────────────────────────────

namespace {

// A scratch context for one graph build. ggml graphs here are small (a few
// hundred nodes) and cheap to rebuild, so each call constructs its own rather
// than caching graphs keyed on shape.
struct scratch {
    std::vector<uint8_t> mem;
    ggml_context * c  = nullptr;
    ggml_cgraph  * gf = nullptr;

    explicit scratch(size_t n_nodes = 4096) {
        mem.resize(ggml_tensor_overhead() * n_nodes +
                   ggml_graph_overhead_custom(n_nodes, false));
        ggml_init_params ip = { mem.size(), mem.data(), /*no_alloc*/ true };
        c  = ggml_init(ip);
        gf = ggml_new_graph_custom(c, n_nodes, false);
    }
    ~scratch() { if (c) ggml_free(c); }
};

} // namespace

static bool zt_compute(zerotts_context * m, scratch & s) {
    if (!ggml_gallocr_alloc_graph(m->alloc, s.gf)) {
        fprintf(stderr, "zerotts: graph allocation failed\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    return ggml_backend_graph_compute(m->backend, s.gf) == GGML_STATUS_SUCCESS;
}

// ── segment setup: text encoder + cross-attention K/V precompute ────────────

static bool zt_run_text(zerotts_context * m, const int32_t * ids, int n_text) {
    const zerotts_hparams & hp = m->hp;
    const int    d_head = hp.d_model / hp.n_heads;
    const size_t es     = sizeof(float);

    scratch s(8192);
    ggml_context * c = s.c;

    ggml_tensor * in_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_text);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_text);
    ggml_set_input(in_pos);

    ggml_tensor * x = ggml_get_rows(c, m->text_embed, in_ids);       // (d, L)

    for (int i = 0; i < hp.n_layers; i++) {
        const zt_txt_layer & l = m->txt[i];
        ggml_tensor * a = zt_layer_norm(c, x, l.n_attn, hp.layer_norm_eps);
        a = zt_attn_full(c, l.attn, a, in_pos, n_text, hp.n_heads, hp.d_model, hp.rope_theta);
        x = ggml_add(c, x, a);
        ggml_tensor * f = zt_layer_norm(c, x, l.n_ffn, hp.layer_norm_eps);
        x = ggml_add(c, x, zt_feed_forward(c, f, l.ffn));
    }
    ggml_tensor * states = ggml_norm(c, x, hp.layer_norm_eps);
    states = ggml_add(c, ggml_mul(c, states, m->text_ln_f_w), m->text_ln_f_b);

    // Project the text once per segment into every decoder layer's cross K/V.
    for (int i = 0; i < hp.n_layers; i++) {
        const zt_dec_layer & l = m->dec[i];
        ggml_tensor * ctxn = zt_layer_norm(c, states, l.n_context, hp.layer_norm_eps);

        ggml_tensor * K = ggml_reshape_3d(c, ggml_mul_mat(c, l.cross_a.k, ctxn),
                                          d_head, hp.n_heads, n_text);
        if (hp.cross_qk_norm) K = ggml_rms_norm(c, K, hp.qk_norm_eps);
        K = zt_rope(c, K, in_pos, d_head, hp.rope_theta);
        ggml_tensor * V = ggml_mul_mat(c, l.cross_a.v, ctxn);        // (d, L)

        ggml_tensor * k_dst = ggml_view_1d(c, m->cx_k[i], (int64_t) n_text * hp.d_model, 0);
        ggml_build_forward_expand(s.gf,
            ggml_cpy(c, ggml_reshape_2d(c, K, hp.d_model, n_text), k_dst));

        ggml_tensor * v_dst = ggml_view_2d(c, m->cx_v[i], n_text, hp.d_model,
                                           es * ZT_MAX_TEXT, 0);
        ggml_build_forward_expand(s.gf, ggml_cpy(c, ggml_transpose(c, V), v_dst));
    }

    if (!ggml_gallocr_alloc_graph(m->alloc, s.gf)) return false;
    ggml_backend_tensor_set(in_ids, ids, 0, n_text * sizeof(int32_t));
    std::vector<int32_t> pos(n_text);
    std::iota(pos.begin(), pos.end(), 0);
    ggml_backend_tensor_set(in_pos, pos.data(), 0, n_text * sizeof(int32_t));

    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    return ggml_backend_graph_compute(m->backend, s.gf) == GGML_STATUS_SUCCESS;
}

// The [voice] block: V latents, self-attention only. Backbone.init_prefix_cache
// gives these positions a bidirectional mask among themselves and zeroes their
// cross-attention output, which is exactly "run them as their own full-attention
// pass with no cross-attention" — so no mask tensor is built anywhere in this
// runtime. Getting that wrong is invisible: the voice latents would still decode
// to a voice, just not the one in the reference.
static bool zt_run_voice_prefix(zerotts_context * m, const float * voice) {
    const zerotts_hparams & hp = m->hp;
    const int V = hp.n_voice_queries;

    scratch s(2048);
    ggml_context * c = s.c;

    ggml_tensor * in_x = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.d_model, V);
    ggml_set_input(in_x);
    ggml_tensor * in_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, V);
    ggml_set_input(in_pos);

    ggml_tensor * x = in_x;
    for (int i = 0; i < hp.n_layers; i++) {
        const zt_dec_layer & l = m->dec[i];
        ggml_tensor * a = zt_layer_norm(c, x, l.n_self, hp.layer_norm_eps);
        a = zt_attn_cached(c, s.gf, l.self_a, a, m->kv_k[i], m->kv_v[i], in_pos,
                           V, /*kv_head*/ 0, /*n_kv*/ V, hp.n_heads, hp.d_model,
                           m->n_ctx, hp.rope_theta);
        x = ggml_add(c, x, a);
        // no cross-attention: cross_query_mask zeroes it for the voice block
        ggml_tensor * f = zt_layer_norm(c, x, l.n_ffn, hp.layer_norm_eps);
        x = ggml_add(c, x, zt_feed_forward(c, f, l.ffn));
    }
    ggml_build_forward_expand(s.gf, x);

    if (!ggml_gallocr_alloc_graph(m->alloc, s.gf)) return false;
    ggml_backend_tensor_set(in_x, voice, 0, (size_t) hp.d_model * V * sizeof(float));
    std::vector<int32_t> pos(V);
    std::iota(pos.begin(), pos.end(), 0);
    ggml_backend_tensor_set(in_pos, pos.data(), 0, V * sizeof(int32_t));

    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    return ggml_backend_graph_compute(m->backend, s.gf) == GGML_STATUS_SUCCESS;
}

// One global decoder position. `codes` null means the <soa> token (the cold
// start's last position); otherwise the input is <slot> plus the sum of the
// frame's K codebook embeddings, per Backbone._audio_frame_input.
static bool zt_run_decoder_step(zerotts_context * m, const int32_t * codes, int position) {
    const zerotts_hparams & hp = m->hp;

    scratch s(2048);
    ggml_context * c = s.c;

    const int n_ids = codes ? 1 + hp.num_codebooks : 1;
    ggml_tensor * in_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_ids);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_input(in_pos);

    ggml_tensor * ctrl_id = ggml_view_1d(c, in_ids, 1, 0);
    ggml_tensor * x = ggml_get_rows(c, m->text_embed, ctrl_id);      // (d, 1)
    if (codes) {
        for (int k = 0; k < hp.num_codebooks; k++) {
            ggml_tensor * id = ggml_view_1d(c, in_ids, 1, (size_t)(1 + k) * sizeof(int32_t));
            x = ggml_add(c, x, ggml_get_rows(c, m->audio_embed[k], id));
        }
    }

    const int kv_head = m->n_kv;
    const int n_kv    = m->n_kv + 1;

    for (int i = 0; i < hp.n_layers; i++) {
        const zt_dec_layer & l = m->dec[i];
        ggml_tensor * a = zt_layer_norm(c, x, l.n_self, hp.layer_norm_eps);
        a = zt_attn_cached(c, s.gf, l.self_a, a, m->kv_k[i], m->kv_v[i], in_pos,
                           1, kv_head, n_kv, hp.n_heads, hp.d_model, m->n_ctx,
                           hp.rope_theta);
        x = ggml_add(c, x, a);

        ggml_tensor * q = zt_layer_norm(c, x, l.n_cross, hp.layer_norm_eps);
        q = zt_cross_attn(c, l.cross_a, q, m->cx_k[i], m->cx_v[i], in_pos, 1,
                          m->n_text, hp.n_heads, hp.d_model, ZT_MAX_TEXT,
                          hp.cross_qk_norm != 0, hp.qk_norm_eps, hp.rope_theta);
        x = ggml_add(c, x, q);

        ggml_tensor * f = zt_layer_norm(c, x, l.n_ffn, hp.layer_norm_eps);
        x = ggml_add(c, x, zt_feed_forward(c, f, l.ffn));
    }

    ggml_tensor * out = ggml_norm(c, x, hp.layer_norm_eps);
    out = ggml_add(c, ggml_mul(c, out, m->ln_f_w), m->ln_f_b);
    ggml_set_output(out);
    ggml_build_forward_expand(s.gf, out);

    if (!ggml_gallocr_alloc_graph(m->alloc, s.gf)) return false;

    std::vector<int32_t> ids(n_ids);
    ids[0] = codes ? SLOT_ID : SOA_ID;
    for (int k = 0; k < n_ids - 1; k++) ids[1 + k] = codes[k];
    ggml_backend_tensor_set(in_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    const int32_t p = position;
    ggml_backend_tensor_set(in_pos, &p, 0, sizeof(p));

    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    if (ggml_backend_graph_compute(m->backend, s.gf) != GGML_STATUS_SUCCESS) return false;

    ggml_backend_tensor_get(out, m->h.data(), 0, (size_t) hp.d_model * sizeof(float));
    m->n_kv += 1;
    return true;
}

// ── local (depth) transformer ───────────────────────────────────────────────

// One depth position of the frame decoder, plus its output head. This is the
// hot loop: n_depth (17) of these per frame, each reading the whole local
// transformer, which is why the local weights dominate the per-frame byte
// count and why quantizing them is where the speedup comes from.
//
// Depth 0 takes the global hidden directly and reads the control head; depths
// 1..K take an embedding lookup and read codebook (depth-1)'s head. No mask:
// depths are generated in order, so the cache is already causal.
static bool zt_local_step(zerotts_context * m, int depth,
                          const float * hidden_in,
                          ggml_tensor * embed_table, int32_t embed_id,
                          float * out_logits) {
    const zerotts_hparams & hp = m->hp;
    const int n_logits = (depth == 0) ? 2 : hp.codebook_size;

    scratch s(1024);
    ggml_context * c = s.c;

    ggml_tensor * in_h  = nullptr;
    ggml_tensor * in_id = nullptr;
    ggml_tensor * x     = nullptr;
    if (depth == 0) {
        in_h = ggml_new_tensor_2d(c, GGML_TYPE_F32, hp.d_model, 1);
        ggml_set_input(in_h);
        x = in_h;
    } else {
        in_id = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_input(in_id);
        x = ggml_get_rows(c, embed_table, in_id);
    }

    // Learned depth position embedding — the local transformer uses no RoPE.
    ggml_tensor * de = ggml_view_2d(c, m->depth_embed, hp.d_model, 1,
                                    m->depth_embed->nb[1],
                                    (size_t) depth * m->depth_embed->nb[1]);
    x = ggml_add(c, x, de);

    for (int i = 0; i < hp.local_n_layers; i++) {
        const zt_loc_layer & l = m->loc[i];
        ggml_tensor * a = zt_layer_norm(c, x, l.n_attn, hp.layer_norm_eps);
        a = zt_attn_cached(c, s.gf, l.attn, a, m->lkv_k[i], m->lkv_v[i],
                           /*pos*/ nullptr, 1, /*kv_head*/ depth, /*n_kv*/ depth + 1,
                           hp.local_n_heads, hp.d_model, hp.n_depth, hp.rope_theta);
        x = ggml_add(c, x, a);
        ggml_tensor * f = zt_layer_norm(c, x, l.n_ffn, hp.layer_norm_eps);
        x = ggml_add(c, x, zt_feed_forward(c, f, l.ffn));
    }
    ggml_tensor * hid = ggml_norm(c, x, hp.layer_norm_eps);
    hid = ggml_add(c, ggml_mul(c, hid, m->local_ln_f_w), m->local_ln_f_b);

    ggml_tensor * logits = nullptr;
    ggml_tensor * in_ctrl_ids = nullptr;
    if (depth == 0) {
        // The control head is tied to the 8192-row embedding table, but only
        // <slot> and <eoa> are ever legal here (export_onnx.py restricts the
        // logits to those two before sampling). Gathering the two rows turns a
        // 8192x768 matvec into a 2x768 one — ~6 MB/frame of reads that the
        // ONNX graph pays and this does not.
        in_ctrl_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2);
        ggml_set_input(in_ctrl_ids);
        ggml_tensor * rows = ggml_get_rows(c, m->text_embed, in_ctrl_ids);  // (d, 2)
        logits = ggml_mul_mat(c, rows, hid);                                // (2, 1)
    } else {
        logits = ggml_mul_mat(c, m->audio_head[depth - 1], hid);            // (C, 1)
    }
    ggml_set_output(logits);
    ggml_build_forward_expand(s.gf, logits);

    if (!ggml_gallocr_alloc_graph(m->alloc, s.gf)) return false;
    if (depth == 0) {
        ggml_backend_tensor_set(in_h, hidden_in, 0, (size_t) hp.d_model * sizeof(float));
        const int32_t ctrl_ids[2] = { SLOT_ID, EOA_ID };
        ggml_backend_tensor_set(in_ctrl_ids, ctrl_ids, 0, sizeof(ctrl_ids));
    } else {
        ggml_backend_tensor_set(in_id, &embed_id, 0, sizeof(embed_id));
    }

    ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
    if (ggml_backend_graph_compute(m->backend, s.gf) != GGML_STATUS_SUCCESS) return false;

    ggml_backend_tensor_get(logits, out_logits, 0, (size_t) n_logits * sizeof(float));
    return true;
}

// ── sampling ────────────────────────────────────────────────────────────────

// Port of inference/export_onnx.py::sample_topk_topp_with_random_u — top-k,
// then nucleus, then walk the CDF against an externally supplied uniform.
// Matching it exactly (rather than reaching for a tidier equivalent) is what
// lets the same seed produce the same frame codes here, in the ONNX runtime and
// in PyTorch, which is the only practical way to test a port of this model.
static int32_t zt_sample(const float * logits, int n, float u,
                         float temperature, int topk, float topp) {
    const float t = std::max(temperature, 1e-4f);
    const int   k = std::max(1, std::min(topk, n));

    // Ranks at or beyond k are -inf and contribute nothing, so only the top k
    // are materialized. Ties break towards the lower index, matching a stable
    // descending sort.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;
                      });
    order.resize(k);

    std::vector<float> p(k);
    const float top = logits[order[0]] / t;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        p[i] = expf(logits[order[i]] / t - top);
        sum += p[i];
    }
    for (int i = 0; i < k; i++) p[i] /= sum;

    // Nucleus: drop a rank once the mass *before* it already covers topp, so the
    // rank that crosses the threshold is kept.
    const float pp = std::min(std::max(topp, 1e-6f), 1.0f);
    float cum = 0.0f, kept = 0.0f;
    std::vector<float> q(k, 0.0f);
    for (int i = 0; i < k; i++) {
        if (cum <= pp) { q[i] = p[i]; kept += p[i]; }
        cum += p[i];
    }
    if (kept <= 0.0f) { q[0] = 1.0f; kept = 1.0f; }

    const float uu = std::min(std::max(u, 0.0f), 0.999999f);
    float cdf = 0.0f;
    for (int i = 0; i < k; i++) {
        cdf += q[i] / kept;
        if (cdf > uu) return (int32_t) order[i];
    }
    // torch.argmax over an all-zero "first_over" mask returns 0; mirror that
    // rather than inventing a different fallback.
    return (int32_t) order[0];
}

static void zt_apply_repetition_penalty(float * logits, int n,
                                        const uint8_t * seen, float penalty) {
    if (penalty == 1.0f) return;
    for (int i = 0; i < n; i++) {
        if (!seen[i]) continue;
        logits[i] = logits[i] < 0.0f ? logits[i] * penalty : logits[i] / penalty;
    }
}

// ── public API ──────────────────────────────────────────────────────────────

int zerotts_begin(zerotts_context * m, const int32_t * text_ids, int n_text,
                  const float * voice_emb) {
    if (!m || !text_ids || !voice_emb || n_text <= 0) return -1;
    if (n_text > ZT_MAX_TEXT) {
        fprintf(stderr, "zerotts: text of %d tokens exceeds the %d-token cache\n",
                n_text, ZT_MAX_TEXT);
        return -1;
    }

    const double t0 = now_ms();
    m->n_text = n_text;
    m->n_kv   = 0;
    std::fill(m->seen.begin(), m->seen.end(), 0);

    if (!zt_run_text(m, text_ids, n_text)) return -1;
    if (!zt_run_voice_prefix(m, voice_emb)) return -1;
    m->n_kv = m->hp.n_voice_queries;

    // <soa> sits at position V and is the first position that cross-attends to
    // the text; its hidden state is what predicts frame 0.
    if (!zt_run_decoder_step(m, nullptr, m->hp.n_voice_queries)) return -1;

    m->t_text += now_ms() - t0;
    return 0;
}

// `forced`, when given, replaces the sampled code in the depth decoder's
// feedback path while out_codes still reports the sampled one — see
// zerotts_frame_forced.
static int zt_frame_impl(zerotts_context * m, bool forbid_eoa, const zerotts_sampling * sp,
                         float ctrl_u, const float * audio_u, const int32_t * forced,
                         int32_t * out_codes, int * out_is_eoa) {
    if (!m || !sp || !audio_u || !out_codes || !out_is_eoa) return -1;
    const zerotts_hparams & hp = m->hp;
    const double t0 = now_ms();

    // depth 0 -> the control channel, restricted to {<slot>, <eoa>}.
    float ctrl[2] = { 0.0f, 0.0f };
    if (!zt_local_step(m, 0, m->h.data(), nullptr, 0, ctrl)) return -1;
    if (forbid_eoa) ctrl[1] = -INFINITY;
    // Only two candidates exist, so top-k is clamped to 2 and top-p is a no-op.
    const int32_t picked = zt_sample(ctrl, 2, ctrl_u, sp->text_temperature,
                                     std::min(std::max(sp->text_topk, 1), 2), 1.0f);
    *out_is_eoa = (picked == 1) ? 1 : 0;

    // depths 1..K -> the codebooks. Depth 1's input is always <slot>: the audio
    // channels do not see the sampled control token (see _local_generate's
    // docstring and LocalFrameDecodeONNX.forward, which agree on this).
    std::vector<float> logits(hp.codebook_size);
    ggml_tensor * table = m->text_embed;
    int32_t       id    = SLOT_ID;
    for (int cbk = 0; cbk < hp.num_codebooks; cbk++) {
        if (!zt_local_step(m, cbk + 1, nullptr, table, id, logits.data())) return -1;

        uint8_t * seen_c = m->seen.data() + (size_t) cbk * hp.codebook_size;
        zt_apply_repetition_penalty(logits.data(), hp.codebook_size, seen_c,
                                    sp->audio_repetition_penalty);
        const int32_t code = zt_sample(logits.data(), hp.codebook_size, audio_u[cbk],
                                       sp->audio_temperature, sp->audio_topk,
                                       sp->audio_topp);
        const int32_t fed = forced ? forced[cbk] : code;
        seen_c[fed] = 1;
        out_codes[cbk] = code;

        table = m->audio_embed[cbk];
        id    = fed;
    }

    m->t_frame += now_ms() - t0;
    m->n_frames++;
    return 0;
}

int zerotts_frame(zerotts_context * m, bool forbid_eoa, const zerotts_sampling * sp,
                  float ctrl_u, const float * audio_u,
                  int32_t * out_codes, int * out_is_eoa) {
    return zt_frame_impl(m, forbid_eoa, sp, ctrl_u, audio_u, nullptr, out_codes, out_is_eoa);
}

int zerotts_frame_forced(zerotts_context * m, bool forbid_eoa, const zerotts_sampling * sp,
                         float ctrl_u, const float * audio_u, const int32_t * forced,
                         int32_t * out_codes, int * out_is_eoa) {
    return zt_frame_impl(m, forbid_eoa, sp, ctrl_u, audio_u, forced, out_codes, out_is_eoa);
}

int zerotts_advance(zerotts_context * m, const int32_t * codes, int frame_index) {
    if (!m || !codes) return -1;
    const int position = m->hp.n_voice_queries + 1 + frame_index;
    if (position != m->n_kv) {
        // The cache is append-only and the caller owns t; a mismatch means the
        // loop skipped or repeated a frame, which would otherwise show up only
        // as RoPE positions quietly drifting out of step.
        fprintf(stderr, "zerotts: frame %d wants position %d but the cache holds %d\n",
                frame_index, position, m->n_kv);
        return -1;
    }
    if (position >= m->n_ctx) return -1;

    const double t0 = now_ms();
    if (!zt_run_decoder_step(m, codes, position)) return -1;
    m->t_advance += now_ms() - t0;
    return 0;
}
