// Emscripten entry points for the browser build.
//
// Deliberately a flat C surface over heap pointers rather than embind: every
// call here is on the per-frame hot path, and the values crossing the boundary
// are small typed arrays that JS can write straight into the WASM heap. The
// JS wrapper in js/src/ggmlBackend.ts owns those staging buffers.

#include "zerotts.h"

#include <emscripten.h>
#include <cstdlib>

extern "C" {

EMSCRIPTEN_KEEPALIVE void * zt_alloc(int n)      { return malloc((size_t) n); }
EMSCRIPTEN_KEEPALIVE void   zt_dealloc(void * p) { free(p); }

// `data` is a pointer into the WASM heap holding the whole .gguf. The caller
// may free it as soon as this returns — the weights have been copied into the
// backend buffer by then.
EMSCRIPTEN_KEEPALIVE zerotts_context * zt_load(void * data, int size, int threads) {
    return zerotts_init_from_buffer(data, (size_t) size, threads);
}

EMSCRIPTEN_KEEPALIVE void zt_free(zerotts_context * c) { zerotts_free(c); }

// Field selectors for zt_hparam, kept in sync with the JS wrapper's enum.
enum {
    ZT_HP_D_MODEL = 0, ZT_HP_N_HEADS, ZT_HP_N_LAYERS, ZT_HP_D_FF,
    ZT_HP_LOCAL_N_LAYERS, ZT_HP_LOCAL_N_HEADS, ZT_HP_LOCAL_D_FF, ZT_HP_N_DEPTH,
    ZT_HP_NUM_CODEBOOKS, ZT_HP_CODEBOOK_SIZE, ZT_HP_VOCAB_SIZE,
    ZT_HP_N_VOICE_QUERIES, ZT_HP_MAX_FRAMES,
};

EMSCRIPTEN_KEEPALIVE int zt_hparam(zerotts_context * c, int which) {
    const zerotts_hparams * h = zerotts_hparams_of(c);
    switch (which) {
        case ZT_HP_D_MODEL:        return h->d_model;
        case ZT_HP_N_HEADS:        return h->n_heads;
        case ZT_HP_N_LAYERS:       return h->n_layers;
        case ZT_HP_D_FF:           return h->d_ff;
        case ZT_HP_LOCAL_N_LAYERS: return h->local_n_layers;
        case ZT_HP_LOCAL_N_HEADS:  return h->local_n_heads;
        case ZT_HP_LOCAL_D_FF:     return h->local_d_ff;
        case ZT_HP_N_DEPTH:        return h->n_depth;
        case ZT_HP_NUM_CODEBOOKS:  return h->num_codebooks;
        case ZT_HP_CODEBOOK_SIZE:  return h->codebook_size;
        case ZT_HP_VOCAB_SIZE:     return h->vocab_size;
        case ZT_HP_N_VOICE_QUERIES:return h->n_voice_queries;
        case ZT_HP_MAX_FRAMES:     return zerotts_max_frames(c);
        default:                   return -1;
    }
}

EMSCRIPTEN_KEEPALIVE int zt_begin(zerotts_context * c, int * ids, int n, float * voice) {
    return zerotts_begin(c, ids, n, voice);
}

// Sampling params are passed as scalars rather than a struct so the JS side
// never has to know the struct's layout or padding.
EMSCRIPTEN_KEEPALIVE int zt_frame(zerotts_context * c, int forbid_eoa,
                                  float text_temperature, int text_topk,
                                  float audio_temperature, int audio_topk,
                                  float audio_topp, float audio_repetition_penalty,
                                  float ctrl_u, float * audio_u,
                                  int * out_codes, int * out_is_eoa) {
    zerotts_sampling sp;
    sp.text_temperature         = text_temperature;
    sp.text_topk                = text_topk;
    sp.audio_temperature        = audio_temperature;
    sp.audio_topk               = audio_topk;
    sp.audio_topp               = audio_topp;
    sp.audio_repetition_penalty = audio_repetition_penalty;
    return zerotts_frame(c, forbid_eoa != 0, &sp, ctrl_u, audio_u, out_codes, out_is_eoa);
}

EMSCRIPTEN_KEEPALIVE int zt_advance(zerotts_context * c, int * codes, int t) {
    return zerotts_advance(c, codes, t);
}

// out: [text_ms, frame_ms, advance_ms, n_frames]
EMSCRIPTEN_KEEPALIVE void zt_timings(zerotts_context * c, double * out) {
    int n = 0;
    zerotts_timings(c, &out[0], &out[1], &out[2], &n);
    out[3] = (double) n;
}

EMSCRIPTEN_KEEPALIVE void zt_reset_timings(zerotts_context * c) { zerotts_reset_timings(c); }

} // extern "C"
