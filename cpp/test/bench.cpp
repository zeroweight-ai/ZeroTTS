// Native driver for the ggml runtime: generates a take and prints the frame
// codes as JSON, so the output can be diffed against the ONNX path's
// js_frames.json (js/test/frames.mjs) for the same text, voice and seed.
//
//   zerotts-bench model.gguf voice.bin text_ids.txt [seed] [threads]
//
// text_ids.txt is whitespace-separated BPE ids — dump them from the JS
// tokenizer so both runtimes are fed exactly the same tokens and any
// difference is the model, not the text front end.

#include "zerotts.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// mulberry32, identical to js/src/rng.ts — the draws are graph inputs, so the
// two runtimes must consume the same sequence to be comparable at all.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {
        s += 0x6d2b79f5u;
        uint32_t t = s;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + (t ^ (t >> 7)) * (t | 61u);
        return (float) (((double) ((t ^ (t >> 14)))) / 4294967296.0);
    }
};

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf voice.bin text_ids.txt [seed] [threads]\n", argv[0]);
        return 1;
    }
    const uint32_t seed    = argc > 4 ? (uint32_t) strtoul(argv[4], nullptr, 10) : 1234u;
    const int      threads = argc > 5 ? atoi(argv[5]) : 0;

    // Defaults mirror js/src/types.ts DEFAULT_SAMPLING.
    const int min_frames      = 4;
    const int max_frames      = 1500;
    const int eoa_extra       = 1;

    std::vector<int32_t> ids;
    {
        std::ifstream f(argv[3]);
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }
        for (int v; f >> v; ) ids.push_back(v);
    }

    std::vector<float> voice;
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        f.seekg(0, std::ios::end);
        const std::streamsize n = f.tellg();
        f.seekg(0);
        voice.resize((size_t) n / sizeof(float));
        f.read((char *) voice.data(), n);
    }

    const auto t_load0 = std::chrono::steady_clock::now();
    zerotts_context * ctx = zerotts_init_from_file(argv[1], threads);
    if (!ctx) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_load0).count();

    const zerotts_hparams * hp = zerotts_hparams_of(ctx);
    const int K = hp->num_codebooks;
    if ((int) voice.size() != hp->n_voice_queries * hp->d_model) {
        fprintf(stderr, "voice.bin has %zu floats, expected %d\n",
                voice.size(), hp->n_voice_queries * hp->d_model);
        return 1;
    }

    zerotts_sampling sp;
    zerotts_sampling_defaults(&sp);

    const auto t0 = std::chrono::steady_clock::now();
    if (zerotts_begin(ctx, ids.data(), (int) ids.size(), voice.data()) != 0) {
        fprintf(stderr, "zerotts_begin failed\n");
        return 1;
    }

    // ZT_DUMP=h prints the global hidden after the prefix, for numeric
    // comparison against the ONNX prefix_step output.
    if (const char * d = getenv("ZT_DUMP")) {
        if (std::string(d) == "h") {
            std::vector<float> h(hp->d_model);
            zerotts_debug_hidden(ctx, h.data());
            for (int i = 0; i < hp->d_model; i++) printf("%s%.6f", i ? " " : "", h[i]);
            printf("\n");
            return 0;
        }
    }

    Rng rng(seed);
    std::vector<int32_t> codes(K);
    std::vector<float>   au(K);
    std::vector<std::vector<int32_t>> frames;

    int tail_left = -1;   // -1 = <eoa> has not fired yet
    for (int t = 0; ; t++) {
        const bool forbid_eoa = (t < min_frames) || (tail_left >= 0);
        const float cu = rng.next();
        for (int i = 0; i < K; i++) au[i] = rng.next();

        int is_eoa = 0;
        if (zerotts_frame(ctx, forbid_eoa, &sp, cu, au.data(), codes.data(), &is_eoa) != 0) {
            fprintf(stderr, "zerotts_frame failed at t=%d\n", t);
            return 1;
        }
        if (tail_left < 0 && is_eoa) tail_left = eoa_extra;
        if ((tail_left >= 0 && tail_left <= 0) || t >= max_frames) break;

        frames.push_back(codes);
        if (tail_left >= 0 && --tail_left <= 0) break;

        if (zerotts_advance(ctx, codes.data(), t) != 0) {
            fprintf(stderr, "zerotts_advance failed at t=%d\n", t);
            return 1;
        }
    }
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    double t_text = 0, t_frame = 0, t_adv = 0;
    int nf = 0;
    zerotts_timings(ctx, &t_text, &t_frame, &t_adv, &nf);

    const double audio_s = frames.size() / 12.5;
    fprintf(stderr, "load           %8.0f ms\n", load_ms);
    fprintf(stderr, "%zu frames = %.2f s audio in %.2f s wall\n", frames.size(), audio_s, wall);
    fprintf(stderr, "RTF = %.3f  (%.2fx realtime)\n", wall / audio_s, audio_s / wall);
    fprintf(stderr, "  begin (text+prefix) %8.1f ms\n", t_text);
    fprintf(stderr, "  frame  (local x%2d)  %8.1f ms   %.2f ms/frame\n",
            hp->n_depth, t_frame, t_frame / nf);
    fprintf(stderr, "  advance (decoder)   %8.1f ms   %.2f ms/frame\n",
            t_adv, t_adv / nf);

    printf("[");
    for (size_t i = 0; i < frames.size(); i++) {
        printf("%s[", i ? "," : "");
        for (int k = 0; k < K; k++) printf("%s%d", k ? "," : "", frames[i][k]);
        printf("]");
    }
    printf("]\n");

    zerotts_free(ctx);
    return 0;
}
