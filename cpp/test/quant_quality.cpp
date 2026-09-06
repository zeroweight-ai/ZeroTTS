// How much does quantization change what the model says?
//
// Two models sampling independently diverge at the first differing token, after
// which every later frame is a different utterance and comparing them measures
// nothing. So this replays a reference model's exact code sequence through a
// second model — forcing the depth decoder's inputs to match at every channel of
// every frame — and reports how often the second model would have drawn a
// different code. That number is attributable to the weights alone.
//
//   zerotts-quality reference.gguf candidate.gguf voice.bin text_ids.txt [seed]

#include "zerotts.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {
        s += 0x6d2b79f5u;
        uint32_t t = s;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + (t ^ (t >> 7)) * (t | 61u);
        return (float) (((double) (t ^ (t >> 14))) / 4294967296.0);
    }
};

static std::vector<int32_t> read_ids(const char * p) {
    std::vector<int32_t> v;
    std::ifstream f(p);
    for (int x; f >> x; ) v.push_back(x);
    return v;
}

static std::vector<float> read_f32(const char * p) {
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<float> v((size_t) n / sizeof(float));
    f.read((char *) v.data(), n);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s reference.gguf candidate.gguf voice.bin ids.txt [seed]\n", argv[0]);
        return 1;
    }
    const uint32_t seed = argc > 5 ? (uint32_t) strtoul(argv[5], nullptr, 10) : 1234u;

    const std::vector<int32_t> ids   = read_ids(argv[4]);
    const std::vector<float>   voice = read_f32(argv[3]);

    zerotts_context * ref = zerotts_init_from_file(argv[1], 0);
    zerotts_context * cnd = zerotts_init_from_file(argv[2], 0);
    if (!ref || !cnd) { fprintf(stderr, "load failed\n"); return 1; }

    const int K = zerotts_hparams_of(ref)->num_codebooks;
    zerotts_sampling sp;
    zerotts_sampling_defaults(&sp);

    if (zerotts_begin(ref, ids.data(), (int) ids.size(), voice.data()) != 0 ||
        zerotts_begin(cnd, ids.data(), (int) ids.size(), voice.data()) != 0) {
        fprintf(stderr, "begin failed\n");
        return 1;
    }

    Rng rng(seed);
    std::vector<int32_t> ref_codes(K), cnd_codes(K), au(K);
    long total = 0, differ = 0, eoa_differ = 0, frames = 0;
    int tail = -1;

    for (int t = 0; ; t++) {
        const bool forbid = (t < 4) || (tail >= 0);
        const float cu = rng.next();
        for (int i = 0; i < K; i++) au[i] = rng.next();

        int ref_eoa = 0, cnd_eoa = 0;
        if (zerotts_frame(ref, forbid, &sp, cu, (float *) au.data(), ref_codes.data(), &ref_eoa) != 0) return 1;
        // The candidate is driven along the reference's codes, not its own.
        if (zerotts_frame_forced(cnd, forbid, &sp, cu, (float *) au.data(), ref_codes.data(),
                                 cnd_codes.data(), &cnd_eoa) != 0) return 1;

        if (ref_eoa != cnd_eoa) eoa_differ++;
        for (int k = 0; k < K; k++) { total++; if (ref_codes[k] != cnd_codes[k]) differ++; }

        if (tail < 0 && ref_eoa) tail = 1;
        if ((tail >= 0 && tail <= 0) || t >= 1500) break;
        frames++;
        if (tail >= 0 && --tail <= 0) break;

        if (zerotts_advance(ref, ref_codes.data(), t) != 0) return 1;
        if (zerotts_advance(cnd, ref_codes.data(), t) != 0) return 1;
    }

    printf("%-28s vs %-28s\n", argv[1], argv[2]);
    printf("  %ld frames, %ld sampled codes\n", frames, total);
    printf("  codes drawn differently : %ld  (%.2f%%)\n", differ, 100.0 * differ / total);
    printf("  control channel differs : %ld frames\n", eoa_differ);

    zerotts_free(ref);
    zerotts_free(cnd);
    return 0;
}
