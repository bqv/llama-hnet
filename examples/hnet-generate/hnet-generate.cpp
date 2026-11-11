#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf prompt [n-tokens] [--greedy]\n", argv[0]);
        return 2;
    }

    const char * model_path = argv[1];
    std::string text = argv[2];
    int n_predict = argc > 3 ? std::atoi(argv[3]) : 24;
    float temperature = 0.0f;
    uint32_t seed = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t) std::atoi(argv[++i]);
    }
    std::mt19937 rng(seed);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    const int n_ctx = std::max<int>(256, (int) text.size() + n_predict + 8);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 256;
    cp.n_ubatch = 256;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }

    std::fprintf(stderr, "prompt: %s\n", text.c_str());

    for (int step = 0; step < n_predict; ++step) {
        llama_batch batch = llama_batch_init(256, 0, 1);
        batch.n_tokens = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            batch.token[batch.n_tokens] = (unsigned char) text[i];
            batch.pos[batch.n_tokens] = (int32_t) i;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = 1;
            batch.n_tokens++;
        }

        if (llama_encode(ctx, batch) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }

        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (!logits) { std::fprintf(stderr, "no logits\n"); return 1; }

        int best = 0;
        if (temperature <= 0.0f) {
            for (int v = 1; v < 256; ++v) {
                if (logits[v] > logits[best]) best = v;
            }
        } else {
            std::vector<double> probs(256);
            double maxv = -1e30;
            for (int v = 0; v < 256; ++v) maxv = std::max<double>(maxv, logits[v]);
            double sum = 0.0;
            for (int v = 0; v < 256; ++v) {
                probs[v] = std::exp((logits[v] - maxv) / temperature);
                sum += probs[v];
            }
            double r = std::uniform_real_distribution<double>(0.0, sum)(rng);
            for (int v = 0; v < 256; ++v) {
                r -= probs[v];
                if (r <= 0.0) { best = v; break; }
            }
        }

        text.push_back((char) best);
        std::fprintf(stderr, "\r%d/%d chunks=%zu", step + 1, n_predict, llama_hnet_n_boundaries(ctx));
        llama_batch_free(batch);
    }

    std::fprintf(stderr, "\n");
    std::printf("%s\n", text.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
