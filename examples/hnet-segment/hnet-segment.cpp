#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string read_all(FILE * f) {
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.gguf [text-file] [--json]\n", argv[0]);
        return 2;
    }

    bool json = false;
    const char * text_path = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) json = true;
        else text_path = argv[i];
    }

    const std::string text = text_path ? read_all(std::fopen(text_path, "rb")) : read_all(stdin);
    if (text.empty()) { std::fprintf(stderr, "empty input\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    const int n_tokens = (int) text.size();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_tokens + 8;
    cp.n_batch = n_tokens + 8;
    cp.n_ubatch = n_tokens + 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }

    llama_batch batch = llama_batch_init(n_tokens + 8, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[batch.n_tokens] = (unsigned char) text[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = 1;
        batch.n_tokens++;
    }

    if (llama_encode(ctx, batch) != 0) { std::fprintf(stderr, "encode failed\n"); return 1; }

    const size_t n_bounds = llama_hnet_n_boundaries(ctx);
    std::vector<int32_t> positions(n_bounds);
    std::vector<float> probs((size_t) n_tokens);
    llama_hnet_boundaries(ctx, positions.data(), n_bounds);
    llama_hnet_probs(ctx, probs.data(), probs.size());

    if (json) {
        std::printf("{\"bytes\":%d,\"chunks\":%zu,\"boundaries\":[", n_tokens, n_bounds);
        for (size_t i = 0; i < n_bounds; ++i) std::printf("%s%d", i ? "," : "", positions[i]);
        std::printf("],\"probs\":[");
        for (int i = 0; i < n_tokens; ++i) std::printf("%s%.6f", i ? "," : "", probs[i]);
        std::printf("]}\n");
    } else {
        std::string out;
        size_t next = 0;
        for (int i = 0; i < n_tokens; ++i) {
            if (next < n_bounds && positions[next] == i) {
                out += "|";
                ++next;
            }
            out += text[i];
        }
        std::printf("%s\n", out.c_str());
        std::printf("chunks %zu of %d bytes\n", n_bounds, n_tokens);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
