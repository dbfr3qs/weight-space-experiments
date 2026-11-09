// actdump: run prompts through llama.cpp, dump named activations (all positions) to a binary file.
// env: ACT_NAMES="l_out-20,ffn_swiglu-8"  ACT_OUT=out.bin ; prompts file via -f, one per line, "\n" escapes.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

struct cbd { std::set<std::string> names; FILE * f; int32_t idx; std::vector<float> buf; };

static void wrec(FILE * f, int32_t idx, const std::string & name, int32_t ne0, int32_t ne1, const void * data, size_t nbytes) {
    int32_t h[4] = {idx, (int32_t)name.size(), ne0, ne1};
    fwrite(h, 4, 4, f); fwrite(name.data(), 1, name.size(), f); fwrite(data, 1, nbytes, f);
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (cbd *) ud;
    bool want = d->names.count(t->name) > 0;
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { LOG_ERR("skip %s type %d\n", t->name, t->type); return true; }
    size_t n = ggml_nelements(t);
    d->buf.resize(n);
    ggml_backend_tensor_get(t, d->buf.data(), 0, n * sizeof(float));
    wrec(d->f, d->idx, t->name, t->ne[0], (int32_t)(n / t->ne[0]), d->buf.data(), n * sizeof(float));
    return true;
}

static std::string unesc(const std::string & s) {
    std::string o; for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i+1] == 'n') { o += '\n'; i++; } else o += s[i]; }
    return o;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    cbd d; d.idx = 0;
    { std::string s = getenv("ACT_NAMES") ? getenv("ACT_NAMES") : "";
      size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
        if (q > p) d.names.insert(s.substr(p, q - p)); p = q + 1; } }
    d.f = fopen(getenv("ACT_OUT") ? getenv("ACT_OUT") : "act.bin", "wb");
    llama_backend_init();
    params.cb_eval = cb; params.cb_eval_user_data = &d; params.warmup = false;
    auto init = common_init_from_params(params);
    auto * model = init->model(); auto * ctx = init->context();
    if (!model || !ctx) return 1;
    std::vector<std::string> prompts;
    { std::ifstream in(params.prompt_file); std::string line; while (std::getline(in, line)) if (!line.empty()) prompts.push_back(unesc(line)); }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (size_t i = 0; i < prompts.size(); i++) {
        d.idx = i;
        llama_memory_clear(llama_get_memory(ctx), true);
        auto toks = common_tokenize(ctx, prompts[i], llama_vocab_get_add_bos(vocab), true);
        wrec(d.f, i, "tokens", toks.size(), 1, toks.data(), toks.size() * 4);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) { LOG_ERR("decode failed %zu\n", i); return 1; }
        llama_synchronize(ctx);
        fprintf(stderr, "prompt %zu/%zu tokens %zu\n", i + 1, prompts.size(), toks.size());
    }
    fclose(d.f);
    llama_backend_free();
    return 0;
}
