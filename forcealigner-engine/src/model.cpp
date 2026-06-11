#include "../include/model.h"

#include "../include/safetensors.h"
#include "../include/cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)
#endif

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error [%s:%d]: status=%d\n", \
                __FILE__, __LINE__, static_cast<int>(_s)); \
        return -1; \
    } \
} while(0)

namespace {

bool read_text_file(const char *path, std::string &out) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "aligner_model_load: failed to open %s: %s\n", path, std::strerror(errno));
        return false;
    }
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    long size = std::ftell(fp);
    if (size < 0) {
        std::fclose(fp);
        return false;
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fclose(fp);
        return false;
    }
    out.resize(static_cast<size_t>(size));
    size_t got = out.empty() ? 0 : std::fread(&out[0], 1, out.size(), fp);
    std::fclose(fp);
    if (got != out.size()) {
        fprintf(stderr, "aligner_model_load: failed to read %s\n", path);
        return false;
    }
    return true;
}

void skip_ws(const char *&p, const char *end) {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
}

const char *match_enclosed(const char *p, const char *end, char open_ch, char close_ch) {
    if (p >= end || *p != open_ch) {
        return nullptr;
    }
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (const char *it = p; it < end; ++it) {
        char ch = *it;
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == open_ch) {
            ++depth;
        } else if (ch == close_ch) {
            --depth;
            if (depth == 0) {
                return it;
            }
        }
    }
    return nullptr;
}

bool find_key_value(const char *obj, size_t obj_len, const char *key,
                    const char *&value_ptr, size_t &value_len) {
    std::string pattern = std::string("\"") + key + "\"";
    std::string object(obj, obj_len);
    size_t key_pos = object.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon = object.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return false;
    }
    size_t start = colon + 1;
    while (start < obj_len && std::isspace(static_cast<unsigned char>(object[start]))) {
        ++start;
    }
    if (start >= obj_len) {
        return false;
    }

    if (object[start] == '{') {
        const char *begin = obj + start;
        const char *close = match_enclosed(begin, obj + obj_len, '{', '}');
        if (!close) {
            return false;
        }
        value_ptr = begin;
        value_len = static_cast<size_t>(close - begin + 1);
        return true;
    }
    if (object[start] == '[') {
        const char *begin = obj + start;
        const char *close = match_enclosed(begin, obj + obj_len, '[', ']');
        if (!close) {
            return false;
        }
        value_ptr = begin;
        value_len = static_cast<size_t>(close - begin + 1);
        return true;
    }
    if (object[start] == '"') {
        size_t pos = start + 1;
        bool escape = false;
        while (pos < obj_len) {
            char ch = object[pos];
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                value_ptr = obj + start;
                value_len = pos - start + 1;
                return true;
            }
            ++pos;
        }
        return false;
    }

    size_t pos = start;
    while (pos < obj_len && object[pos] != ',' && object[pos] != '}') {
        ++pos;
    }
    size_t finish = pos;
    while (finish > start && std::isspace(static_cast<unsigned char>(object[finish - 1]))) {
        --finish;
    }
    value_ptr = obj + start;
    value_len = finish - start;
    return true;
}

bool parse_int_key(const char *obj, size_t obj_len, const char *key, int *out) {
    const char *value_ptr = nullptr;
    size_t value_len = 0;
    if (!find_key_value(obj, obj_len, key, value_ptr, value_len)) {
        fprintf(stderr, "aligner_model_load: missing key %s\n", key);
        return false;
    }
    std::string tmp(value_ptr, value_len);
    char *end_ptr = nullptr;
    long parsed = std::strtol(tmp.c_str(), &end_ptr, 10);
    if (end_ptr == tmp.c_str()) {
        fprintf(stderr, "aligner_model_load: invalid integer for key %s\n", key);
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

bool parse_float_key(const char *obj, size_t obj_len, const char *key, float *out) {
    const char *value_ptr = nullptr;
    size_t value_len = 0;
    if (!find_key_value(obj, obj_len, key, value_ptr, value_len)) {
        fprintf(stderr, "aligner_model_load: missing key %s\n", key);
        return false;
    }
    std::string tmp(value_ptr, value_len);
    char *end_ptr = nullptr;
    double parsed = std::strtod(tmp.c_str(), &end_ptr);
    if (end_ptr == tmp.c_str()) {
        fprintf(stderr, "aligner_model_load: invalid float for key %s\n", key);
        return false;
    }
    *out = static_cast<float>(parsed);
    return true;
}

bool extract_object(const char *obj, size_t obj_len, const char *key,
                    const char *&subobject_ptr, size_t &subobject_len) {
    if (!find_key_value(obj, obj_len, key, subobject_ptr, subobject_len) || subobject_len == 0 ||
        subobject_ptr[0] != '{') {
        fprintf(stderr, "aligner_model_load: missing object %s\n", key);
        return false;
    }
    return true;
}

}  // namespace

int aligner_model_load(AlignerModel *m, const char *model_dir) {
    if (!m || !model_dir) {
        fprintf(stderr, "aligner_model_load: invalid argument\n");
        return -1;
    }

    std::memset(m, 0, sizeof(*m));

    char config_path[4096];
    char weights_path[4096];
    std::snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
    std::snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    std::string config;
    if (!read_text_file(config_path, config)) {
        return -1;
    }

    const char *root_ptr = config.data();
    size_t root_len = config.size();
    const char *thinker_cfg_ptr = nullptr;
    size_t thinker_cfg_len = 0;
    const char *audio_cfg_ptr = nullptr;
    size_t audio_cfg_len = 0;
    const char *text_cfg_ptr = nullptr;
    size_t text_cfg_len = 0;

    if (!parse_int_key(root_ptr, root_len, "timestamp_token_id", &m->cfg.timestamp_id) ||
        !parse_int_key(root_ptr, root_len, "timestamp_segment_time", &m->cfg.timestamp_segment_ms) ||
        !extract_object(root_ptr, root_len, "thinker_config", thinker_cfg_ptr, thinker_cfg_len) ||
        !extract_object(thinker_cfg_ptr, thinker_cfg_len, "audio_config", audio_cfg_ptr, audio_cfg_len) ||
        !extract_object(thinker_cfg_ptr, thinker_cfg_len, "text_config", text_cfg_ptr, text_cfg_len) ||
        !parse_int_key(thinker_cfg_ptr, thinker_cfg_len, "classify_num", &m->cfg.classify_num) ||
        !parse_int_key(thinker_cfg_ptr, thinker_cfg_len, "audio_start_token_id", &m->cfg.audio_start_id) ||
        !parse_int_key(thinker_cfg_ptr, thinker_cfg_len, "audio_end_token_id", &m->cfg.audio_end_id) ||
        !parse_int_key(thinker_cfg_ptr, thinker_cfg_len, "audio_token_id", &m->cfg.audio_pad_id) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "d_model", &m->cfg.enc_d_model) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "encoder_layers", &m->cfg.enc_layers) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "encoder_attention_heads", &m->cfg.enc_heads) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "encoder_ffn_dim", &m->cfg.enc_ffn_dim) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "num_mel_bins", &m->cfg.num_mel_bins) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "max_source_positions", &m->cfg.max_source_positions) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "n_window", &m->cfg.n_window) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "n_window_infer", &m->cfg.n_window_infer) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "downsample_hidden_size", &m->cfg.downsample_hidden) ||
        !parse_int_key(audio_cfg_ptr, audio_cfg_len, "output_dim", &m->cfg.enc_output_dim) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "hidden_size", &m->cfg.dec_hidden) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "num_hidden_layers", &m->cfg.dec_layers) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "num_attention_heads", &m->cfg.dec_heads) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "num_key_value_heads", &m->cfg.dec_kv_heads) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "head_dim", &m->cfg.dec_head_dim) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "intermediate_size", &m->cfg.dec_ffn) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "vocab_size", &m->cfg.dec_vocab) ||
        !parse_int_key(text_cfg_ptr, text_cfg_len, "max_position_embeddings", &m->cfg.dec_max_pos) ||
        !parse_float_key(text_cfg_ptr, text_cfg_len, "rms_norm_eps", &m->cfg.dec_rms_eps) ||
        !parse_float_key(text_cfg_ptr, text_cfg_len, "rope_theta", &m->cfg.dec_rope_theta)) {
        aligner_model_free(m);
        return -1;
    }

    {
        const char *env_infer = getenv("FA_WINDOW_INFER_OVERRIDE");
        if (env_infer && env_infer[0]) {
            int override_val = atoi(env_infer);
            if (override_val > 0) {
                fprintf(stderr, "[model] Overriding n_window_infer: %d -> %d\n", m->cfg.n_window_infer, override_val);
                m->cfg.n_window_infer = override_val;
            }
        }
    }

    if (m->cfg.enc_heads <= 0 || (m->cfg.enc_d_model % m->cfg.enc_heads) != 0) {
        fprintf(stderr, "aligner_model_load: invalid encoder head config (%d / %d)\n",
                m->cfg.enc_d_model, m->cfg.enc_heads);
        aligner_model_free(m);
        return -1;
    }
    m->cfg.enc_head_dim = m->cfg.enc_d_model / m->cfg.enc_heads;

    fprintf(stderr, "[model] Config: rope_theta=%.0f enc_layers=%d dec_layers=%d dec_heads=%d/%d dec_hidden=%d\n",
            m->cfg.dec_rope_theta, m->cfg.enc_layers, m->cfg.dec_layers,
            m->cfg.dec_heads, m->cfg.dec_kv_heads, m->cfg.dec_hidden);

    if (safetensors_load(weights_path, &m->ws) != 0) {
        aligner_model_free(m);
        return -1;
    }

    Tensor *embed = ws_get(&m->ws, "thinker.model.embed_tokens.weight");
    Tensor *lm_head = ws_get(&m->ws, "thinker.lm_head.weight");
    if (!embed || !lm_head) {
        fprintf(stderr, "aligner_model_load: missing required weights in %s\n", weights_path);
        aligner_model_free(m);
        return -1;
    }
    m->embed_weight = *embed;
    m->lm_head_weight = *lm_head;

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    /* Allocate cuBLAS workspace for bf16 GEMMs on Blackwell (sm_120).
     * cublasSetStream() resets the workspace, so we register it globally
     * and restore after every cublasSetStream call. */
    {
        size_t ws_size = 256 << 20;
        void *ws_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ws_ptr, ws_size));
        CUBLAS_CHECK(cublasSetWorkspace(handle, ws_ptr, ws_size));
        cublas_set_workspace(ws_ptr, ws_size);
    }
    m->cublas = handle;

    m->scratch_elems = static_cast<size_t>(m->cfg.dec_ffn) * static_cast<size_t>(m->cfg.dec_hidden);
    if (m->scratch_elems > 0) {
        m->scratch.dtype = DTYPE_BF16;
        m->scratch.ndim = 2;
        m->scratch.shape[0] = m->cfg.dec_ffn;
        m->scratch.shape[1] = m->cfg.dec_hidden;
        m->scratch.nbytes = m->scratch_elems * sizeof(bf16_t);
        CUDA_CHECK(cudaMalloc(&m->scratch.data, m->scratch.nbytes));
    }

    return 0;
}

void aligner_model_free(AlignerModel *m) {
    if (!m) {
        return;
    }

    if (m->scratch.data) {
        CUDA_CHECK(cudaFree(m->scratch.data));
    }
    std::memset(&m->scratch, 0, sizeof(m->scratch));
    m->scratch_elems = 0;

    if (m->cublas) {
        cublasDestroy(static_cast<cublasHandle_t>(m->cublas));
        m->cublas = nullptr;
    }

    m->embed_weight = Tensor{};
    m->lm_head_weight = Tensor{};
    safetensors_free(&m->ws);
    std::memset(&m->cfg, 0, sizeof(m->cfg));
}
