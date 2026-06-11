#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "model.h"
#include "safetensors.h"
#include "types.h"
#include "cuda_kernels.h"

/* Minimal JSON helpers (same style as safetensors.c) */
static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int64_t parse_json_int(const char *p, const char **end_out) {
    int64_t val = 0;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *end_out = p;
    return neg ? -val : val;
}

static int env_int_bounded(const char *name, int default_value, int min_value, int max_value) {
    const char *env = getenv(name);
    long parsed;
    char *endptr = NULL;

    if (env == NULL || env[0] == '\0') {
        return default_value;
    }

    parsed = strtol(env, &endptr, 10);
    if (endptr == env || *endptr != '\0') {
        return default_value;
    }
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return (int)parsed;
}

/* Parse nested object key:value */
static int64_t parse_nested_int(const char *p, const char *key, const char **end_out) {
    int depth = 0;
    while (*p) {
        if (*p == '{') { depth++; p++; continue; }
        if (*p == '}') { if (depth <= 1) break; depth--; p++; continue; }
        if (depth == 1 && *p == '"') {
            p++;
            int key_len = strlen(key);
            if (strncmp(p, key, key_len) == 0 && p[key_len] == '"') {
                p += key_len + 1;
                while (*p == ' ' || *p == ':') p++;
                return parse_json_int(p, end_out);
            }
            while (*p && *p != '"') p++;
            if (*p == '"') p++;
            continue;
        }
        p++;
    }
    *end_out = p;
    return 0;
}

static int model_decoder_needs_scratch(const WeightStore *ws)
{
    const Tensor *probe = ws_get_const(ws, "language_model.model.layers.0.self_attn.q_proj.weight");
    if (probe == NULL) {
        return 1;
    }
    return probe->dtype == DTYPE_UINT8 || probe->dtype == DTYPE_BF16;
}

void model_config_from_json(const char *path, ModelConfig *cfg) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        exit(1);
    }

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(len + 1);
    fread(json, 1, len, f);
    json[len] = '\0';
    fclose(f);

    const char *p = json;

    /* Extract top-level values */
    p = skip_ws(p);

    const char *ati = strstr(json, "\"audio_token_id\"");
    if (ati) {
        ati += 16;
        ati = skip_ws(ati);
        if (*ati == ':') ati++;
        ati = skip_ws(ati);
        cfg->audio_token_id = (int)parse_json_int(ati, &ati);
    }
    const char *vs = strstr(json, "\"vocab_size\"");
    if (vs) {
        vs += 12;
        vs = skip_ws(vs);
        if (*vs == ':') vs++;
        vs = skip_ws(vs);
        cfg->vocab_size = (int)parse_json_int(vs, &vs);
    }

    /* Parse audio_config */
    const char *ac = strstr(json, "\"audio_config\"");
    if (ac) {
        const char *sec = strchr(ac, '{');
        const char *tmp;
        if (sec) {
            cfg->enc_hidden = (int)parse_nested_int(sec, "hidden_size", &tmp);
            cfg->enc_layers = (int)parse_nested_int(sec, "num_hidden_layers", &tmp);
            cfg->enc_heads = (int)parse_nested_int(sec, "num_attention_heads", &tmp);
            cfg->enc_head_dim = (int)parse_nested_int(sec, "head_dim", &tmp);
            cfg->enc_ffn = (int)parse_nested_int(sec, "intermediate_size", &tmp);
            cfg->num_mel_bins = (int)parse_nested_int(sec, "num_mel_bins", &tmp);
            cfg->max_audio_frames = (int)parse_nested_int(sec, "max_position_embeddings", &tmp);
        }
    }

    /* Parse text_config */
    const char *tc = strstr(json, "\"text_config\"");
    if (tc) {
        const char *sec = strchr(tc, '{');
        const char *tmp;
        if (sec) {
            cfg->dec_hidden = (int)parse_nested_int(sec, "hidden_size", &tmp);
            cfg->dec_layers = (int)parse_nested_int(sec, "num_hidden_layers", &tmp);
            cfg->dec_heads = (int)parse_nested_int(sec, "num_attention_heads", &tmp);
            cfg->dec_kv_heads = (int)parse_nested_int(sec, "num_key_value_heads", &tmp);
            cfg->dec_head_dim = (int)parse_nested_int(sec, "head_dim", &tmp);
            cfg->dec_ffn = (int)parse_nested_int(sec, "intermediate_size", &tmp);
            cfg->vocab_size = (int)parse_nested_int(sec, "vocab_size", &tmp);
        }
    }

    /* eos_token_id is an array in text_config */
    p = strstr(json, "\"eos_token_id\"");
    if (p) {
        p = strchr(p, '[');
        if (p) {
            p++;
            p = skip_ws(p);
            cfg->eos_ids[0] = (int)parse_json_int(p, &p);
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
            cfg->eos_ids[1] = (int)parse_json_int(p, &p);
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
            cfg->eos_ids[2] = (int)parse_json_int(p, &p);
            cfg->eos_ids[3] = -1;
        }
    }

    /* Set defaults not in config */
    cfg->proj_mid = 4096;
    cfg->max_audio_frames = 3000;
    cfg->pad_token_id = cfg->audio_token_id;
    cfg->max_new_tokens = env_int_bounded("GLMASR_MAX_NEW_TOKENS", 500, 1, 8192);

    free(json);

}

int model_load(Model *m, const char *model_dir) {
    char path[512];

    /* Build config.json path */
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    model_config_from_json(path, &m->cfg);

    /* Build model.safetensors path */
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (safetensors_load(path, &m->ws) != 0) {
        fprintf(stderr, "ERROR: Failed to load safetensors\n");
        return -1;
    }

    /* Create cuBLAS handle */
    cublasHandle_t handle;
    cublasCreate(&handle);
    m->cublas = (void*)handle;

    /* Decoder scratch: needed for FP4 dequant AND for BF16 weight
     * fusion in linear_proj_pair_same_input (K/V and gate/up). */
    m->dequant_buf.data = NULL;
    m->dequant_buf.dtype = DTYPE_BF16;
    m->dequant_buf.ndim = 2;
    m->dequant_buf.shape[0] = 0;
    m->dequant_buf.shape[1] = 1;
    m->dequant_buf.nbytes = 0;
    m->dequant_weight_scratch_elems = 0;
    m->dequant_aux_scratch_elems = 0;

    if (model_decoder_needs_scratch(&m->ws)) {
        m->dequant_weight_scratch_elems = (size_t)m->cfg.dec_ffn * (size_t)m->cfg.dec_hidden;
        m->dequant_aux_scratch_elems = m->dequant_weight_scratch_elems;
        size_t dequant_size = (m->dequant_weight_scratch_elems +
                               m->dequant_aux_scratch_elems) * sizeof(bf16_t);
        cudaMalloc(&m->dequant_buf.data, dequant_size);
        m->dequant_buf.shape[0] = (int64_t)(m->dequant_weight_scratch_elems +
                                            m->dequant_aux_scratch_elems);
        m->dequant_buf.nbytes = dequant_size;
    }

    /* Allocate KV cache: max 1024 tokens (legacy contiguous) */
    kv_cache_alloc(&m->kv, &m->cfg, 1024);

    /* Allocate paged KV pool (Phase 1: coexist with legacy, used by new decode path) */
    m->kv_pool = calloc(1, sizeof(KVPool));
    int num_blocks = kv_pool_warmup(KV_DEFAULT_BLOCK_SIZE,
                                     m->cfg.dec_layers,
                                     m->cfg.dec_kv_heads,
                                     m->cfg.dec_head_dim);
    if (num_blocks < 4) num_blocks = 4;  /* minimum for a single short request */
    m->kv_pool->max_seqs = 1;
    m->kv_pool->max_blocks_per_seq = num_blocks;
    if (kv_pool_init(m->kv_pool, num_blocks, KV_DEFAULT_BLOCK_SIZE,
                      m->cfg.dec_layers, m->cfg.dec_kv_heads,
                      m->cfg.dec_head_dim) != 0) {
        fprintf(stderr, "WARNING: kv_pool_init failed, using legacy KV only\n");
        free(m->kv_pool);
        m->kv_pool = NULL;
    }

    return 0;
}

void model_free(Model *m) {
    /* Free weight store */
    safetensors_free(&m->ws);

    /* Destroy cuBLAS handle */
    if (m->cublas) {
        cublasDestroy((cublasHandle_t)m->cublas);
        m->cublas = NULL;
    }

    /* Free dequant buffer */
    if (m->dequant_buf.data) {
        cudaFree(m->dequant_buf.data);
        m->dequant_buf.data = NULL;
    }
    m->dequant_weight_scratch_elems = 0;
    m->dequant_aux_scratch_elems = 0;

    /* Free KV cache (legacy) */
    kv_cache_free(&m->kv);

    /* Free paged KV pool */
    if (m->kv_pool) {
        kv_pool_free(m->kv_pool);
        free(m->kv_pool);
        m->kv_pool = NULL;
    }
}
