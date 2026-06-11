#ifndef GLMASR_TYPES_H
#define GLMASR_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* ── Dtype tags ───────────────────────────────────────── */
#define DTYPE_FP32   0
#define DTYPE_BF16   1
#define DTYPE_FP16   2
#define DTYPE_UINT8  3   /* packed FP4 (2 values per byte) */
#define DTYPE_FP8    4   /* E4M3 block scales              */

static inline size_t dtype_size(int dtype) {
    switch (dtype) {
        case DTYPE_FP32: return 4;
        case DTYPE_BF16: return 2;
        case DTYPE_FP16: return 2;
        case DTYPE_UINT8: return 1;
        case DTYPE_FP8:  return 1;
        default: return 0;
    }
}

/* ── Tensor ───────────────────────────────────────────── */
#define MAX_DIMS 4

typedef struct {
    void    *data;          /* device (GPU) memory         */
    int      dtype;
    int      ndim;
    int64_t  shape[MAX_DIMS];
    size_t   nbytes;
} Tensor;

/* Total element count */
static inline int64_t tensor_numel(const Tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* ── Model Config (from config.json) ─────────────────── */
typedef struct {
    /* Audio Encoder */
    int enc_hidden;         /* 1280 */
    int enc_layers;         /* 32   */
    int enc_heads;          /* 20   */
    int enc_head_dim;       /* 64   */
    int enc_ffn;            /* 5120 */
    int num_mel_bins;       /* 128  */
    int max_audio_frames;   /* 3000 (30 s) */

    /* Projector */
    int proj_mid;           /* 4096 */

    /* Decoder (Llama) */
    int dec_hidden;         /* 2048  */
    int dec_layers;         /* 28    */
    int dec_heads;          /* 16    */
    int dec_kv_heads;       /* 4     */
    int dec_head_dim;       /* 128   */
    int dec_ffn;           /* 6144  */
    int vocab_size;         /* 59264 */
    int max_position_embeddings; /* 8192 */

    /* Special tokens */
    int audio_token_id;     /* 59260 */
    int eos_ids[4];         /* 59246, 59253, 59255, -1 sentinel */
    int pad_token_id;       /* 59260 (<|pad|>) */

    /* Generation */
    int max_new_tokens;     /* 500 */
} ModelConfig;

/* ── KV Cache (per decoder layer) ────────────────────── */
typedef struct {
    Tensor k;   /* [batch, kv_heads, max_seq, head_dim] bf16 */
    Tensor v;   /* same */
} LayerKV;

typedef struct {
    LayerKV *layers;    /* array[dec_layers] */
    int      seq_len;   /* current cached length */
    int      max_seq;   /* allocated capacity */
} KVCache;

/* ── Weight Store (name → Tensor lookup) ─────────────── */
#define MAX_WEIGHTS 2048

typedef struct {
    char    *names[MAX_WEIGHTS];
    Tensor   tensors[MAX_WEIGHTS];
    int      count;
} WeightStore;

typedef struct {
    uint64_t alloc_calls;
    uint64_t free_calls;
    uint64_t alloc_bytes;
    uint64_t free_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
} ScratchProfileStats;

typedef struct {
    uint64_t fa_calls;
    uint64_t fa_simple_calls;
    uint64_t fa_pipeline_calls;
    uint64_t fa_rows;
    uint64_t fa_score_elems;
    uint64_t cublas_calls;
    uint64_t cublas_rows;
    uint64_t cublas_score_elems;
    uint64_t compare_calls;
} EncoderAttentionProfileStats;

typedef struct {
    uint64_t qkv_ns_total;
    uint64_t qkv_ns_max;
    uint64_t cache_ns_total;
    uint64_t cache_ns_max;
    uint64_t attn_ns_total;
    uint64_t attn_ns_max;
    uint64_t o_proj_ns_total;
    uint64_t o_proj_ns_max;
    uint64_t mlp_ns_total;
    uint64_t mlp_ns_max;
} DecodeKernelProfileStats;

typedef struct {
    uint64_t qkv_ns_total;
    uint64_t qkv_ns_max;
    uint64_t cache_ns_total;
    uint64_t cache_ns_max;
    uint64_t attn_ns_total;
    uint64_t attn_ns_max;
    uint64_t o_proj_ns_total;
    uint64_t o_proj_ns_max;
    uint64_t mlp_ns_total;
    uint64_t mlp_ns_max;
} PrefillKernelProfileStats;

/* Lookup weight by name. Returns NULL if not found. */
static inline Tensor *ws_get(WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++) {
        /* fast reject on first char */
        if (ws->names[i][0] == name[0]) {
            int j = 0;
            while (ws->names[i][j] && ws->names[i][j] == name[j]) j++;
            if (ws->names[i][j] == 0 && name[j] == 0) return &ws->tensors[i];
        }
    }
    return NULL;
}

/* Lookup weight by name (const version). Returns NULL if not found. */
static inline const Tensor *ws_get_const(const WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++) {
        /* fast reject on first char */
        if (ws->names[i][0] == name[0]) {
            int j = 0;
            while (ws->names[i][j] && ws->names[i][j] == name[j]) j++;
            if (ws->names[i][j] == 0 && name[j] == 0) return &ws->tensors[i];
        }
    }
    return NULL;
}

/* ── CUDA/C compatibility ────────────────────────────── */
#ifdef __CUDACC__
#include <cuda_bf16.h>
#include <cuda_fp16.h>
typedef __nv_bfloat16 bf16_t;
typedef __half        fp16_t;
#else
typedef uint16_t bf16_t;
typedef uint16_t fp16_t;
#endif

/* ── Error handling ──────────────────────────────────── */
#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "ERROR [%s:%d]: " fmt "\n", \
                __FILE__, __LINE__, ##__VA_ARGS__); \
        exit(1); \
    } \
} while(0)

#ifdef __CUDACC__
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)
#endif

#endif /* GLMASR_TYPES_H */
