#ifndef FA_TYPES_H
#define FA_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* ── Dtype tags ───────────────────────────────────────── */
#define DTYPE_FP32   0
#define DTYPE_BF16   1
#define DTYPE_FP16   2
#define DTYPE_INT32  3

static inline size_t dtype_size(int dtype) {
    switch (dtype) {
        case DTYPE_FP32: return 4;
        case DTYPE_BF16: return 2;
        case DTYPE_FP16: return 2;
        case DTYPE_INT32: return 4;
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

static inline int64_t tensor_numel(const Tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/* ── Aligner Model Config (from config.json) ─────────── */
typedef struct {
    /* Audio Encoder (Whisper-style) */
    int enc_d_model;              /* 1024 */
    int enc_layers;               /* 24   */
    int enc_heads;                /* 16   */
    int enc_ffn_dim;              /* 4096 */
    int enc_head_dim;             /* d_model / heads = 64 */
    int num_mel_bins;             /* 128  */
    int max_source_positions;     /* 1500 */
    int n_window;                 /* 50/100, chunk size = n_window * 2 */
    int n_window_infer;           /* 800, for chunked infer attention */
    int downsample_hidden;        /* 480  */
    int enc_output_dim;           /* 1024 */

    /* Text Decoder (Qwen-style) */
    int dec_hidden;               /* 1024 */
    int dec_layers;               /* 28   */
    int dec_heads;                /* 16   */
    int dec_kv_heads;             /* 8    */
    int dec_head_dim;             /* 128  */
    int dec_ffn;                  /* 3072 */
    int dec_vocab;                /* 152064 */
    int dec_max_pos;              /* 8192 */
    float dec_rms_eps;            /* 1e-6 */
    float dec_rope_theta;         /* 1000000.0 */

    /* Timestamp classifier */
    int classify_num;             /* 5000 */
    int timestamp_segment_ms;     /* 80 */

    /* Special token IDs */
    int audio_start_id;           /* 151669 */
    int audio_end_id;             /* 151670 */
    int audio_pad_id;             /* 151676 */
    int timestamp_id;             /* 151705 */
} AlignerConfig;

/* ── Weight Store (name -> Tensor lookup) ─────────────── */
#define MAX_WEIGHTS 2048

typedef struct {
    char    *names[MAX_WEIGHTS];
    Tensor   tensors[MAX_WEIGHTS];
    int      count;
} WeightStore;

static inline Tensor *ws_get(WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++) {
        if (ws->names[i][0] == name[0]) {
            int j = 0;
            while (ws->names[i][j] && ws->names[i][j] == name[j]) j++;
            if (ws->names[i][j] == 0 && name[j] == 0) return &ws->tensors[i];
        }
    }
    return NULL;
}

static inline const Tensor *ws_get_const(const WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++) {
        if (ws->names[i][0] == name[0]) {
            int j = 0;
            while (ws->names[i][j] && ws->names[i][j] == name[j]) j++;
            if (ws->names[i][j] == 0 && name[j] == 0) return &ws->tensors[i];
        }
    }
    return NULL;
}

/* ── Aligned word result ──────────────────────────────── */
typedef struct {
    char     text[256];
    double   start_time;   /* seconds */
    double   end_time;     /* seconds */
} AlignWord;

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
#include <stdio.h>
#include <stdlib.h>

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

#endif /* FA_TYPES_H */
