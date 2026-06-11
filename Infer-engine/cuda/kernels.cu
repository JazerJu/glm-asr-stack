/**
 * GLM-ASR CUDA Kernels Implementation
 * Target: RTX 5070 Ti (SM120, Blackwell), CUDA 12.8
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cufft.h>
#include <cub/cub.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/types.h"

extern "C" {

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

static cublasLtHandle_t g_bf16_lt_handle = NULL;
static void *g_bf16_lt_workspace = NULL;
static size_t g_bf16_lt_workspace_size = 32 * 1024 * 1024;

static int env_int_bounded(const char *name, int default_value, int min_value, int max_value) {
    const char *env = getenv(name);
    char *endptr = NULL;
    long parsed;

    if (env == NULL || env[0] == '\0') {
        return default_value;
    }
    parsed = strtol(env, &endptr, 10);
    if (endptr == env) {
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

static cublasGemmAlgo_t bf16_linear_gemm_algo(void) {
    static int cached = -1;

    if (cached < 0) {
        cached = env_int_bounded("GLMASR_BF16_LINEAR_TENSOR_OP", 0, 0, 1);
    }
    return cached ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
}

static int bf16_linear_lt_bias_enabled(void) {
    static int cached = -1;

    if (cached < 0) {
        cached = env_int_bounded("GLMASR_BF16_LINEAR_LT_BIAS", 1, 0, 1);
    }
    return cached;
}

static int ensure_bf16_lt_resources(void) {
    cudaError_t err;

    if (g_bf16_lt_handle == NULL) {
        if (cublasLtCreate(&g_bf16_lt_handle) != CUBLAS_STATUS_SUCCESS) {
            return 0;
        }
    }
    if (g_bf16_lt_workspace == NULL) {
        err = cudaMalloc(&g_bf16_lt_workspace, g_bf16_lt_workspace_size);
        if (err != cudaSuccess) {
            return 0;
        }
    }
    return 1;
}

// FP4 E2M1 lookup table: 16 values
__constant__ float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// FP8 E4M3 to float conversion (no NaN, all patterns are numbers)
// FP8 E4M3: sign(1) + exp(4) + mantissa(3), bias=7
__device__ __forceinline__ float fp8_to_float_e4m3(uint8_t x) {
    int sign = (x >> 7) & 1;
    int exp = (x >> 3) & 0x0F;
    int mant = x & 0x07;
    
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        /* denormal: (-1)^sign * 2^(1-bias) * (mant/8) = mant/8 * 2^-6 */
        float v = (mant / 8.0f) * 0.015625f;  /* 0.015625 = 2^-6 */
        return sign ? -v : v;
    }
    
    // FP8 E4M3: value = (1 + mant/8) * 2^(exp - 7)
    float v = (8.0f + mant) / 8.0f;
    v *= powf(2.0f, exp - 7);
    
    return sign ? -v : v;
}

static __nv_bfloat16 *d_lm_head_logits = NULL;
static float *d_lm_head_logits_f32 = NULL;
static int *d_lm_head_tokens = NULL;
static size_t lm_head_logits_capacity = 0;
static size_t lm_head_logits_f32_capacity = 0;
static int lm_head_token_capacity = 0;

static void ensure_lm_head_buffers(int rows, int vocab_size) {
    size_t needed_logits;

    if (rows <= 0 || vocab_size <= 0) {
        return;
    }

    needed_logits = (size_t) rows * (size_t) vocab_size;
    if (needed_logits > lm_head_logits_capacity) {
        if (d_lm_head_logits != NULL) {
            CUDA_CHECK(cudaFree(d_lm_head_logits));
            d_lm_head_logits = NULL;
        }
        CUDA_CHECK(cudaMalloc(&d_lm_head_logits, needed_logits * sizeof(*d_lm_head_logits)));
        lm_head_logits_capacity = needed_logits;
    }

    if (rows > lm_head_token_capacity) {
        if (d_lm_head_tokens != NULL) {
            CUDA_CHECK(cudaFree(d_lm_head_tokens));
            d_lm_head_tokens = NULL;
        }
        CUDA_CHECK(cudaMalloc(&d_lm_head_tokens, (size_t) rows * sizeof(*d_lm_head_tokens)));
        lm_head_token_capacity = rows;
    }
}

static void ensure_lm_head_single_buffers(int vocab_size) {
    size_t needed_logits;

    if (vocab_size <= 0) {
        return;
    }

    needed_logits = (size_t) vocab_size;
    if (needed_logits > lm_head_logits_f32_capacity) {
        if (d_lm_head_logits_f32 != NULL) {
            CUDA_CHECK(cudaFree(d_lm_head_logits_f32));
            d_lm_head_logits_f32 = NULL;
        }
        CUDA_CHECK(cudaMalloc(&d_lm_head_logits_f32, needed_logits * sizeof(*d_lm_head_logits_f32)));
        lm_head_logits_f32_capacity = needed_logits;
    }

    if (lm_head_token_capacity < 1) {
        if (d_lm_head_tokens != NULL) {
            CUDA_CHECK(cudaFree(d_lm_head_tokens));
            d_lm_head_tokens = NULL;
        }
        CUDA_CHECK(cudaMalloc(&d_lm_head_tokens, sizeof(*d_lm_head_tokens)));
        lm_head_token_capacity = 1;
    }
}

/* ============================================================================
 * FP4 Dequantization: uint8 (2 FP4 values) → bf16
 * block_scales: FP8 E4M3, one per 16 elements along K dimension
 * global_scale: additional scale factor
 * ============================================================================ */
__global__ void fp4_dequant_kernel(
    __nv_bfloat16 *out,
    const uint8_t *packed_w,
    const uint8_t *block_scales,
    float global_scale,
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    
    if (idx >= total_elements) return;
    
    int row = idx / cols;
    int col = idx % cols;
    
    // Each block scale covers 16 elements along K dimension
    int block_idx = col / 16;
    int pack_pos = col % 16;  // position within the 16-element block
    int byte_idx = pack_pos / 2;  // which byte in the packed uint8
    int nibble = pack_pos % 2;  // 0 = low nibble, 1 = high nibble
    
    // Calculate packed weight byte position: [row, col/2]
    int packed_col = col / 2;
    uint8_t packed_byte = packed_w[row * ((cols + 1) / 2) + packed_col];
    
    // Extract FP4 value (0-15)
    uint8_t fp4_val;
    if (nibble == 0) {
        fp4_val = packed_byte & 0x0F;  // low nibble
    } else {
        fp4_val = (packed_byte >> 4) & 0x0F;  // high nibble
    }
    
    // Get FP8 scale for this block
    float scale = 1.0f;
    if (block_scales != NULL) {
        scale = fp8_to_float_e4m3(block_scales[row * ((cols + 15) / 16) + block_idx]);
    }
    
    // Compute final value: fp4 * fp8_scale * global_scale
    float fp4_val_float = fp4_lut[fp4_val];
    float final_val = fp4_val_float * scale * global_scale;
    
    out[idx] = __float2bfloat16(final_val);
}

void fp4_dequant_to_bf16(__nv_bfloat16 *out,
                         const uint8_t *packed_w,
                         const uint8_t *block_scales,
                         float global_scale,
                         int rows, int cols) {
    int total_elements = rows * cols;
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    
    fp4_dequant_kernel<<<gridSize, blockSize>>>(out, packed_w, block_scales, 
                                                 global_scale, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * RMSNorm: out = x / rms * weight, where rms = sqrt(mean(x^2) + eps)
 * One warp per row, use warp shuffle for reduction
 * ============================================================================ */
__global__ void rmsnorm_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    float eps, int rows, int dim
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const __nv_bfloat16 *x_row = x + row * dim;
    __nv_bfloat16 *out_row = out + row * dim;
    
    // Warp-level reduction using shuffle
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += WARP_SIZE) {
        float val = __bfloat162float(x_row[i]);
        sum_sq += val * val;
    }
    
    // Warp shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xFFFFFFFF, sum_sq, 0);
    
    // Compute rms
    float rms = sqrtf(sum_sq / dim + eps);
    
    // Apply normalization and weight
    for (int i = threadIdx.x; i < dim; i += WARP_SIZE) {
        float val = __bfloat162float(x_row[i]) / rms;
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(val * w);
    }
}

void rmsnorm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
             const __nv_bfloat16 *weight, float eps, int rows, int dim) {
    int blockSize = WARP_SIZE;  // 32 threads per warp
    int gridSize = rows;
    
    rmsnorm_kernel<<<gridSize, blockSize>>>(out, x, weight, eps, rows, dim);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
 * ============================================================================ */
__global__ void layernorm_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *weight,
    const __nv_bfloat16 *bias,
    float eps, int rows, int dim
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const __nv_bfloat16 *x_row = x + row * dim;
    __nv_bfloat16 *out_row = out + row * dim;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += WARP_SIZE) {
        sum += __bfloat162float(x_row[i]);
    }
    
    // Warp shuffle reduction for mean
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    
    float mean = sum / dim;
    
    // Compute variance: mean((x - mean)^2)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += WARP_SIZE) {
        float diff = __bfloat162float(x_row[i]) - mean;
        sum_sq += diff * diff;
    }
    
    // Warp shuffle reduction for variance
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xFFFFFFFF, sum_sq, 0);
    
    float inv_std = 1.0f / sqrtf(sum_sq / dim + eps);
    
    // Apply normalization with weight and bias
    for (int i = threadIdx.x; i < dim; i += WARP_SIZE) {
        float val = (__bfloat162float(x_row[i]) - mean) * inv_std;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        out_row[i] = __float2bfloat16(val * w + b);
    }
}

void layernorm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
               const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
               float eps, int rows, int dim) {
    int blockSize = WARP_SIZE;
    int gridSize = rows;
    
    layernorm_kernel<<<gridSize, blockSize>>>(out, x, weight, bias, eps, rows, dim);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 * ============================================================================ */
__global__ void gelu_kernel(__nv_bfloat16 *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float v = __bfloat162float(x[idx]);
    float tanh_arg = 0.79788456f * (v + 0.044715f * v * v * v);
    float gelu = 0.5f * v * (1.0f + tanhf(tanh_arg));
    x[idx] = __float2bfloat16(gelu);
}

void gelu_inplace(__nv_bfloat16 *x, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    gelu_kernel<<<gridSize, blockSize>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * SiLU activation: x * sigmoid(x)
 * ============================================================================ */
__global__ void silu_kernel(__nv_bfloat16 *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float v = __bfloat162float(x[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-v));
    x[idx] = __float2bfloat16(v * sigmoid);
}

void silu_inplace(__nv_bfloat16 *x, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    silu_kernel<<<gridSize, blockSize>>>(x, n);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * silu_elementwise_mul: gate = silu(gate) * up
 * ============================================================================ */
__global__ void silu_mul_kernel(__nv_bfloat16 *gate, const __nv_bfloat16 *up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sigmoid = 1.0f / (1.0f + expf(-g));
    gate[idx] = __float2bfloat16(g * sigmoid * u);
}

void silu_elementwise_mul(__nv_bfloat16 *gate, const __nv_bfloat16 *up, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    silu_mul_kernel<<<gridSize, blockSize>>>(gate, up, n);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * RoPE for encoder: partial_rotary_factor=0.5, head_dim=64, rotate first 32 dims
 * freq = 1.0 / (10000^(2i/d)) where d = head_dim * partial_factor
 * ============================================================================ */
__global__ void rope_encoder_kernel(
    __nv_bfloat16 *q, __nv_bfloat16 *k,
    int seq_len, int num_heads, int head_dim,
    float partial_factor
) {
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (head_idx >= num_heads || seq_idx >= seq_len || dim_idx >= head_dim) return;

    /* GLM-ASR encoder RoPE (partial rotary):
     * rot_dim = head_dim * partial_factor = 32
     * half    = rot_dim / 2               = 16  (number of (i, i+half) pairs)
     * inv_freq[i] = 1 / (10000 ^ (2i / rot_dim))  for i = 0..half-1
     * Rotates dims [0, half) paired with [half, rot_dim).
     * Dims [rot_dim, head_dim) are left unchanged.
     * Follows rotate_half convention: new[i] = x[i]*cos - x[i+half]*sin
     *                                 new[i+half] = x[i+half]*cos + x[i]*sin */
    int rot_dim = (int)(head_dim * partial_factor);  // 32
    int half    = rot_dim / 2;                        // 16

    /* Only the first `half` threads do work */
    if (dim_idx >= half) return;

    float inv_freq = 1.0f / powf(10000.0f, (2.0f * dim_idx) / (float)rot_dim);
    float theta    = seq_idx * inv_freq;
    float cos_v    = cosf(theta);
    float sin_v    = sinf(theta);

    /* Apply to Q */
    if (q != NULL) {
        __nv_bfloat16 *q_head = q + seq_idx * num_heads * head_dim + head_idx * head_dim;
        float q0 = __bfloat162float(q_head[dim_idx]);
        float q1 = __bfloat162float(q_head[dim_idx + half]);
        q_head[dim_idx]        = __float2bfloat16(q0 * cos_v - q1 * sin_v);
        q_head[dim_idx + half] = __float2bfloat16(q1 * cos_v + q0 * sin_v);
    }

    /* Apply to K */
    if (k != NULL) {
        __nv_bfloat16 *k_head = k + seq_idx * num_heads * head_dim + head_idx * head_dim;
        float k0 = __bfloat162float(k_head[dim_idx]);
        float k1 = __bfloat162float(k_head[dim_idx + half]);
        k_head[dim_idx]        = __float2bfloat16(k0 * cos_v - k1 * sin_v);
        k_head[dim_idx + half] = __float2bfloat16(k1 * cos_v + k0 * sin_v);
    }
}

void rope_encoder(__nv_bfloat16 *q, __nv_bfloat16 *k,
                 int seq_len, int num_heads, int head_dim,
                 float partial_factor) {
    dim3 block(head_dim);
    dim3 grid(num_heads, seq_len);
    
    rope_encoder_kernel<<<grid, block>>>(q, k, seq_len, num_heads, head_dim, partial_factor);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * RoPE for decoder: full rotation, head_dim=128
 * Applies to ONE buffer (either Q or K) at a time.
 * buf layout: [seq_len, n_heads, head_dim]
 * grid: (n_heads, seq_len), block: (head_dim/2)
 * ============================================================================ */
__global__ void rope_generic_kernel(
    __nv_bfloat16 *buf,
    int pos_offset, int seq_len,
    int n_heads, int head_dim
) {
    int head_idx = blockIdx.x;
    int seq_idx  = blockIdx.y;
    int dim_idx  = threadIdx.x;   /* 0 .. head_dim/2 - 1 */

    if (head_idx >= n_heads || seq_idx >= seq_len) return;

    int half_dim = head_dim / 2;
    if (dim_idx >= half_dim) return;

    float inv_freq = 1.0f / powf(10000.0f, (2.0f * dim_idx) / head_dim);
    float theta = (seq_idx + pos_offset) * inv_freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    __nv_bfloat16 *head_ptr = buf + seq_idx * n_heads * head_dim + head_idx * head_dim;
    float v0 = __bfloat162float(head_ptr[dim_idx]);
    float v1 = __bfloat162float(head_ptr[dim_idx + half_dim]);

    head_ptr[dim_idx]          = __float2bfloat16(v0 * cos_t - v1 * sin_t);
    head_ptr[dim_idx + half_dim] = __float2bfloat16(v0 * sin_t + v1 * cos_t);
}

 void rope_decoder(__nv_bfloat16 *q, __nv_bfloat16 *k,
                  int pos_offset, int seq_len,
                  int q_heads, int kv_heads, int head_dim) {
     int half_dim = head_dim / 2;
     if (q != NULL) {
         dim3 block(half_dim);
         dim3 grid(q_heads, seq_len);
         rope_generic_kernel<<<grid, block>>>(q, pos_offset, seq_len, q_heads, head_dim);
         CUDA_CHECK(cudaGetLastError());
     }
     if (k != NULL) {
         dim3 block(half_dim);
         dim3 grid(kv_heads, seq_len);
         rope_generic_kernel<<<grid, block>>>(k, pos_offset, seq_len, kv_heads, head_dim);
         CUDA_CHECK(cudaGetLastError());
     }
 }

/* Batched varlen decoder RoPE — single kernel launch for all sequences.
 * Replaces per-sequence rope_decoder loop in prefill path.
 * Uses pre-computed cos/sin table (pos_offset=0, decoder head_dim=128).
 * Layout: buf[token_offset * n_heads * head_dim + head * head_dim + dim]
 * cu_seqlens: [num_seqs+1], cu_seqlens[0]=0, cu_seqlens[s]=sum(seq_lens[0..s-1])
 */
static float *d_dec_rope_cos = NULL;
static float *d_dec_rope_sin = NULL;
static int    d_dec_rope_cap = 0;

#define DEC_ROPE_HALF_DIM 64  /* head_dim(128) / 2 */

static void dec_rope_ensure_table(int max_pos) {
    if (max_pos <= d_dec_rope_cap) return;
    if (d_dec_rope_cos) cudaFree(d_dec_rope_cos);
    if (d_dec_rope_sin) cudaFree(d_dec_rope_sin);
    int size = max_pos * DEC_ROPE_HALF_DIM;
    CUDA_CHECK(cudaMalloc(&d_dec_rope_cos, (size_t)size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dec_rope_sin, (size_t)size * sizeof(float)));
    float *h_cos = (float *)malloc((size_t)size * sizeof(float));
    float *h_sin = (float *)malloc((size_t)size * sizeof(float));
    for (int pos = 0; pos < max_pos; pos++) {
        for (int d = 0; d < DEC_ROPE_HALF_DIM; d++) {
            float inv_freq = 1.0f / powf(10000.0f, (2.0f * d) / 128.0f);
            float theta = pos * inv_freq;
            h_cos[pos * DEC_ROPE_HALF_DIM + d] = cosf(theta);
            h_sin[pos * DEC_ROPE_HALF_DIM + d] = sinf(theta);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_dec_rope_cos, h_cos, (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dec_rope_sin, h_sin, (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_cos); free(h_sin);
    d_dec_rope_cap = max_pos;
}

/* Batched varlen decoder RoPE — 3D grid, no reverse-map.
 * Grid: (n_heads, max_seq_len, num_seqs) — each thread checks if it's
 * within its sequence's actual length (via cu_seqlens).
 * Much simpler than flat-idx reverse-map, and the idle threads for
 * short sequences are negligible vs 14K kernel launches saved. */
__global__ void __launch_bounds__(128)
rope_decoder_batched_kernel(__nv_bfloat16 *buf,
                            const float * __restrict__ cos_table,
                            const float * __restrict__ sin_table,
                            const int * __restrict__ cu_seqlens,
                            int num_seqs,
                            int n_heads,
                            int head_dim,
                            int half_dim) {
    int d       = threadIdx.x;                  /* 0 .. half_dim-1 */
    int head    = blockIdx.x;
    int t_in_seq = blockIdx.y;
    int seq     = blockIdx.z;

    if (seq >= num_seqs || head >= n_heads || d >= half_dim) return;

    int seq_start = cu_seqlens[seq];
    int seq_end   = cu_seqlens[seq + 1];
    int slen = seq_end - seq_start;
    if (t_in_seq >= slen) return;

    float c = cos_table[t_in_seq * half_dim + d];
    float s_val = sin_table[t_in_seq * half_dim + d];

    int offset = seq_start + t_in_seq;
    __nv_bfloat16 *ptr = buf + (size_t)offset * n_heads * head_dim + head * head_dim;
    float v0 = __bfloat162float(ptr[d]);
    float v1 = __bfloat162float(ptr[d + half_dim]);
    ptr[d]            = __float2bfloat16(v0 * c - v1 * s_val);
    ptr[d + half_dim] = __float2bfloat16(v0 * s_val + v1 * c);
}

void rope_decoder_batched(__nv_bfloat16 *buf,
                          const int *cu_seqlens,
                          int num_seqs,
                          int max_seq_len,
                          int n_heads,
                          int head_dim) {
    if (num_seqs <= 0 || max_seq_len <= 0) return;
    int half_dim = head_dim / 2;
    dec_rope_ensure_table(max_seq_len + 1);

    dim3 block(half_dim);
    dim3 grid(n_heads, max_seq_len, num_seqs);
    rope_decoder_batched_kernel<<<grid, block>>>(
        buf, d_dec_rope_cos, d_dec_rope_sin, cu_seqlens,
        num_seqs, n_heads, head_dim, half_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void __launch_bounds__(128)
split_packed_qkv_rope_decoder_batched_kernel(
    __nv_bfloat16 *q_dst,
    __nv_bfloat16 *k_dst,
    __nv_bfloat16 *v_dst,
    const __nv_bfloat16 *packed,
    const float * __restrict__ cos_table,
    const float * __restrict__ sin_table,
    const int * __restrict__ cu_seqlens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int half_dim)
{
    int d = threadIdx.x;
    int head = blockIdx.x;
    int t_in_seq = blockIdx.y;
    int seq = blockIdx.z;

    if (seq >= num_seqs) return;
    int seq_start = cu_seqlens[seq];
    int seq_end = cu_seqlens[seq + 1];
    int slen = seq_end - seq_start;
    if (t_in_seq >= slen) return;

    int row = seq_start + t_in_seq;
    int hidden_dim = num_heads * head_dim;
    int kv_hidden = num_kv_heads * head_dim;
    int packed_cols = hidden_dim + 2 * kv_hidden;
    const __nv_bfloat16 *src = packed + (size_t)row * packed_cols;
    float c = d < half_dim ? cos_table[t_in_seq * half_dim + d] : 0.0f;
    float s_val = d < half_dim ? sin_table[t_in_seq * half_dim + d] : 0.0f;

    if (head < num_heads && d < half_dim) {
        const __nv_bfloat16 *q_src = src + head * head_dim;
        __nv_bfloat16 *q_out = q_dst + (size_t)row * hidden_dim + head * head_dim;
        float v0 = __bfloat162float(q_src[d]);
        float v1 = __bfloat162float(q_src[d + half_dim]);
        q_out[d] = __float2bfloat16(v0 * c - v1 * s_val);
        q_out[d + half_dim] = __float2bfloat16(v0 * s_val + v1 * c);
    }

    if (head < num_kv_heads) {
        const __nv_bfloat16 *k_src = src + hidden_dim + head * head_dim;
        const __nv_bfloat16 *v_src = src + hidden_dim + kv_hidden + head * head_dim;
        __nv_bfloat16 *k_out = k_dst + (size_t)row * kv_hidden + head * head_dim;
        __nv_bfloat16 *v_out = v_dst + (size_t)row * kv_hidden + head * head_dim;
        if (d < half_dim) {
            float v0 = __bfloat162float(k_src[d]);
            float v1 = __bfloat162float(k_src[d + half_dim]);
            k_out[d] = __float2bfloat16(v0 * c - v1 * s_val);
            k_out[d + half_dim] = __float2bfloat16(v0 * s_val + v1 * c);
        }
        if (d < head_dim) {
            v_out[d] = v_src[d];
        }
    }
}

void split_packed_qkv_rope_decoder_batched(__nv_bfloat16 *q_dst,
                                           __nv_bfloat16 *k_dst,
                                           __nv_bfloat16 *v_dst,
                                           const __nv_bfloat16 *packed,
                                           const int *cu_seqlens,
                                           int num_seqs,
                                           int max_seq_len,
                                           int num_heads,
                                           int num_kv_heads,
                                           int head_dim)
{
    if (num_seqs <= 0 || max_seq_len <= 0) return;
    int half_dim = head_dim / 2;
    dec_rope_ensure_table(max_seq_len + 1);
    dim3 block(head_dim);
    dim3 grid(num_heads, max_seq_len, num_seqs);
    split_packed_qkv_rope_decoder_batched_kernel<<<grid, block>>>(
        q_dst, k_dst, v_dst, packed, d_dec_rope_cos, d_dec_rope_sin,
        cu_seqlens, num_seqs, num_heads, num_kv_heads, head_dim, half_dim);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * Embedding lookup: out[i] = table[ids[i]]
 * ============================================================================ */
__global__ void embedding_lookup_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *table,
    const int *ids, int n, int dim
) {
    int i = blockIdx.x;
    if (i >= n) return;
    
    int vocab_id = ids[i];
    const __nv_bfloat16 *row = table + vocab_id * dim;
    __nv_bfloat16 *out_row = out + i * dim;
    
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        out_row[d] = row[d];
    }
}

void embedding_lookup(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                     const int *ids, int n, int dim) {
    int blockSize = 256;
    int gridSize = n;
    
    embedding_lookup_kernel<<<gridSize, blockSize>>>(out, table, ids, n, dim);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * Residual add: out = a + b (in-place on a)
 * ============================================================================ */
__global__ void residual_add_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16(av + bv);
}

void residual_add(__nv_bfloat16 *out, const __nv_bfloat16 *a,
                 const __nv_bfloat16 *b, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    residual_add_kernel<<<gridSize, blockSize>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void residual_add_bias_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *a,
    const __nv_bfloat16 *b,
    const __nv_bfloat16 *bias,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int col = idx % N;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    if (bias != NULL) {
        av += __bfloat162float(bias[col]);
    }
    av = __bfloat162float(__float2bfloat16(av));
    out[idx] = __float2bfloat16(av + bv);
}

void residual_add_bias(__nv_bfloat16 *out,
                       const __nv_bfloat16 *a,
                       const __nv_bfloat16 *b,
                       const __nv_bfloat16 *bias,
                       int M, int N) {
    int blockSize = 256;
    int total = M * N;
    int gridSize = (total + blockSize - 1) / blockSize;

    residual_add_bias_kernel<<<gridSize, blockSize>>>(out, a, b, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * Concat 4 frames: [seq, dim] → [seq/4, dim*4]
 * Concatenate 4 consecutive frames into one
 * ============================================================================ */
__global__ void concat_4frames_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *in,
    int seq_len, int dim
) {
    int out_seq = blockIdx.x;
    int in_seq_base = out_seq * 4;
    if (in_seq_base + 3 >= seq_len) return;

    __nv_bfloat16 *out_base = out + (size_t)out_seq * 4 * dim;
    for (int frame = 0; frame < 4; frame++) {
        const __nv_bfloat16 *in_frame = in + (size_t)(in_seq_base + frame) * dim;
        __nv_bfloat16 *out_frame = out_base + frame * dim;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
            out_frame[d] = in_frame[d];
    }
}

void concat_4frames(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                   int seq_len, int dim) {
    int out_seq_len = seq_len / 4;
    concat_4frames_kernel<<<out_seq_len, 256>>>(out, in, seq_len, dim);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * BF16 Linear: Y[M,N] = X[M,K] @ W[N,K]^T + bias
 * Row-major layout. cuBLAS: interpret as col-major transposed
 * ============================================================================ */

// Add bias kernel definition
__global__ void add_bias_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    
    int m = idx / N;
    int n = idx % N;
    
    float val = __bfloat162float(out[idx]);
    float b = __bfloat162float(bias[n]);
    out[idx] = __float2bfloat16(val + b);
}

__global__ void add_bias_gelu_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int n = idx % N;
    float val = __bfloat162float(out[idx]);
    if (bias != NULL) {
        val += __bfloat162float(bias[n]);
    }
    float tanh_arg = 0.79788456f * (val + 0.044715f * val * val * val);
    float gelu = 0.5f * val * (1.0f + tanhf(tanh_arg));
    out[idx] = __float2bfloat16(gelu);
}

static int bf16_linear_lt_bias(cublasHandle_t handle,
                               __nv_bfloat16 *out,
                               const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w,
                               const __nv_bfloat16 *bias,
                               int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasLtMatmulDesc_t matmul_desc = NULL;
    cublasLtMatrixLayout_t layout_a = NULL;
    cublasLtMatrixLayout_t layout_b = NULL;
    cublasLtMatrixLayout_t layout_c = NULL;
    cublasLtMatrixLayout_t layout_d = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasStatus_t status;
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cudaDataType_t bias_type = CUDA_R_16BF;
    int returned = 0;
    int ok = 0;
    cudaStream_t stream = NULL;

    if (bias == NULL || M <= 0 || N <= 0 || K <= 0) {
        return 0;
    }
    if (!ensure_bf16_lt_resources()) {
        return 0;
    }
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) {
        stream = NULL;
    }

    status = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, K, N, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulPreferenceSetAttribute(preference,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &g_bf16_lt_workspace_size,
                                                  sizeof(g_bf16_lt_workspace_size));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulAlgoGetHeuristic(g_bf16_lt_handle, matmul_desc,
                                            layout_a, layout_b, layout_c, layout_d,
                                            preference, 1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    status = cublasLtMatmul(g_bf16_lt_handle, matmul_desc,
                            &alpha,
                            w, layout_a,
                            x, layout_b,
                            &beta,
                            out, layout_c,
                            out, layout_d,
                            &heuristic.algo,
                            g_bf16_lt_workspace,
                            g_bf16_lt_workspace_size,
                            stream);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    ok = 1;

cleanup:
    if (preference != NULL) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_d != NULL) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != NULL) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != NULL) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != NULL) cublasLtMatrixLayoutDestroy(layout_a);
    if (matmul_desc != NULL) cublasLtMatmulDescDestroy(matmul_desc);
    return ok;
}

static int bf16_linear_lt_gelu_bias(cublasHandle_t handle,
                                    __nv_bfloat16 *out,
                                    const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w,
                                    const __nv_bfloat16 *bias,
                                    int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasLtMatmulDesc_t matmul_desc = NULL;
    cublasLtMatrixLayout_t layout_a = NULL;
    cublasLtMatrixLayout_t layout_b = NULL;
    cublasLtMatrixLayout_t layout_c = NULL;
    cublasLtMatrixLayout_t layout_d = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasStatus_t status;
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    cudaDataType_t bias_type = CUDA_R_16BF;
    int returned = 0;
    int ok = 0;
    cudaStream_t stream = NULL;

    if (bias == NULL) {
        return 0;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return 0;
    }
    if (!ensure_bf16_lt_resources()) {
        return 0;
    }
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) {
        stream = NULL;
    }

    status = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, K, N, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulPreferenceSetAttribute(preference,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &g_bf16_lt_workspace_size,
                                                  sizeof(g_bf16_lt_workspace_size));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulAlgoGetHeuristic(g_bf16_lt_handle, matmul_desc,
                                            layout_a, layout_b, layout_c, layout_d,
                                            preference, 1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    status = cublasLtMatmul(g_bf16_lt_handle, matmul_desc,
                            &alpha,
                            w, layout_a,
                            x, layout_b,
                            &beta,
                            out, layout_c,
                            out, layout_d,
                            &heuristic.algo,
                            g_bf16_lt_workspace,
                            g_bf16_lt_workspace_size,
                            stream);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    ok = 1;

cleanup:
    if (preference != NULL) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_d != NULL) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != NULL) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != NULL) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != NULL) cublasLtMatrixLayoutDestroy(layout_a);
    if (matmul_desc != NULL) cublasLtMatmulDescDestroy(matmul_desc);
    return ok;
}

static int bf16_linear_lt_bias_residual(cublasHandle_t handle,
                                        __nv_bfloat16 *out,
                                        const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w,
                                        const __nv_bfloat16 *bias,
                                        const __nv_bfloat16 *residual,
                                        int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasLtMatmulDesc_t matmul_desc = NULL;
    cublasLtMatrixLayout_t layout_a = NULL;
    cublasLtMatrixLayout_t layout_b = NULL;
    cublasLtMatrixLayout_t layout_c = NULL;
    cublasLtMatrixLayout_t layout_d = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasStatus_t status;
    cublasOperation_t op_a = CUBLAS_OP_T;
    cublasOperation_t op_b = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cudaDataType_t bias_type = CUDA_R_16BF;
    int returned = 0;
    int ok = 0;
    cudaStream_t stream = NULL;

    if (bias == NULL || residual == NULL || M <= 0 || N <= 0 || K <= 0) {
        return 0;
    }
    if (!ensure_bf16_lt_resources()) {
        return 0;
    }
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) {
        stream = NULL;
    }

    status = cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, K, N, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, K, M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, N, M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulPreferenceSetAttribute(preference,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &g_bf16_lt_workspace_size,
                                                  sizeof(g_bf16_lt_workspace_size));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulAlgoGetHeuristic(g_bf16_lt_handle, matmul_desc,
                                            layout_a, layout_b, layout_c, layout_d,
                                            preference, 1, &heuristic, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        goto cleanup;
    }

    status = cublasLtMatmul(g_bf16_lt_handle, matmul_desc,
                            &alpha,
                            w, layout_a,
                            x, layout_b,
                            &beta,
                            residual, layout_c,
                            out, layout_d,
                            &heuristic.algo,
                            g_bf16_lt_workspace,
                            g_bf16_lt_workspace_size,
                            stream);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    ok = 1;

cleanup:
    if (preference != NULL) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_d != NULL) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != NULL) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != NULL) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != NULL) cublasLtMatrixLayoutDestroy(layout_a);
    if (matmul_desc != NULL) cublasLtMatmulDescDestroy(matmul_desc);
    return ok;
}

void bf16_linear(cublasHandle_t handle,
                 __nv_bfloat16 *out,
                 const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                 const __nv_bfloat16 *bias,
                 int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS is column-major, so we transpose the operation
    // Y[M,N] = X[M,K] @ W[N,K]^T -> col-major: C[N,M] = B[K,M] @ A[N,K]^T
    // In row-major: C^T = (A[K,N])^T @ (B[K,M])^T = A[N,K] @ B[K,M]
    // So we call: cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...)

    if (bias != NULL && bf16_linear_lt_bias_enabled() &&
        bf16_linear_lt_bias(handle, out, x, w, bias, M, N, K)) {
        return;
    }
    
    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,           // Transpose W: W[N,K] -> W[K,N]
        CUBLAS_OP_N,           // No transpose X: X[M,K]
        N, M, K,               // N, M, K
        &alpha,
        w, CUDA_R_16BF, K,     // W[K,N] in bf16
        x, CUDA_R_16BF, K,     // X[M,K] in bf16
        &beta,
        out, CUDA_R_16BF, N,   // Y[M,N] in bf16
        CUBLAS_COMPUTE_32F,
        bf16_linear_gemm_algo()
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error in bf16_linear\n");
        exit(1);
    }
    
    // Add bias if provided
    if (bias != NULL) {
        int blockSize = 256;
        int gridSize = (M * N + blockSize - 1) / blockSize;
        
        // Each thread handles one element: out[m,n] += bias[n]
        add_bias_kernel<<<gridSize, blockSize>>>(out, bias, M, N);
        CUDA_CHECK(cudaGetLastError());
    }
}

void bf16_linear_bias_residual(cublasHandle_t handle,
                               __nv_bfloat16 *out,
                               const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w,
                               const __nv_bfloat16 *bias,
                               const __nv_bfloat16 *residual,
                               int M, int N, int K) {
    if (residual != NULL && bf16_linear_lt_bias_enabled() &&
        bf16_linear_lt_bias_residual(handle, out, x, w, bias, residual, M, N, K)) {
        return;
    }

    bf16_linear(handle, out, x, w, NULL, M, N, K);
    residual_add_bias(out, out, residual, bias, M, N);
}

void bf16_linear_gelu(cublasHandle_t handle,
                      __nv_bfloat16 *out,
                      const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                      const __nv_bfloat16 *bias,
                      int M, int N, int K) {
    if (bias != NULL && bf16_linear_lt_bias_enabled() &&
        bf16_linear_lt_gelu_bias(handle, out, x, w, bias, M, N, K)) {
        return;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;
    int blockSize;
    int gridSize;

    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w, CUDA_R_16BF, K,
        x, CUDA_R_16BF, K,
        &beta,
        out, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        bf16_linear_gemm_algo()
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error in bf16_linear_gelu\n");
        exit(1);
    }

    blockSize = 256;
    gridSize = (M * N + blockSize - 1) / blockSize;
    add_bias_gelu_kernel<<<gridSize, blockSize>>>(out, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void silu_mul_split_packed_kernel(__nv_bfloat16 *out,
                                             const __nv_bfloat16 *packed_gate_up,
                                             int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int m = idx / N;
    int n = idx % N;
    const __nv_bfloat16 *row = packed_gate_up + (size_t)m * (2 * N);
    float gate = __bfloat162float(row[n]);
    float up = __bfloat162float(row[N + n]);
    float sig = 1.0f / (1.0f + expf(-gate));
    out[idx] = __float2bfloat16((gate * sig) * up);
}

void silu_mul_split_packed(__nv_bfloat16 *out,
                           const __nv_bfloat16 *packed_gate_up,
                           int M, int N) {
    int blockSize = 256;
    int gridSize = (M * N + blockSize - 1) / blockSize;
    silu_mul_split_packed_kernel<<<gridSize, blockSize>>>(out, packed_gate_up, M, N);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================================================================
 * FP4 Linear: FP4 weight GEMM
 * ============================================================================ */

// Pre-quant scale kernel definition
__global__ void apply_prequant_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *pre_quant_scale,
    int M, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    
    int m = idx / K;
    int k = idx % K;
    
    float val = __bfloat162float(x[idx]);
    float scale = __bfloat162float(pre_quant_scale[k]);
    out[idx] = __float2bfloat16(val * scale);
}

void fp4_linear(cublasHandle_t handle,
                __nv_bfloat16 *out,
                __nv_bfloat16 *weight_scratch,
                __nv_bfloat16 *prequant_scratch,
                const __nv_bfloat16 *x,
                const uint8_t *w_packed, const uint8_t *w_scales,
                float w_global_scale,
                const __nv_bfloat16 *pre_quant_scale,
                const __nv_bfloat16 *bias,
                int M, int N, int K) {
    const __nv_bfloat16 *x_to_use = x;

    if (pre_quant_scale != NULL) {
        int total = M * K;
        apply_prequant_kernel<<<(total+255)/256, 256>>>(prequant_scratch, x, pre_quant_scale, M, K);
        CUDA_CHECK(cudaGetLastError());
        x_to_use = prequant_scratch;
    }

    fp4_dequant_to_bf16(weight_scratch, w_packed, w_scales, w_global_scale, N, K);
    bf16_linear(handle, out, x_to_use, weight_scratch, bias, M, N, K);
}

/* ============================================================================
 * LM Head Argmax: Compute logits for ONE position, return argmax
 * logits[v] = dot(hidden[dim], lm_weight[v, dim])
 * ============================================================================ */

// Compute logits kernel
__global__ void compute_logits_kernel(
    float *logits,
    const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *lm_weight,
    int dim, int vocab_size
) {
    int v = blockIdx.x;
    if (v >= vocab_size) return;
    
    const __nv_bfloat16 *weight_row = lm_weight + v * dim;
    
    float dot = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float h = __bfloat162float(hidden[d]);
        float w = __bfloat162float(weight_row[d]);
        dot += h * w;
    }
    
    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
    }
    
    if (threadIdx.x == 0) {
        logits[v] = dot;
    }
}

// Find argmax kernel for float logits
__global__ void find_argmax_f32_kernel(const float *logits, int *out_token, int vocab_size) {
    __shared__ float sh_max[256];
    __shared__ int sh_idx[256];
    int tid = threadIdx.x;
    float best_val = -1e20f;
    int best_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    sh_max[tid] = best_val;
    sh_idx[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && sh_max[tid + offset] > sh_max[tid]) {
            sh_max[tid] = sh_max[tid + offset];
            sh_idx[tid] = sh_idx[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = sh_idx[0];
    }
}

__global__ void find_argmax_bf16_batched_kernel(const __nv_bfloat16 *logits,
                                                int *out_tokens,
                                                int rows,
                                                int vocab_size) {
    __shared__ float sh_max[256];
    __shared__ int sh_idx[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float best_val = -1e20f;
    int best_idx = 0;

    if (row >= rows) {
        return;
    }

    logits += (size_t) row * (size_t) vocab_size;
    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(logits[v]);
        if (val > best_val) {
            best_val = val;
            best_idx = v;
        }
    }

    sh_max[tid] = best_val;
    sh_idx[tid] = best_idx;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && sh_max[tid + offset] > sh_max[tid]) {
            sh_max[tid] = sh_max[tid + offset];
            sh_idx[tid] = sh_idx[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_tokens[row] = sh_idx[0];
    }
}

void lm_head_argmax(const __nv_bfloat16 *hidden,
                    const __nv_bfloat16 *lm_weight,
                    int dim, int vocab_size, int *out_token) {
    ensure_lm_head_single_buffers(vocab_size);

    /* 32 threads per block = one warp, so the warp-shuffle reduction is correct.
     * Each thread accumulates dim/32 elements then the warp reduces to thread 0. */
    compute_logits_kernel<<<vocab_size, 32>>>(d_lm_head_logits_f32, hidden, lm_weight, dim, vocab_size);
    CUDA_CHECK(cudaGetLastError());

    find_argmax_f32_kernel<<<1, 256>>>(d_lm_head_logits_f32, d_lm_head_tokens, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_token, d_lm_head_tokens, sizeof(int), cudaMemcpyDeviceToHost));
}

void lm_head_argmax_batched(cublasHandle_t handle,
                            const __nv_bfloat16 *hidden,
                            const __nv_bfloat16 *lm_weight,
                            int rows, int dim, int vocab_size,
                            int *out_tokens) {
    if (rows <= 0) {
        return;
    }

    ensure_lm_head_buffers(rows, vocab_size);
    bf16_linear(handle, d_lm_head_logits, hidden, lm_weight, NULL, rows, vocab_size, dim);
    find_argmax_bf16_batched_kernel<<<rows, 256>>>(d_lm_head_logits, d_lm_head_tokens, rows, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out_tokens, d_lm_head_tokens,
                          (size_t) rows * sizeof(*out_tokens),
                          cudaMemcpyDeviceToHost));
}

void lm_head_cleanup(void) {
    if (d_lm_head_logits != NULL) {
        CUDA_CHECK(cudaFree(d_lm_head_logits));
        d_lm_head_logits = NULL;
    }
    if (d_lm_head_logits_f32 != NULL) {
        CUDA_CHECK(cudaFree(d_lm_head_logits_f32));
        d_lm_head_logits_f32 = NULL;
    }
    if (d_lm_head_tokens != NULL) {
        CUDA_CHECK(cudaFree(d_lm_head_tokens));
        d_lm_head_tokens = NULL;
    }
    lm_head_logits_capacity = 0;
    lm_head_logits_f32_capacity = 0;
    lm_head_token_capacity = 0;
}

/* ============================================================================
 * KV Cache Allocation/Free
 * ============================================================================ */
void kv_cache_alloc(KVCache *kv, const ModelConfig *cfg, int max_seq) {
    kv->seq_len = 0;
    kv->max_seq = max_seq;
    
    int kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    
    /* +1 for NULL sentinel so kv_cache_free's loop terminates correctly */
    kv->layers = (LayerKV *)calloc(cfg->dec_layers + 1, sizeof(LayerKV));
    
    for (int l = 0; l < cfg->dec_layers; l++) {
        size_t k_size = kv_heads * max_seq * head_dim * sizeof(__nv_bfloat16);
        
        CUDA_CHECK(cudaMalloc(&kv->layers[l].k.data, k_size));
        CUDA_CHECK(cudaMalloc(&kv->layers[l].v.data, k_size));
        
        kv->layers[l].k.dtype = DTYPE_BF16;
        kv->layers[l].k.ndim = 4;
        kv->layers[l].k.shape[0] = 1;  // batch
        kv->layers[l].k.shape[1] = kv_heads;
        kv->layers[l].k.shape[2] = max_seq;
        kv->layers[l].k.shape[3] = head_dim;
        kv->layers[l].k.nbytes = k_size;
        
        kv->layers[l].v.dtype = DTYPE_BF16;
        kv->layers[l].v.ndim = 4;
        kv->layers[l].v.shape[0] = 1;
        kv->layers[l].v.shape[1] = kv_heads;
        kv->layers[l].v.shape[2] = max_seq;
        kv->layers[l].v.shape[3] = head_dim;
        kv->layers[l].v.nbytes = k_size;
    }
}

void kv_cache_free(KVCache *kv) {
    if (kv->layers == NULL) return;
    
    for (int l = 0; kv->layers[l].k.data != NULL; l++) {
        CUDA_CHECK(cudaFree(kv->layers[l].k.data));
        CUDA_CHECK(cudaFree(kv->layers[l].v.data));
    }
    
    free(kv->layers);
    kv->layers = NULL;
    kv->seq_len = 0;
    kv->max_seq = 0;
}

__global__ void extract_conv_weight_slice(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int out_ch, int in_ch, int kernel_size, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_ch * in_ch) return;
    int oc = idx / in_ch;
    int ic = idx % in_ch;
    dst[idx] = src[oc * in_ch * kernel_size + ic * kernel_size + k];
}

__global__ void add_bias_2d_kernel(__nv_bfloat16 *data, const __nv_bfloat16 *bias,
                                   int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    float v = __bfloat162float(data[idx]) + __bfloat162float(bias[r]);
    data[idx] = __float2bfloat16(v);
}

__global__ void strided_copy_kernel(__nv_bfloat16 *dst, const __nv_bfloat16 *src,
                                     int channels, int out_len, int in_len,
                                     int stride, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * out_len) return;
    int c = idx / out_len;
    int f = idx % out_len;
    int src_f = f * stride + offset;
    dst[c * out_len + f] = (src_f >= 0 && src_f < in_len) ? src[c * in_len + src_f] : __float2bfloat16(0.0f);
}

void conv1d_forward(__nv_bfloat16 *out, const __nv_bfloat16 *input,
                    const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
                    int in_ch, int out_ch, int seq_len, int kernel_size,
                    int stride, int padding) {
    int out_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
    static cublasHandle_t conv_handle = NULL;
    static __nv_bfloat16 *shifted = NULL;
    static size_t shifted_capacity = 0;
    static __nv_bfloat16 *w_slice = NULL;
    static size_t w_slice_capacity = 0;
    size_t shifted_elems = (size_t)in_ch * (size_t)out_len;
    size_t w_slice_elems = (size_t)out_ch * (size_t)in_ch;

    if (conv_handle == NULL) {
        cublasStatus_t st = cublasCreate(&conv_handle);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "conv1d cublasCreate failed: %d", st);
    }

    if (shifted_capacity < shifted_elems) {
        if (shifted != NULL) {
            cudaFree(shifted);
            shifted = NULL;
        }
        CUDA_CHECK(cudaMalloc(&shifted, shifted_elems * sizeof(__nv_bfloat16)));
        shifted_capacity = shifted_elems;
    }

    cudaMemset(out, 0, out_ch * out_len * sizeof(__nv_bfloat16));

    if (w_slice_capacity < w_slice_elems) {
        if (w_slice != NULL) {
            cudaFree(w_slice);
            w_slice = NULL;
        }
        CUDA_CHECK(cudaMalloc(&w_slice, w_slice_elems * sizeof(__nv_bfloat16)));
        w_slice_capacity = w_slice_elems;
    }

    float alpha = 1.0f, beta;
    for (int k = 0; k < kernel_size; k++) {
        int src_offset = k - padding;
        int n = in_ch * out_len;
        strided_copy_kernel<<<(n+255)/256, 256>>>(
            shifted, input, in_ch, out_len, seq_len, stride, src_offset);

        {
            int wn = out_ch * in_ch;
            extract_conv_weight_slice<<<(wn+255)/256, 256>>>(
                w_slice, weight, out_ch, in_ch, kernel_size, k);
        }

        beta = (k == 0) ? 0.0f : 1.0f;

        cublasStatus_t st = cublasGemmEx(conv_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     out_len, out_ch, in_ch,
                     &alpha,
                     shifted, CUDA_R_16BF, out_len,
                     w_slice, CUDA_R_16BF, in_ch,
                     &beta,
                     out, CUDA_R_16BF, out_len,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "conv1d GEMM failed at k=%d status=%d", k, st);
    }

    if (bias) {
        int total = out_ch * out_len;
        add_bias_2d_kernel<<<(total+255)/256, 256>>>(out, bias, out_ch, out_len);
    }
}

__global__ void transpose_kernel(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                                  int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[r * cols + c];
}

void transpose_2d(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                  int rows, int cols) {
    int n = rows * cols;
    transpose_kernel<<<(n+255)/256, 256>>>(out, in, rows, cols);
}

__global__ void softmax_kernel(__nv_bfloat16 *data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16 *r = data + row * cols;
    float max_v = -1e20f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(r[j]);
        if (v > max_v) max_v = v;
    }
    for (int off = 16; off > 0; off >>= 1)
        max_v = fmaxf(max_v, __shfl_down_sync(0xffffffff, max_v, off));
    max_v = __shfl_sync(0xffffffff, max_v, 0);

    float sum_v = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = expf(__bfloat162float(r[j]) - max_v);
        r[j] = __float2bfloat16(v);
        sum_v += v;
    }
    for (int off = 16; off > 0; off >>= 1)
        sum_v += __shfl_down_sync(0xffffffff, sum_v, off);
    sum_v = __shfl_sync(0xffffffff, sum_v, 0);

    float inv_sum = 1.0f / (sum_v + 1e-9f);
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        r[j] = __float2bfloat16(__bfloat162float(r[j]) * inv_sum);
}

void softmax_inplace(__nv_bfloat16 *data, int rows, int cols) {
    softmax_kernel<<<rows, 32>>>(data, rows, cols);
}

} /* extern "C" */
