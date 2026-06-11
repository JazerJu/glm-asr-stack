/*
 * GLM-ASR Encoder Forward Pass CUDA Implementation
 * 32-layer audio encoder with bidirectional attention
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "kernels.cuh"
#include "flash_fwd_vllm.h"
#include "../include/types.h"

#define MAX_SEQ_LEN 3000
#define WARP_SIZE 32

// Device memory for scratch buffers
static __nv_bfloat16 *d_q_buf = NULL;
static __nv_bfloat16 *d_k_buf = NULL;
static __nv_bfloat16 *d_v_buf = NULL;
static __nv_bfloat16 *d_qkv_buf = NULL;
static __nv_bfloat16 *d_qk_buf = NULL;
static __nv_bfloat16 *d_attn_buf = NULL;
static __nv_bfloat16 *d_attn_out = NULL;
static __nv_bfloat16 *d_hidden_buf = NULL;
static int *d_encoder_seq_lens = NULL;
static int *d_encoder_seq_offsets = NULL;
static int *d_encoder_row_pos = NULL;
static int *d_encoder_cu_seqlens = NULL;
static int  d_encoder_cu_seqlens_cap = 0;
static float *d_enc_rope_cos = NULL;
static float *d_enc_rope_sin = NULL;
static int    d_enc_rope_table_cap = 0;
static float *d_softmax_max = NULL;
static float *d_softmax_sum = NULL;
static bool buffers_allocated = false;
static int encoder_row_capacity = 0;
static int encoder_attn_seq_capacity = 0;
static ScratchProfileStats encoder_scratch_stats = {};
static EncoderAttentionProfileStats encoder_attention_stats = {};
static size_t encoder_qkv_bytes = 0;
static size_t encoder_qkv_fused_out_bytes = 0;
static size_t encoder_attn_matrix_bytes = 0;
static size_t encoder_attn_out_bytes = 0;
static size_t encoder_hidden_ffn_bytes = 0;
static size_t encoder_softmax_bytes = 0;
static size_t encoder_seq_meta_bytes = 0;
static size_t encoder_row_pos_bytes = 0;
#define ENCODER_MAX_LAYERS 64
static __nv_bfloat16 *d_encoder_qkv_fused_w[ENCODER_MAX_LAYERS] = {};
static __nv_bfloat16 *d_encoder_qkv_fused_b[ENCODER_MAX_LAYERS] = {};

static void encoder_note_alloc(size_t bytes) {
    encoder_scratch_stats.alloc_calls++;
    encoder_scratch_stats.alloc_bytes += (uint64_t)bytes;
    encoder_scratch_stats.current_bytes += (uint64_t)bytes;
    if (encoder_scratch_stats.current_bytes > encoder_scratch_stats.peak_bytes) {
        encoder_scratch_stats.peak_bytes = encoder_scratch_stats.current_bytes;
    }
}

static void encoder_note_free(size_t bytes) {
    encoder_scratch_stats.free_calls++;
    encoder_scratch_stats.free_bytes += (uint64_t)bytes;
    if (encoder_scratch_stats.current_bytes >= (uint64_t)bytes) {
        encoder_scratch_stats.current_bytes -= (uint64_t)bytes;
    } else {
        encoder_scratch_stats.current_bytes = 0;
    }
}

static void encoder_qkv_fused_cache_free(void) {
    for (int layer = 0; layer < ENCODER_MAX_LAYERS; layer++) {
        if (d_encoder_qkv_fused_w[layer] != NULL) {
            cudaFree(d_encoder_qkv_fused_w[layer]);
            d_encoder_qkv_fused_w[layer] = NULL;
        }
        if (d_encoder_qkv_fused_b[layer] != NULL) {
            cudaFree(d_encoder_qkv_fused_b[layer]);
            d_encoder_qkv_fused_b[layer] = NULL;
        }
    }
}

static void encoder_attention_note_fa(int seq_len, int num_heads, int used_pipeline) {
    if (seq_len <= 0 || num_heads <= 0) {
        return;
    }

    encoder_attention_stats.fa_calls++;
    if (used_pipeline) {
        encoder_attention_stats.fa_pipeline_calls++;
    } else {
        encoder_attention_stats.fa_simple_calls++;
    }
    encoder_attention_stats.fa_rows += (uint64_t)seq_len;
    encoder_attention_stats.fa_score_elems +=
        (uint64_t)seq_len * (uint64_t)seq_len * (uint64_t)num_heads;
}

static void encoder_attention_note_cublas(int seq_len, int num_heads) {
    if (seq_len <= 0 || num_heads <= 0) {
        return;
    }

    encoder_attention_stats.cublas_calls++;
    encoder_attention_stats.cublas_rows += (uint64_t)seq_len;
    encoder_attention_stats.cublas_score_elems +=
        (uint64_t)seq_len * (uint64_t)seq_len * (uint64_t)num_heads;
}

static void free_encoder_buffers(void);

#define ENCODER_FA_HEAD_DIM 64
#define ENCODER_FA_HEAD_PAIRS (ENCODER_FA_HEAD_DIM / 2)
#define ENCODER_FA_WARPS 8
#define ENCODER_FA_THREADS (ENCODER_FA_WARPS * WARP_SIZE)
#define ENCODER_FA_SUBGROUP_SIZE (WARP_SIZE / 2)
#define ENCODER_FA_ROWS_PER_WARP 2
#define ENCODER_FA_Q_TILE (ENCODER_FA_WARPS * ENCODER_FA_ROWS_PER_WARP)
#define ENCODER_FA_HEAD_QUADS (ENCODER_FA_HEAD_PAIRS / 2)
#define ENCODER_FA_HEAD_OCTETS (ENCODER_FA_HEAD_PAIRS / 4)
#define ENCODER_FA_PIPE_PAD 4
#define ENCODER_FA_PV_MMA_K_TILE 16
#define ENCODER_FA_PV_MMA_WARPS 8
#define ENCODER_FA_PV_MMA_THREADS (ENCODER_FA_PV_MMA_WARPS * WARP_SIZE)
#define ENCODER_FA_PV_MMA_SCORE_TILE 16
#define ENCODER_FA_PV_MMA_SCORE_STRIDE (ENCODER_FA_PV_MMA_K_TILE + 8)
#define ENCODER_FA_PV_MMA_K_STRIDE (ENCODER_FA_HEAD_DIM + 8)
#define ENCODER_FA_PV_MMA_N_TILE 16
#define ENCODER_FA_PV_MMA_N_WARPS (ENCODER_FA_HEAD_DIM / ENCODER_FA_PV_MMA_N_TILE)
#define ENCODER_FA_PV_MMA_PROB_STRIDE (ENCODER_FA_PV_MMA_K_TILE + 8)
#define ENCODER_FA_PV_MMA_ACC_STRIDE (ENCODER_FA_HEAD_DIM + 8)
#define ENCODER_FA_Q32_TILE 32
#define ENCODER_FA_Q32_WARPS 8
#define ENCODER_FA_Q32_THREADS (ENCODER_FA_Q32_WARPS * WARP_SIZE)
#define ENCODER_FA_Q32_ROW_BLOCK 16
#define ENCODER_FA_Q32_PV_WARPS (2 * ENCODER_FA_PV_MMA_N_WARPS)
#define ENCODER_FA_Q32_SCORE_STRIDE ENCODER_FA_PV_MMA_SCORE_STRIDE
#define ENCODER_FA_Q32_PROB_STRIDE ENCODER_FA_PV_MMA_PROB_STRIDE
#define ENCODER_FA_Q32_ACC_STRIDE ENCODER_FA_PV_MMA_ACC_STRIDE
#define ENCODER_FA_Q32_V_PAD ENCODER_FA_PIPE_PAD
#define ENCODER_FA_Q32_K32_K_TILE 32
#define ENCODER_FA_Q32_K32_K_STRIDE ENCODER_FA_PV_MMA_K_STRIDE
#define ENCODER_FA_Q32_K32_V_PAD ENCODER_FA_Q32_V_PAD
#define ENCODER_FA_Q32_K32_SCORE_STRIDE (ENCODER_FA_Q32_K32_K_TILE + 8)
#define ENCODER_FA_Q32_K32_PROB_STRIDE (ENCODER_FA_Q32_K32_K_TILE + 8)
#define ENCODER_FA_FA2_Q_TILE 64
#define ENCODER_FA_FA2_K_TILE 128
#define ENCODER_FA_FA2_WARPS 8
#define ENCODER_FA_FA2_THREADS (ENCODER_FA_FA2_WARPS * WARP_SIZE)
#define ENCODER_FA_FA2_ROW_BLOCK 16
#define ENCODER_FA_FA2_SCORE_TILE 16
#define ENCODER_FA_FA2_K_STRIDE (ENCODER_FA_HEAD_DIM + 8)
#define ENCODER_FA_FA2_V_PAD ENCODER_FA_PIPE_PAD
#define ENCODER_FA_FA2_PROB_STRIDE (ENCODER_FA_FA2_K_TILE + 8)
#define ENCODER_FA_FA2_QK_WARPS 4
#define ENCODER_FA_FA2_PV_WARPS ENCODER_FA_FA2_WARPS
#define ENCODER_FA_FA2_N_TILE 32
#define ENCODER_FA_FA2_N_SUBS 2
#define ENCODER_FA_DEFAULT_MIN_SEQ 1500
#define ENCODER_FA_DEFAULT_LONG_MIN_SEQ ENCODER_FA_DEFAULT_MIN_SEQ
#define MAX_FA_VARLEN_BATCH 256
#define ENCODER_FA_DEFAULT_SMEM_LIMIT (99 * 1024)
#define ENCODER_QK_MMA_TILE 16
__device__ __constant__ float k_encoder_rope_inv_freq_even[8] = {
    1.000000000000f, 0.316227766017f, 0.100000000000f, 0.031622776602f,
    0.010000000000f, 0.003162277660f, 0.001000000000f, 0.000316227766f
};
__device__ __constant__ float k_encoder_rope_inv_freq_odd[8] = {
    0.562341325190f, 0.177827941004f, 0.056234132519f, 0.017782794100f,
    0.005623413252f, 0.001778279410f, 0.000562341325f, 0.000177827941f
};

static int encoder_env_int_bounded(const char *name, int default_value, int min_value, int max_value) {
    const char *env = getenv(name);
    char *endptr;
    long parsed;

    if (env == NULL || env[0] == '\0') {
        return default_value;
    }

    parsed = strtol(env, &endptr, 10);
    if (endptr == env || *endptr != '\0' || parsed < (long)min_value || parsed > (long)max_value) {
        return default_value;
    }

    return (int)parsed;
}

static int encoder_fa_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_ENABLE", 1, 0, 1);
    }
    return cached;
}

static int encoder_fa_force_fallback(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_FORCE_FALLBACK", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_pipelined_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_PIPELINED", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_pv_mma_q32_k32_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_PV_MMA_Q32_K32", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_mma_batched_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_BATCHED", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_fa2_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_FA2", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_vllm_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_VLLM", 0, 0, 1);
    }
    return cached;
}

static int encoder_fa_dense_tma_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_DENSE_TMA", 0, 0, 1);
    }
    return cached;
}

static int encoder_qkv_fused_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_QKV_FUSED", 1, 0, 1);
    }
    return cached;
}

static int encoder_fa_legacy_min_seq_len(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = encoder_env_int_bounded("GLMASR_ENCODER_FA_MIN_SEQ",
                                         ENCODER_FA_DEFAULT_MIN_SEQ,
                                         1,
                                         MAX_SEQ_LEN);
    }
    return cached;
}

static int encoder_fa_medium_min_seq_len(void) {
    static int cached = -1;
    const char *env;
    char *endptr;
    long parsed;

    if (cached >= 0) {
        return cached;
    }

    env = getenv("GLMASR_ENCODER_FA_MEDIUM_MIN_SEQ");
    if (env == NULL || env[0] == '\0') {
        cached = encoder_fa_legacy_min_seq_len();
        return cached;
    }

    parsed = strtol(env, &endptr, 10);
    if (endptr == env || *endptr != '\0') {
        cached = encoder_fa_legacy_min_seq_len();
        return cached;
    }
    if (parsed < 1) {
        parsed = 1;
    }
    if (parsed > MAX_SEQ_LEN) {
        parsed = MAX_SEQ_LEN;
    }
    cached = (int)parsed;
    return cached;
}

static int encoder_fa_long_min_seq_len(void) {
    static int cached = -1;
    int medium_min;
    const char *env;
    char *endptr;
    long parsed;

    if (cached >= 0) {
        return cached;
    }

    medium_min = encoder_fa_medium_min_seq_len();
    env = getenv("GLMASR_ENCODER_FA_LONG_MIN_SEQ");
    if (env == NULL || env[0] == '\0') {
        cached = ENCODER_FA_DEFAULT_LONG_MIN_SEQ;
    } else {
        parsed = strtol(env, &endptr, 10);
        if (endptr == env || *endptr != '\0') {
            cached = ENCODER_FA_DEFAULT_LONG_MIN_SEQ;
        } else {
            if (parsed < 1) {
                parsed = 1;
            }
            if (parsed > MAX_SEQ_LEN) {
                parsed = MAX_SEQ_LEN;
            }
            cached = (int)parsed;
        }
    }

    if (cached < medium_min) {
        cached = medium_min;
    }
    return cached;
}

static int encoder_qk_mma_supported(void) {
    static int cached = -1;
    int major = 0;
    int minor = 0;
    cudaError_t st;

    if (cached >= 0) {
        return cached;
    }

    st = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    if (st == cudaSuccess) {
        st = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    }
    if (st != cudaSuccess) {
        cached = 0;
        return cached;
    }

    cached = (major > 8 || (major == 8 && minor >= 0)) ? 1 : 0;
    return cached;
}

static int encoder_fa_common_eligible(int hidden_dim, int num_heads, int head_dim) {
    if (!encoder_fa_enabled() || encoder_fa_force_fallback()) {
        return 0;
    }
    if (head_dim != ENCODER_FA_HEAD_DIM || num_heads <= 0 || hidden_dim <= 0) {
        return 0;
    }
    if (hidden_dim < num_heads * head_dim) {
        return 0;
    }
    return 1;
}

static int encoder_fa_pipeline_can_implement(int seq_len,
                                              int hidden_dim,
                                              int num_heads,
                                              int head_dim) {
    if (!encoder_fa_pipelined_enabled()) {
        return 0;
    }
    if (!encoder_fa_common_eligible(hidden_dim, num_heads, head_dim)) {
        return 0;
    }
    if (seq_len < encoder_fa_long_min_seq_len()) {
        return 0;
    }
    return 1;
}

__device__ __forceinline__ void encoder_rope_partial_pair(float2 in_low,
                                                           float2 in_high,
                                                           int seq_pos,
                                                           int pair_idx,
                                                           float2 *out_low,
                                                           float2 *out_high) {
    float sin0;
    float cos0;
    float sin1;
    float cos1;
    float theta0 = (float)seq_pos * k_encoder_rope_inv_freq_even[pair_idx];
    float theta1 = (float)seq_pos * k_encoder_rope_inv_freq_odd[pair_idx];
    __sincosf(theta0, &sin0, &cos0);
    __sincosf(theta1, &sin1, &cos1);

    out_low->x = in_low.x * cos0 - in_high.x * sin0;
    out_low->y = in_low.y * cos1 - in_high.y * sin1;
    out_high->x = in_high.x * cos0 + in_low.x * sin0;
    out_high->y = in_high.y * cos1 + in_low.y * sin1;
}

__device__ __forceinline__ float encoder_fa_fast_exp(float x) {
    float y;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
#else
    return __expf(x);
#endif
}

__global__ __launch_bounds__(ENCODER_FA_Q32_THREADS, 4)
void encoder_flash_attention_tile_kernel_pv_mma_q32_k32(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)seq_len;
    (void)hidden_dim;
    (void)q_stride;
    (void)k_stride;
    (void)v_stride;
    (void)num_heads;
#else
    __shared__ __align__(16) __nv_bfloat16 sh_q[ENCODER_FA_Q32_TILE][ENCODER_FA_HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 sh_k[ENCODER_FA_Q32_K32_K_TILE][ENCODER_FA_Q32_K32_K_STRIDE];
    __shared__ __align__(16) __nv_bfloat162 sh_v[ENCODER_FA_Q32_K32_K_TILE][ENCODER_FA_HEAD_PAIRS + ENCODER_FA_Q32_K32_V_PAD];
    __shared__ __align__(16) __nv_bfloat16 sh_prob[ENCODER_FA_Q32_TILE][ENCODER_FA_Q32_K32_PROB_STRIDE];
    __shared__ float sh_m[ENCODER_FA_Q32_TILE];
    __shared__ float sh_l[ENCODER_FA_Q32_TILE];
    __shared__ float sh_alpha[ENCODER_FA_Q32_TILE];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int q_start = blockIdx.x * ENCODER_FA_Q32_TILE;
    int head = blockIdx.y;
    const int v_ld = (ENCODER_FA_HEAD_PAIRS + ENCODER_FA_Q32_K32_V_PAD) * 2;

    if (head >= num_heads) {
        return;
    }

    for (int idx = threadIdx.x; idx < ENCODER_FA_Q32_TILE * ENCODER_FA_HEAD_DIM; idx += blockDim.x) {
        int tq = idx / ENCODER_FA_HEAD_DIM;
        int td = idx % ENCODER_FA_HEAD_DIM;
        int q_row = q_start + tq;
        __nv_bfloat16 q_val = __float2bfloat16(0.0f);
        if (q_row < seq_len) {
            float qf = __bfloat162float(q[(size_t)q_row * q_stride + head * ENCODER_FA_HEAD_DIM + td]);
            q_val = __float2bfloat16_rn(qf * 0.125f);
        }
        sh_q[tq][td] = q_val;
    }
    if (threadIdx.x < ENCODER_FA_Q32_TILE) {
        sh_m[threadIdx.x] = -INFINITY;
        sh_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                           ENCODER_FA_Q32_ROW_BLOCK,
                           ENCODER_FA_PV_MMA_SCORE_TILE,
                           ENCODER_QK_MMA_TILE,
                           __nv_bfloat16,
                           nvcuda::wmma::row_major> q_frag[ENCODER_FA_HEAD_DIM / ENCODER_QK_MMA_TILE];
    if (warp < 2) {
        int row_base = warp * ENCODER_FA_Q32_ROW_BLOCK;
#pragma unroll
        for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
            nvcuda::wmma::load_matrix_sync(q_frag[d / ENCODER_QK_MMA_TILE],
                                           &sh_q[row_base][d],
                                           ENCODER_FA_HEAD_DIM);
        }
    }

    int pv_row_base = (warp / ENCODER_FA_PV_MMA_N_WARPS) * ENCODER_FA_Q32_ROW_BLOCK;
    int pv_n_col = (warp & (ENCODER_FA_PV_MMA_N_WARPS - 1)) * ENCODER_FA_PV_MMA_N_TILE;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                           ENCODER_FA_Q32_ROW_BLOCK,
                           ENCODER_FA_PV_MMA_N_TILE,
                           ENCODER_FA_PV_MMA_SCORE_TILE,
                           float> pv_frag;
    nvcuda::wmma::fill_fragment(pv_frag, 0.0f);

    for (int k_start = 0; k_start < seq_len; k_start += ENCODER_FA_Q32_K32_K_TILE) {
        int tile_len = seq_len - k_start;
        if (tile_len > ENCODER_FA_Q32_K32_K_TILE) {
            tile_len = ENCODER_FA_Q32_K32_K_TILE;
        }
        int tile_elems = tile_len * ENCODER_FA_HEAD_OCTETS;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int tk = idx / ENCODER_FA_HEAD_OCTETS;
            int td = idx % ENCODER_FA_HEAD_OCTETS;
            int k_row = k_start + tk;
            int bf16_col = td * 8;
            int pair_col = td * 4;
            const __nv_bfloat16 *k_ptr = k + (size_t)k_row * k_stride + head * ENCODER_FA_HEAD_DIM;
            const __nv_bfloat16 *v_ptr = v + (size_t)k_row * v_stride + head * ENCODER_FA_HEAD_DIM;
            const uint4 *k_ptr_u4 = reinterpret_cast<const uint4 *>(k_ptr);
            const uint4 *v_ptr_u4 = reinterpret_cast<const uint4 *>(v_ptr);
            __nv_bfloat16 *sh_k_dst = &sh_k[tk][bf16_col];
            __nv_bfloat162 *sh_v_dst = &sh_v[tk][pair_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_k_dst)), "l"(k_ptr_u4 + td));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_v_dst)), "l"(v_ptr_u4 + td));
        }
        asm volatile("cp.async.commit_group;\n" : :);
        asm volatile("cp.async.wait_group 0;\n" : :);
        __syncthreads();

        if (warp < 2) {
            int row_base = warp * ENCODER_FA_Q32_ROW_BLOCK;
            int lane_group = lane >> 2;
            unsigned row_group_mask = 0xFu << (lane_group * 4);
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   float> c_frag[ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE];
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
                nvcuda::wmma::fill_fragment(c_frag[sub_idx], 0.0f);
                if (sub >= tile_len) {
                    continue;
                }
                for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
                    nvcuda::wmma::load_matrix_sync(b_frag, &sh_k[sub][d], ENCODER_FA_Q32_K32_K_STRIDE);
                    nvcuda::wmma::mma_sync(c_frag[sub_idx], q_frag[d / ENCODER_QK_MMA_TILE], b_frag, c_frag[sub_idx]);
                }
            }

            float row_max_low = -INFINITY;
            float row_max_high = -INFINITY;
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float score = (tk < tile_len) ? c_frag[sub_idx].x[frag_idx] : -INFINITY;
                    if (local_row < 8) {
                        row_max_low = fmaxf(row_max_low, score);
                    } else {
                        row_max_high = fmaxf(row_max_high, score);
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                row_max_low = fmaxf(row_max_low, __shfl_xor_sync(row_group_mask, row_max_low, offset));
                row_max_high = fmaxf(row_max_high, __shfl_xor_sync(row_group_mask, row_max_high, offset));
            }

            float old_m_low = sh_m[row_base + lane_group];
            float old_l_low = sh_l[row_base + lane_group];
            float old_m_high = sh_m[row_base + lane_group + 8];
            float old_l_high = sh_l[row_base + lane_group + 8];
            float new_m_low = fmaxf(old_m_low, row_max_low);
            float new_m_high = fmaxf(old_m_high, row_max_high);
            float p_sum_low = 0.0f;
            float p_sum_high = 0.0f;

#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float p = 0.0f;
                    if (tk < tile_len) {
                        float new_m = (local_row < 8) ? new_m_low : new_m_high;
                        p = encoder_fa_fast_exp(c_frag[sub_idx].x[frag_idx] - new_m);
                    }
                    sh_prob[row_base + local_row][tk] = __float2bfloat16_rn(p);
                    if (local_row < 8) {
                        p_sum_low += p;
                    } else {
                        p_sum_high += p;
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                p_sum_low += __shfl_xor_sync(row_group_mask, p_sum_low, offset);
                p_sum_high += __shfl_xor_sync(row_group_mask, p_sum_high, offset);
            }

            if ((lane & 3) == 0) {
                float alpha_low = encoder_fa_fast_exp(old_m_low - new_m_low);
                float alpha_high = encoder_fa_fast_exp(old_m_high - new_m_high);
                sh_m[row_base + lane_group] = new_m_low;
                sh_l[row_base + lane_group] = old_l_low * alpha_low + p_sum_low;
                sh_alpha[row_base + lane_group] = alpha_low;
                sh_m[row_base + lane_group + 8] = new_m_high;
                sh_l[row_base + lane_group + 8] = old_l_high * alpha_high + p_sum_high;
                sh_alpha[row_base + lane_group + 8] = alpha_high;
            }
        }
        __syncthreads();

        if (warp < ENCODER_FA_Q32_PV_WARPS) {
            const __nv_bfloat16 *sh_v_bf16 = reinterpret_cast<const __nv_bfloat16 *>(&sh_v[0][0]);
            if (k_start != 0) {
                for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    pv_frag.x[frag_idx] *= sh_alpha[pv_row_base + local_row];
                }
            }
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_N_TILE,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> p_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_N_TILE,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> v_frag;
            for (int sub = 0; sub < ENCODER_FA_Q32_K32_K_TILE; sub += ENCODER_FA_PV_MMA_SCORE_TILE) {
                nvcuda::wmma::load_matrix_sync(p_frag, &sh_prob[pv_row_base][sub], ENCODER_FA_Q32_K32_PROB_STRIDE);
                nvcuda::wmma::load_matrix_sync(v_frag, &sh_v_bf16[sub * v_ld + pv_n_col], v_ld);
                nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < ENCODER_FA_Q32_TILE) {
        sh_alpha[threadIdx.x] = __frcp_rn(sh_l[threadIdx.x]);
    }
    __syncthreads();

    if (warp < ENCODER_FA_Q32_PV_WARPS) {
        for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
            int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
            int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
            int row = pv_row_base + local_row;
            int col = pv_n_col + local_col;
            int q_idx = q_start + row;
            if (q_idx < seq_len) {
                out[(size_t)q_idx * hidden_dim + head * ENCODER_FA_HEAD_DIM + col] =
                    __float2bfloat16_rn(pv_frag.x[frag_idx] * sh_alpha[row]);
            }
        }
    }
#endif
}

__global__ __launch_bounds__(ENCODER_FA_FA2_THREADS, 1)
void encoder_flash_attention_tile_kernel_fa2(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
    (void)q;
    (void)k;
    (void)v;
    (void)out;
    (void)seq_len;
    (void)hidden_dim;
    (void)q_stride;
    (void)k_stride;
    (void)v_stride;
    (void)num_heads;
#else
    __shared__ __align__(16) __nv_bfloat16 sh_q[ENCODER_FA_FA2_Q_TILE][ENCODER_FA_HEAD_DIM];
    __shared__ __align__(16) union {
        __nv_bfloat16 k[ENCODER_FA_FA2_K_TILE][ENCODER_FA_FA2_K_STRIDE];
        __nv_bfloat16 prob[ENCODER_FA_FA2_Q_TILE][ENCODER_FA_FA2_PROB_STRIDE];
    } sh_k_prob;
    __shared__ __align__(16) __nv_bfloat162 sh_v[ENCODER_FA_FA2_K_TILE][ENCODER_FA_HEAD_PAIRS + ENCODER_FA_FA2_V_PAD];
    __shared__ float sh_m[ENCODER_FA_FA2_Q_TILE];
    __shared__ float sh_l[ENCODER_FA_FA2_Q_TILE];
    __shared__ float sh_alpha[ENCODER_FA_FA2_Q_TILE];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int q_start = blockIdx.x * ENCODER_FA_FA2_Q_TILE;
    int head = blockIdx.y;
    const int v_ld = (ENCODER_FA_HEAD_PAIRS + ENCODER_FA_FA2_V_PAD) * 2;

    if (head >= num_heads) {
        return;
    }

    for (int idx = threadIdx.x; idx < ENCODER_FA_FA2_Q_TILE * ENCODER_FA_HEAD_DIM; idx += blockDim.x) {
        int tq = idx / ENCODER_FA_HEAD_DIM;
        int td = idx % ENCODER_FA_HEAD_DIM;
        int q_row = q_start + tq;
        __nv_bfloat16 q_val = __float2bfloat16(0.0f);
        if (q_row < seq_len) {
            float qf = __bfloat162float(q[(size_t)q_row * q_stride + head * ENCODER_FA_HEAD_DIM + td]);
            q_val = __float2bfloat16_rn(qf * 0.125f);
        }
        sh_q[tq][td] = q_val;
    }
    if (threadIdx.x < ENCODER_FA_FA2_Q_TILE) {
        sh_m[threadIdx.x] = -INFINITY;
        sh_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                           ENCODER_FA_FA2_ROW_BLOCK,
                           ENCODER_FA_FA2_SCORE_TILE,
                           ENCODER_QK_MMA_TILE,
                           __nv_bfloat16,
                           nvcuda::wmma::row_major> q_frag[ENCODER_FA_HEAD_DIM / ENCODER_QK_MMA_TILE];
    if (warp < ENCODER_FA_FA2_QK_WARPS) {
        int row_base = warp * ENCODER_FA_FA2_ROW_BLOCK;
#pragma unroll
        for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
            nvcuda::wmma::load_matrix_sync(q_frag[d / ENCODER_QK_MMA_TILE],
                                           &sh_q[row_base][d],
                                           ENCODER_FA_HEAD_DIM);
        }
    }

    int pv_row_base = (warp / 2) * ENCODER_FA_FA2_ROW_BLOCK;
    int pv_n_col = (warp & 1) * ENCODER_FA_FA2_N_TILE;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                           ENCODER_FA_FA2_ROW_BLOCK,
                           ENCODER_FA_FA2_SCORE_TILE,
                           ENCODER_FA_FA2_SCORE_TILE,
                           float> pv_frag[ENCODER_FA_FA2_N_SUBS];
#pragma unroll
    for (int n_sub = 0; n_sub < ENCODER_FA_FA2_N_SUBS; n_sub++) {
        nvcuda::wmma::fill_fragment(pv_frag[n_sub], 0.0f);
    }

    for (int k_start = 0; k_start < seq_len; k_start += ENCODER_FA_FA2_K_TILE) {
        int tile_len = seq_len - k_start;
        if (tile_len > ENCODER_FA_FA2_K_TILE) {
            tile_len = ENCODER_FA_FA2_K_TILE;
        }
        int tile_elems = tile_len * ENCODER_FA_HEAD_OCTETS;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int tk = idx / ENCODER_FA_HEAD_OCTETS;
            int td = idx % ENCODER_FA_HEAD_OCTETS;
            int k_row = k_start + tk;
            int bf16_col = td * 8;
            int pair_col = td * 4;
            const __nv_bfloat16 *k_ptr = k + (size_t)k_row * k_stride + head * ENCODER_FA_HEAD_DIM;
            const __nv_bfloat16 *v_ptr = v + (size_t)k_row * v_stride + head * ENCODER_FA_HEAD_DIM;
            const uint4 *k_ptr_u4 = reinterpret_cast<const uint4 *>(k_ptr);
            const uint4 *v_ptr_u4 = reinterpret_cast<const uint4 *>(v_ptr);
            __nv_bfloat16 *sh_k_dst = &sh_k_prob.k[tk][bf16_col];
            __nv_bfloat162 *sh_v_dst = &sh_v[tk][pair_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_k_dst)), "l"(k_ptr_u4 + td));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_v_dst)), "l"(v_ptr_u4 + td));
        }
        asm volatile("cp.async.commit_group;\n" : :);
        asm volatile("cp.async.wait_group 0;\n" : :);
        __syncthreads();

        if (warp < ENCODER_FA_FA2_QK_WARPS) {
            int row_base = warp * ENCODER_FA_FA2_ROW_BLOCK;
            int lane_group = lane >> 2;
            unsigned row_group_mask = 0xFu << (lane_group * 4);
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_FA2_ROW_BLOCK,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                   ENCODER_FA_FA2_ROW_BLOCK,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   float> c_frag[ENCODER_FA_FA2_K_TILE / ENCODER_FA_FA2_SCORE_TILE];
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_FA2_K_TILE / ENCODER_FA_FA2_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_FA2_SCORE_TILE;
                nvcuda::wmma::fill_fragment(c_frag[sub_idx], 0.0f);
                if (sub >= tile_len) {
                    continue;
                }
#pragma unroll
                for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
                    nvcuda::wmma::load_matrix_sync(b_frag, &sh_k_prob.k[sub][d], ENCODER_FA_FA2_K_STRIDE);
                    nvcuda::wmma::mma_sync(c_frag[sub_idx], q_frag[d / ENCODER_QK_MMA_TILE], b_frag, c_frag[sub_idx]);
                }
            }

            float row_max_low = -INFINITY;
            float row_max_high = -INFINITY;
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_FA2_K_TILE / ENCODER_FA_FA2_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_FA2_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float score = (tk < tile_len) ? c_frag[sub_idx].x[frag_idx] : -INFINITY;
                    if (local_row < 8) {
                        row_max_low = fmaxf(row_max_low, score);
                    } else {
                        row_max_high = fmaxf(row_max_high, score);
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                row_max_low = fmaxf(row_max_low, __shfl_xor_sync(row_group_mask, row_max_low, offset));
                row_max_high = fmaxf(row_max_high, __shfl_xor_sync(row_group_mask, row_max_high, offset));
            }

            float old_m_low = sh_m[row_base + lane_group];
            float old_l_low = sh_l[row_base + lane_group];
            float old_m_high = sh_m[row_base + lane_group + 8];
            float old_l_high = sh_l[row_base + lane_group + 8];
            float new_m_low = fmaxf(old_m_low, row_max_low);
            float new_m_high = fmaxf(old_m_high, row_max_high);
            float p_sum_low = 0.0f;
            float p_sum_high = 0.0f;

#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_FA2_K_TILE / ENCODER_FA_FA2_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_FA2_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float p = 0.0f;
                    if (tk < tile_len) {
                        float new_m = (local_row < 8) ? new_m_low : new_m_high;
                        p = encoder_fa_fast_exp(c_frag[sub_idx].x[frag_idx] - new_m);
                    }
                    sh_k_prob.prob[row_base + local_row][tk] = __float2bfloat16_rn(p);
                    if (local_row < 8) {
                        p_sum_low += p;
                    } else {
                        p_sum_high += p;
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                p_sum_low += __shfl_xor_sync(row_group_mask, p_sum_low, offset);
                p_sum_high += __shfl_xor_sync(row_group_mask, p_sum_high, offset);
            }

            if ((lane & 3) == 0) {
                float alpha_low = encoder_fa_fast_exp(old_m_low - new_m_low);
                float alpha_high = encoder_fa_fast_exp(old_m_high - new_m_high);
                sh_m[row_base + lane_group] = new_m_low;
                sh_l[row_base + lane_group] = old_l_low * alpha_low + p_sum_low;
                sh_alpha[row_base + lane_group] = alpha_low;
                sh_m[row_base + lane_group + 8] = new_m_high;
                sh_l[row_base + lane_group + 8] = old_l_high * alpha_high + p_sum_high;
                sh_alpha[row_base + lane_group + 8] = alpha_high;
            }
        }
        __syncthreads();

        if (warp < ENCODER_FA_FA2_PV_WARPS) {
            const __nv_bfloat16 *sh_v_bf16 = reinterpret_cast<const __nv_bfloat16 *>(&sh_v[0][0]);
            if (k_start != 0) {
#pragma unroll
                for (int n_sub = 0; n_sub < ENCODER_FA_FA2_N_SUBS; n_sub++) {
                    for (int frag_idx = 0; frag_idx < pv_frag[n_sub].num_elements; frag_idx++) {
                        int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                        pv_frag[n_sub].x[frag_idx] *= sh_alpha[pv_row_base + local_row];
                    }
                }
            }
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                   ENCODER_FA_FA2_ROW_BLOCK,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> p_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_FA2_ROW_BLOCK,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   ENCODER_FA_FA2_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> v_frag;
            for (int sub = 0; sub < ENCODER_FA_FA2_K_TILE; sub += ENCODER_FA_FA2_SCORE_TILE) {
                nvcuda::wmma::load_matrix_sync(p_frag, &sh_k_prob.prob[pv_row_base][sub], ENCODER_FA_FA2_PROB_STRIDE);
#pragma unroll
                for (int n_sub = 0; n_sub < ENCODER_FA_FA2_N_SUBS; n_sub++) {
                    nvcuda::wmma::load_matrix_sync(v_frag, &sh_v_bf16[sub * v_ld + pv_n_col + n_sub * ENCODER_FA_FA2_SCORE_TILE], v_ld);
                    nvcuda::wmma::mma_sync(pv_frag[n_sub], p_frag, v_frag, pv_frag[n_sub]);
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < ENCODER_FA_FA2_Q_TILE) {
        sh_alpha[threadIdx.x] = __frcp_rn(sh_l[threadIdx.x]);
    }
    __syncthreads();

    if (warp < ENCODER_FA_FA2_PV_WARPS) {
#pragma unroll
        for (int n_sub = 0; n_sub < ENCODER_FA_FA2_N_SUBS; n_sub++) {
            for (int frag_idx = 0; frag_idx < pv_frag[n_sub].num_elements; frag_idx++) {
                int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                int row = pv_row_base + local_row;
                int col = pv_n_col + n_sub * ENCODER_FA_FA2_SCORE_TILE + local_col;
                int q_idx = q_start + row;
                if (q_idx < seq_len) {
                    out[(size_t)q_idx * hidden_dim + head * ENCODER_FA_HEAD_DIM + col] =
                        __float2bfloat16_rn(pv_frag[n_sub].x[frag_idx] * sh_alpha[row]);
                }
            }
        }
    }
#endif
}

__global__ __launch_bounds__(ENCODER_FA_Q32_THREADS, 4)
void encoder_flash_attention_tile_kernel_pv_mma_q32_k32_batched(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *out,
    const int *seq_lens,
    const int *seq_offsets,
    int num_seqs,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
    (void)q;(void)k;(void)v;(void)out;
    (void)seq_lens;(void)seq_offsets;(void)num_seqs;
    (void)hidden_dim;(void)q_stride;(void)k_stride;(void)v_stride;(void)num_heads;
#else
    int seq_idx = blockIdx.z;
    if (seq_idx >= num_seqs) return;
    int seq_len = seq_lens[seq_idx];
    int seq_offset = seq_offsets[seq_idx];

    const __nv_bfloat16 *q_seq = q + (size_t)seq_offset * q_stride;
    const __nv_bfloat16 *k_seq = k + (size_t)seq_offset * k_stride;
    const __nv_bfloat16 *v_seq = v + (size_t)seq_offset * v_stride;
    __nv_bfloat16 *out_seq = out + (size_t)seq_offset * hidden_dim;

    __shared__ __align__(16) __nv_bfloat16 sh_q[ENCODER_FA_Q32_TILE][ENCODER_FA_HEAD_DIM];
    __shared__ __align__(16) __nv_bfloat16 sh_k[ENCODER_FA_Q32_K32_K_TILE][ENCODER_FA_Q32_K32_K_STRIDE];
    __shared__ __align__(16) __nv_bfloat162 sh_v[ENCODER_FA_Q32_K32_K_TILE][ENCODER_FA_HEAD_PAIRS + ENCODER_FA_Q32_K32_V_PAD];
    __shared__ __align__(16) __nv_bfloat16 sh_prob[ENCODER_FA_Q32_TILE][ENCODER_FA_Q32_K32_PROB_STRIDE];
    __shared__ float sh_m[ENCODER_FA_Q32_TILE];
    __shared__ float sh_l[ENCODER_FA_Q32_TILE];
    __shared__ float sh_alpha[ENCODER_FA_Q32_TILE];

    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x / WARP_SIZE;
    int q_start = blockIdx.x * ENCODER_FA_Q32_TILE;
    int head = blockIdx.y;
    const int v_ld = (ENCODER_FA_HEAD_PAIRS + ENCODER_FA_Q32_K32_V_PAD) * 2;

    if (head >= num_heads) {
        return;
    }
    if (q_start >= seq_len) return;

    for (int idx = threadIdx.x; idx < ENCODER_FA_Q32_TILE * ENCODER_FA_HEAD_DIM; idx += blockDim.x) {
        int tq = idx / ENCODER_FA_HEAD_DIM;
        int td = idx % ENCODER_FA_HEAD_DIM;
        int q_row = q_start + tq;
        __nv_bfloat16 q_val = __float2bfloat16(0.0f);
        if (q_row < seq_len) {
            float qf = __bfloat162float(q_seq[(size_t)q_row * q_stride + head * ENCODER_FA_HEAD_DIM + td]);
            q_val = __float2bfloat16_rn(qf * 0.125f);
        }
        sh_q[tq][td] = q_val;
    }
    if (threadIdx.x < ENCODER_FA_Q32_TILE) {
        sh_m[threadIdx.x] = -INFINITY;
        sh_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                           ENCODER_FA_Q32_ROW_BLOCK,
                           ENCODER_FA_PV_MMA_SCORE_TILE,
                           ENCODER_QK_MMA_TILE,
                           __nv_bfloat16,
                           nvcuda::wmma::row_major> q_frag[ENCODER_FA_HEAD_DIM / ENCODER_QK_MMA_TILE];
    if (warp < 2) {
        int row_base = warp * ENCODER_FA_Q32_ROW_BLOCK;
#pragma unroll
        for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
            nvcuda::wmma::load_matrix_sync(q_frag[d / ENCODER_QK_MMA_TILE],
                                           &sh_q[row_base][d],
                                           ENCODER_FA_HEAD_DIM);
        }
    }

    int pv_row_base = (warp / ENCODER_FA_PV_MMA_N_WARPS) * ENCODER_FA_Q32_ROW_BLOCK;
    int pv_n_col = (warp & (ENCODER_FA_PV_MMA_N_WARPS - 1)) * ENCODER_FA_PV_MMA_N_TILE;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                           ENCODER_FA_Q32_ROW_BLOCK,
                           ENCODER_FA_PV_MMA_N_TILE,
                           ENCODER_FA_PV_MMA_SCORE_TILE,
                           float> pv_frag;
    nvcuda::wmma::fill_fragment(pv_frag, 0.0f);

    for (int k_start = 0; k_start < seq_len; k_start += ENCODER_FA_Q32_K32_K_TILE) {
        int tile_len = seq_len - k_start;
        if (tile_len > ENCODER_FA_Q32_K32_K_TILE) {
            tile_len = ENCODER_FA_Q32_K32_K_TILE;
        }
        int tile_elems = tile_len * ENCODER_FA_HEAD_OCTETS;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            int tk = idx / ENCODER_FA_HEAD_OCTETS;
            int td = idx % ENCODER_FA_HEAD_OCTETS;
            int k_row = k_start + tk;
            int bf16_col = td * 8;
            int pair_col = td * 4;
            const __nv_bfloat16 *k_ptr = k_seq + (size_t)k_row * k_stride + head * ENCODER_FA_HEAD_DIM;
            const __nv_bfloat16 *v_ptr = v_seq + (size_t)k_row * v_stride + head * ENCODER_FA_HEAD_DIM;
            const uint4 *k_ptr_u4 = reinterpret_cast<const uint4 *>(k_ptr);
            const uint4 *v_ptr_u4 = reinterpret_cast<const uint4 *>(v_ptr);
            __nv_bfloat16 *sh_k_dst = &sh_k[tk][bf16_col];
            __nv_bfloat162 *sh_v_dst = &sh_v[tk][pair_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_k_dst)), "l"(k_ptr_u4 + td));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned)__cvta_generic_to_shared(sh_v_dst)), "l"(v_ptr_u4 + td));
        }
        asm volatile("cp.async.commit_group;\n" : :);
        asm volatile("cp.async.wait_group 0;\n" : :);
        __syncthreads();

        if (warp < 2) {
            int row_base = warp * ENCODER_FA_Q32_ROW_BLOCK;
            int lane_group = lane >> 2;
            unsigned row_group_mask = 0xFu << (lane_group * 4);
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   ENCODER_QK_MMA_TILE,
                                   float> c_frag[ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE];
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
                nvcuda::wmma::fill_fragment(c_frag[sub_idx], 0.0f);
                if (sub >= tile_len) {
                    continue;
                }
                for (int d = 0; d < ENCODER_FA_HEAD_DIM; d += ENCODER_QK_MMA_TILE) {
                    nvcuda::wmma::load_matrix_sync(b_frag, &sh_k[sub][d], ENCODER_FA_Q32_K32_K_STRIDE);
                    nvcuda::wmma::mma_sync(c_frag[sub_idx], q_frag[d / ENCODER_QK_MMA_TILE], b_frag, c_frag[sub_idx]);
                }
            }

            float row_max_low = -INFINITY;
            float row_max_high = -INFINITY;
#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float score = (tk < tile_len) ? c_frag[sub_idx].x[frag_idx] : -INFINITY;
                    if (local_row < 8) {
                        row_max_low = fmaxf(row_max_low, score);
                    } else {
                        row_max_high = fmaxf(row_max_high, score);
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                row_max_low = fmaxf(row_max_low, __shfl_xor_sync(row_group_mask, row_max_low, offset));
                row_max_high = fmaxf(row_max_high, __shfl_xor_sync(row_group_mask, row_max_high, offset));
            }

            float old_m_low = sh_m[row_base + lane_group];
            float old_l_low = sh_l[row_base + lane_group];
            float old_m_high = sh_m[row_base + lane_group + 8];
            float old_l_high = sh_l[row_base + lane_group + 8];
            float new_m_low = fmaxf(old_m_low, row_max_low);
            float new_m_high = fmaxf(old_m_high, row_max_high);
            float p_sum_low = 0.0f;
            float p_sum_high = 0.0f;

#pragma unroll
            for (int sub_idx = 0; sub_idx < ENCODER_FA_Q32_K32_K_TILE / ENCODER_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                int sub = sub_idx * ENCODER_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    int tk = sub + local_col;
                    float p = 0.0f;
                    if (tk < tile_len) {
                        float new_m = (local_row < 8) ? new_m_low : new_m_high;
                        p = encoder_fa_fast_exp(c_frag[sub_idx].x[frag_idx] - new_m);
                    }
                    sh_prob[row_base + local_row][tk] = __float2bfloat16_rn(p);
                    if (local_row < 8) {
                        p_sum_low += p;
                    } else {
                        p_sum_high += p;
                    }
                }
            }

#pragma unroll
            for (int offset = 2; offset > 0; offset >>= 1) {
                p_sum_low += __shfl_xor_sync(row_group_mask, p_sum_low, offset);
                p_sum_high += __shfl_xor_sync(row_group_mask, p_sum_high, offset);
            }

            if ((lane & 3) == 0) {
                float alpha_low = encoder_fa_fast_exp(old_m_low - new_m_low);
                float alpha_high = encoder_fa_fast_exp(old_m_high - new_m_high);
                sh_m[row_base + lane_group] = new_m_low;
                sh_l[row_base + lane_group] = old_l_low * alpha_low + p_sum_low;
                sh_alpha[row_base + lane_group] = alpha_low;
                sh_m[row_base + lane_group + 8] = new_m_high;
                sh_l[row_base + lane_group + 8] = old_l_high * alpha_high + p_sum_high;
                sh_alpha[row_base + lane_group + 8] = alpha_high;
            }
        }
        __syncthreads();

        if (warp < ENCODER_FA_Q32_PV_WARPS) {
            const __nv_bfloat16 *sh_v_bf16 = reinterpret_cast<const __nv_bfloat16 *>(&sh_v[0][0]);
            if (k_start != 0) {
                for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
                    int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    pv_frag.x[frag_idx] *= sh_alpha[pv_row_base + local_row];
                }
            }
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_N_TILE,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> p_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   ENCODER_FA_Q32_ROW_BLOCK,
                                   ENCODER_FA_PV_MMA_N_TILE,
                                   ENCODER_FA_PV_MMA_SCORE_TILE,
                                   __nv_bfloat16,
                                   nvcuda::wmma::row_major> v_mma_frag;
            for (int sub = 0; sub < ENCODER_FA_Q32_K32_K_TILE; sub += ENCODER_FA_PV_MMA_SCORE_TILE) {
                nvcuda::wmma::load_matrix_sync(p_frag, &sh_prob[pv_row_base][sub], ENCODER_FA_Q32_K32_PROB_STRIDE);
                nvcuda::wmma::load_matrix_sync(v_mma_frag, &sh_v_bf16[sub * v_ld + pv_n_col], v_ld);
                nvcuda::wmma::mma_sync(pv_frag, p_frag, v_mma_frag, pv_frag);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < ENCODER_FA_Q32_TILE) {
        sh_alpha[threadIdx.x] = __frcp_rn(sh_l[threadIdx.x]);
    }
    __syncthreads();

    if (warp < ENCODER_FA_Q32_PV_WARPS) {
        for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
            int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
            int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
            int row = pv_row_base + local_row;
            int col = pv_n_col + local_col;
            int q_idx = q_start + row;
            if (q_idx < seq_len) {
                out_seq[(size_t)q_idx * hidden_dim + head * ENCODER_FA_HEAD_DIM + col] =
                    __float2bfloat16_rn(pv_frag.x[frag_idx] * sh_alpha[row]);
            }
        }
    }
#endif
}

static void allocate_encoder_buffers(int row_capacity, int attn_seq_capacity) {
    size_t qkv_bytes;
    size_t qkv_fused_out_bytes = 0;
    size_t attn_matrix_bytes;
    size_t attn_out_bytes;
    size_t hidden_ffn_bytes;
    size_t softmax_bytes;
    size_t seq_meta_bytes;
    size_t row_pos_bytes;

    if (buffers_allocated &&
        encoder_row_capacity >= row_capacity &&
        encoder_attn_seq_capacity >= attn_seq_capacity) {
        return;
    }

    if (buffers_allocated) {
        free_encoder_buffers();
    }

    int enc_hidden = 1280;
    int enc_heads = 20;
    int enc_head_dim = 64;
    int enc_ffn = 5120;

    (void) enc_head_dim;

    qkv_bytes = (size_t)row_capacity * enc_hidden * sizeof(__nv_bfloat16);
    if (encoder_qkv_fused_enabled()) {
        qkv_fused_out_bytes = 3 * qkv_bytes;
    }
    attn_matrix_bytes = (size_t)enc_heads * attn_seq_capacity * attn_seq_capacity * sizeof(__nv_bfloat16);
    attn_out_bytes = (size_t)row_capacity * enc_hidden * sizeof(__nv_bfloat16);
    hidden_ffn_bytes = (size_t)row_capacity * enc_ffn * sizeof(__nv_bfloat16);
    softmax_bytes = (size_t)attn_seq_capacity * enc_heads * sizeof(float);
    seq_meta_bytes = (size_t)row_capacity * sizeof(int);
    row_pos_bytes = (size_t)row_capacity * sizeof(int);
    encoder_qkv_bytes = qkv_bytes;
    encoder_qkv_fused_out_bytes = qkv_fused_out_bytes;
    encoder_attn_matrix_bytes = attn_matrix_bytes;
    encoder_attn_out_bytes = attn_out_bytes;
    encoder_hidden_ffn_bytes = hidden_ffn_bytes;
    encoder_softmax_bytes = softmax_bytes;
    encoder_seq_meta_bytes = seq_meta_bytes;
    encoder_row_pos_bytes = row_pos_bytes;
    encoder_row_capacity = row_capacity;
    encoder_attn_seq_capacity = attn_seq_capacity;

    // Q, K, V buffers: [seq, hidden]
    CUDA_CHECK(cudaMalloc(&d_q_buf, qkv_bytes));
    encoder_note_alloc(qkv_bytes);
    CUDA_CHECK(cudaMalloc(&d_k_buf, qkv_bytes));
    encoder_note_alloc(qkv_bytes);
    CUDA_CHECK(cudaMalloc(&d_v_buf, qkv_bytes));
    encoder_note_alloc(qkv_bytes);
    if (qkv_fused_out_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_qkv_buf, qkv_fused_out_bytes));
        encoder_note_alloc(qkv_fused_out_bytes);
    }

    // QK^T and attention: [heads, seq, seq]
    CUDA_CHECK(cudaMalloc(&d_qk_buf, attn_matrix_bytes));
    encoder_note_alloc(attn_matrix_bytes);
    CUDA_CHECK(cudaMalloc(&d_attn_buf, attn_matrix_bytes));
    encoder_note_alloc(attn_matrix_bytes);

    // Attention output: [seq, hidden]
    CUDA_CHECK(cudaMalloc(&d_attn_out, attn_out_bytes));
    encoder_note_alloc(attn_out_bytes);

    // Hidden and residual buffers: [seq, hidden]
    CUDA_CHECK(cudaMalloc(&d_hidden_buf, hidden_ffn_bytes));
    encoder_note_alloc(hidden_ffn_bytes);

    CUDA_CHECK(cudaMalloc(&d_encoder_seq_lens, seq_meta_bytes));
    encoder_note_alloc(seq_meta_bytes);
    CUDA_CHECK(cudaMalloc(&d_encoder_seq_offsets, seq_meta_bytes));
    encoder_note_alloc(seq_meta_bytes);
    CUDA_CHECK(cudaMalloc(&d_encoder_row_pos, row_pos_bytes));
    encoder_note_alloc(row_pos_bytes);

    // Softmax temp buffers
    CUDA_CHECK(cudaMalloc(&d_softmax_max, softmax_bytes));
    encoder_note_alloc(softmax_bytes);
    CUDA_CHECK(cudaMalloc(&d_softmax_sum, softmax_bytes));
    encoder_note_alloc(softmax_bytes);
    buffers_allocated = true;
}

static void free_encoder_buffers(void) {
    if (!buffers_allocated) return;

    cudaFree(d_q_buf);
    encoder_note_free(encoder_qkv_bytes);
    cudaFree(d_k_buf);
    encoder_note_free(encoder_qkv_bytes);
    cudaFree(d_v_buf);
    encoder_note_free(encoder_qkv_bytes);
    if (d_qkv_buf != NULL) {
        cudaFree(d_qkv_buf);
        encoder_note_free(encoder_qkv_fused_out_bytes);
    }
    cudaFree(d_qk_buf);
    encoder_note_free(encoder_attn_matrix_bytes);
    cudaFree(d_attn_buf);
    encoder_note_free(encoder_attn_matrix_bytes);
    cudaFree(d_attn_out);
    encoder_note_free(encoder_attn_out_bytes);
    cudaFree(d_hidden_buf);
    encoder_note_free(encoder_hidden_ffn_bytes);
    cudaFree(d_encoder_seq_lens);
    encoder_note_free(encoder_seq_meta_bytes);
    cudaFree(d_encoder_seq_offsets);
    encoder_note_free(encoder_seq_meta_bytes);
    cudaFree(d_encoder_row_pos);
    encoder_note_free(encoder_row_pos_bytes);
    cudaFree(d_softmax_max);
    encoder_note_free(encoder_softmax_bytes);
    cudaFree(d_softmax_sum);
    encoder_note_free(encoder_softmax_bytes);
    d_q_buf = NULL;
    d_k_buf = NULL;
    d_v_buf = NULL;
    d_qkv_buf = NULL;
    d_qk_buf = NULL;
    d_attn_buf = NULL;
    d_attn_out = NULL;
    d_hidden_buf = NULL;
    d_encoder_seq_lens = NULL;
    d_encoder_seq_offsets = NULL;
    d_encoder_row_pos = NULL;
    d_softmax_max = NULL;
    d_softmax_sum = NULL;
    encoder_qkv_bytes = 0;
    encoder_qkv_fused_out_bytes = 0;
    encoder_attn_matrix_bytes = 0;
    encoder_attn_out_bytes = 0;
    encoder_hidden_ffn_bytes = 0;
    encoder_softmax_bytes = 0;
    encoder_seq_meta_bytes = 0;
    encoder_row_pos_bytes = 0;
    encoder_row_capacity = 0;
    encoder_attn_seq_capacity = 0;
    encoder_qkv_fused_cache_free();

    buffers_allocated = false;
}

static int encoder_attention_flash_pv_mma_q32_k32_launch(__nv_bfloat16 *q_slice,
                                                          __nv_bfloat16 *k_slice,
                                                          __nv_bfloat16 *v_slice,
                                                          __nv_bfloat16 *out_slice,
                                                         int seq_len,
                                                         int hidden_dim,
                                                         int q_stride,
                                                         int k_stride,
                                                         int v_stride,
                                                         int num_heads,
                                                         int head_dim) {
    dim3 block;
    dim3 grid;

    if (q_slice == NULL || k_slice == NULL || v_slice == NULL || out_slice == NULL) {
        return 0;
    }
    if (!encoder_qk_mma_supported()) {
        return 0;
    }
    if (head_dim != ENCODER_FA_HEAD_DIM || seq_len <= 0 || hidden_dim <= 0 ||
        q_stride < hidden_dim || k_stride < hidden_dim || v_stride < hidden_dim ||
        num_heads <= 0) {
        return 0;
    }

    block = dim3(ENCODER_FA_Q32_THREADS);
    grid = dim3((seq_len + ENCODER_FA_Q32_TILE - 1) / ENCODER_FA_Q32_TILE, num_heads);
    encoder_flash_attention_tile_kernel_pv_mma_q32_k32<<<grid, block>>>(
        q_slice, k_slice, v_slice, out_slice, seq_len, hidden_dim,
        q_stride, k_stride, v_stride, num_heads);
    CUDA_CHECK(cudaGetLastError());
    return 1;
}

static int encoder_attention_flash_fa2_launch(__nv_bfloat16 *q_slice,
                                              __nv_bfloat16 *k_slice,
                                              __nv_bfloat16 *v_slice,
                                              __nv_bfloat16 *out_slice,
                                              int seq_len,
                                              int hidden_dim,
                                              int q_stride,
                                              int k_stride,
                                              int v_stride,
                                              int num_heads,
                                              int head_dim) {
    dim3 block;
    dim3 grid;

    if (q_slice == NULL || k_slice == NULL || v_slice == NULL || out_slice == NULL) {
        return 0;
    }
    if (!encoder_qk_mma_supported()) {
        return 0;
    }
    if (head_dim != ENCODER_FA_HEAD_DIM || seq_len <= 0 || hidden_dim <= 0 ||
        q_stride < hidden_dim || k_stride < hidden_dim || v_stride < hidden_dim ||
        num_heads <= 0) {
        return 0;
    }

    block = dim3(ENCODER_FA_FA2_THREADS);
    grid = dim3((seq_len + ENCODER_FA_FA2_Q_TILE - 1) / ENCODER_FA_FA2_Q_TILE, num_heads);
    encoder_flash_attention_tile_kernel_fa2<<<grid, block>>>(
        q_slice, k_slice, v_slice, out_slice, seq_len, hidden_dim,
        q_stride, k_stride, v_stride, num_heads);
    CUDA_CHECK(cudaGetLastError());
    return 1;
}

static void encoder_attention_cublas(cublasHandle_t handle,
                                     __nv_bfloat16 *q_slice,
                                     __nv_bfloat16 *k_slice,
                                     __nv_bfloat16 *v_slice,
                                     __nv_bfloat16 *out_slice,
                                     int seq_len,
                                     int hidden_dim,
                                     int num_heads,
                                     int head_dim) {
    float alpha_f = 1.0f / sqrtf((float)head_dim);
    float beta_f = 0.0f;
    float one_f = 1.0f;

    cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, head_dim,
        &alpha_f,
        k_slice, CUDA_R_16BF, hidden_dim, head_dim,
        q_slice, CUDA_R_16BF, hidden_dim, head_dim,
        &beta_f,
        d_qk_buf, CUDA_R_16BF, seq_len, seq_len * seq_len,
        num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );

    softmax_inplace(d_qk_buf, num_heads * seq_len, seq_len);

    cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_len, seq_len,
        &one_f,
        v_slice, CUDA_R_16BF, hidden_dim, head_dim,
        d_qk_buf, CUDA_R_16BF, seq_len, seq_len * seq_len,
        &beta_f,
        out_slice, CUDA_R_16BF, hidden_dim, head_dim,
        num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
}

__global__ void encoder_rope_strided_kernel(__nv_bfloat16 *q,
                                            __nv_bfloat16 *k,
                                            int seq_len,
                                            int num_heads,
                                            int head_dim,
                                            int q_stride,
                                            int k_stride,
                                            float partial_factor) {
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (head_idx >= num_heads || seq_idx >= seq_len || dim_idx >= head_dim) {
        return;
    }

    int rot_dim = (int)(head_dim * partial_factor);
    int half = rot_dim / 2;
    if (dim_idx >= half) {
        return;
    }

    float inv_freq = 1.0f / powf(10000.0f, (2.0f * dim_idx) / (float)rot_dim);
    float theta = seq_idx * inv_freq;
    float cos_v = cosf(theta);
    float sin_v = sinf(theta);

    if (q != NULL) {
        __nv_bfloat16 *q_head = q + (size_t)seq_idx * q_stride + head_idx * head_dim;
        float q0 = __bfloat162float(q_head[dim_idx]);
        float q1 = __bfloat162float(q_head[dim_idx + half]);
        q_head[dim_idx] = __float2bfloat16(q0 * cos_v - q1 * sin_v);
        q_head[dim_idx + half] = __float2bfloat16(q1 * cos_v + q0 * sin_v);
    }

    if (k != NULL) {
        __nv_bfloat16 *k_head = k + (size_t)seq_idx * k_stride + head_idx * head_dim;
        float k0 = __bfloat162float(k_head[dim_idx]);
        float k1 = __bfloat162float(k_head[dim_idx + half]);
        k_head[dim_idx] = __float2bfloat16(k0 * cos_v - k1 * sin_v);
        k_head[dim_idx + half] = __float2bfloat16(k1 * cos_v + k0 * sin_v);
    }
}

static void encoder_rope_maybe_strided(__nv_bfloat16 *q,
                                       __nv_bfloat16 *k,
                                       int seq_len,
                                       int hidden_dim,
                                       int num_heads,
                                       int head_dim,
                                       int q_stride,
                                       int k_stride) {
    if (q_stride == hidden_dim && k_stride == hidden_dim) {
        rope_encoder(q, k, seq_len, num_heads, head_dim, 0.5f);
        return;
    }

    dim3 block(head_dim);
    dim3 grid(num_heads, seq_len);
    encoder_rope_strided_kernel<<<grid, block>>>(q, k, seq_len, num_heads, head_dim,
                                                 q_stride, k_stride, 0.5f);
    CUDA_CHECK(cudaGetLastError());
}

/* ---------------------------------------------------------------------------
 * Optimized encoder RoPE: pre-computed cos/sin table + vectorized kernel
 * Encoder: head_dim=64, partial_rotary_dim=32, half_rot=16
 * Standard RoPE base=10000, inv_freq[i] = 10000^(-i/16)
 * cos/sin table shape: [max_pos, 16] = max_pos * 16 * 4 bytes
 * For max_pos=1500: 96KB — tiny, fits in L2 cache
 * --------------------------------------------------------------------------- */

#define ENC_ROPE_HALF_DIM 16  /* head_dim(64) * partial_factor(0.5) / 2 */

static void encoder_rope_ensure_table(int max_pos) {
    if (max_pos <= d_enc_rope_table_cap) return;
    if (d_enc_rope_cos) { cudaFree(d_enc_rope_cos); }
    if (d_enc_rope_sin) { cudaFree(d_enc_rope_sin); }
    int size = max_pos * ENC_ROPE_HALF_DIM;
    CUDA_CHECK(cudaMalloc(&d_enc_rope_cos, (size_t)size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_enc_rope_sin, (size_t)size * sizeof(float)));
    /* Compute on host, upload */
    float *h_cos = (float *)malloc((size_t)size * sizeof(float));
    float *h_sin = (float *)malloc((size_t)size * sizeof(float));
    for (int pos = 0; pos < max_pos; pos++) {
        for (int d = 0; d < ENC_ROPE_HALF_DIM; d++) {
            float inv_freq = powf(10000.0f, -(2.0f * d) / 32.0f);
            float theta = pos * inv_freq;
            h_cos[pos * ENC_ROPE_HALF_DIM + d] = cosf(theta);
            h_sin[pos * ENC_ROPE_HALF_DIM + d] = sinf(theta);
        }
    }
    CUDA_CHECK(cudaMemcpy(d_enc_rope_cos, h_cos,
                           (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_enc_rope_sin, h_sin,
                           (size_t)size * sizeof(float), cudaMemcpyHostToDevice));
    free(h_cos);
    free(h_sin);
    d_enc_rope_table_cap = max_pos;
}

/* Optimized kernel: 128 threads/block, each thread handles 1 (head, row, dim_half) element
 * Processes Q and K simultaneously. Uses pre-computed cos/sin table.
 * Grid: (total_rows * num_heads, 1, 1)  Block: (128, 1, 1)
 * Each block processes one (row, head) pair across up to 16 half-dim elements */
__global__ void __launch_bounds__(128)
encoder_rope_fast_kernel(__nv_bfloat16 *q,
                         __nv_bfloat16 *k,
                         const float * __restrict__ cos_table,
                         const float * __restrict__ sin_table,
                         const int * __restrict__ row_pos,
                         int total_elements,  /* total_rows * num_heads */
                         int head_dim,
                         int rot_half,        /* 16 */
                         int q_stride,
                         int k_stride,
                         int num_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements * rot_half) return;

    int d = idx % rot_half;             /* dim half index 0..15 */
    int rh = idx / rot_half;            /* (row * num_heads + head) */
    int head = rh % num_heads;
    int row  = rh / num_heads;

    int pos = row_pos[row];
    float c = cos_table[pos * rot_half + d];
    float s = sin_table[pos * rot_half + d];

    if (q != NULL) {
        __nv_bfloat16 *q_ptr = q + (size_t)row * q_stride + head * head_dim;
        float q0 = __bfloat162float(q_ptr[d]);
        float q1 = __bfloat162float(q_ptr[d + rot_half]);
        q_ptr[d]              = __float2bfloat16(q0 * c - q1 * s);
        q_ptr[d + rot_half]   = __float2bfloat16(q1 * c + q0 * s);
    }

    if (k != NULL) {
        __nv_bfloat16 *k_ptr = k + (size_t)row * k_stride + head * head_dim;
        float k0 = __bfloat162float(k_ptr[d]);
        float k1 = __bfloat162float(k_ptr[d + rot_half]);
        k_ptr[d]              = __float2bfloat16(k0 * c - k1 * s);
        k_ptr[d + rot_half]   = __float2bfloat16(k1 * c + k0 * s);
    }
}

static void encoder_rope_fast(__nv_bfloat16 *q,
                              __nv_bfloat16 *k,
                              int total_rows,
                              int num_heads,
                              int head_dim,
                              int q_stride,
                              int k_stride) {
    if (total_rows <= 0) return;

    /* Ensure cos/sin table covers max possible position */
    /* row_pos values are 0..seq_len-1, max ~3000 for 30s audio */
    encoder_rope_ensure_table(3072);

    int rot_half = ENC_ROPE_HALF_DIM;  /* 16 */
    int total_elements = total_rows * num_heads;
    int total_work = total_elements * rot_half;
    int block = 128;
    int grid = (total_work + block - 1) / block;

    encoder_rope_fast_kernel<<<grid, block>>>(
        q, k, d_enc_rope_cos, d_enc_rope_sin, d_encoder_row_pos,
        total_elements, head_dim, rot_half,
        q_stride, k_stride, num_heads);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void encoder_fill_row_positions_kernel(const int *seq_lens,
                                                  const int *seq_offsets,
                                                  int *row_pos,
                                                  int num_seqs) {
    int seq_idx = blockIdx.x;
    int lane_idx = threadIdx.x;
    if (seq_idx >= num_seqs) {
        return;
    }

    int seq_len = seq_lens[seq_idx];
    int seq_offset = seq_offsets[seq_idx];
    for (int i = lane_idx; i < seq_len; i += blockDim.x) {
        row_pos[seq_offset + i] = i;
    }
}

static void encoder_prepare_row_positions(const int *seq_lens,
                                          const int *seq_offsets,
                                          int total_rows,
                                          int num_seqs) {
    if (num_seqs <= 0 || total_rows <= 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy(d_encoder_seq_lens, seq_lens,
                          (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_encoder_seq_offsets, seq_offsets,
                          (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice));
    encoder_fill_row_positions_kernel<<<num_seqs, 256>>>(d_encoder_seq_lens,
                                                         d_encoder_seq_offsets,
                                                         d_encoder_row_pos,
                                                         num_seqs);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void encoder_rope_batched_strided_kernel(__nv_bfloat16 *q,
                                                    __nv_bfloat16 *k,
                                                    const int *row_pos,
                                                    int total_rows,
                                                    int num_heads,
                                                    int head_dim,
                                                    int q_stride,
                                                    int k_stride,
                                                    float partial_factor) {
    int head_idx = blockIdx.x;
    int row_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (head_idx >= num_heads || row_idx >= total_rows || dim_idx >= head_dim) {
        return;
    }

    int rot_dim = (int)(head_dim * partial_factor);
    int half = rot_dim / 2;
    if (dim_idx >= half) {
        return;
    }

    float inv_freq = (dim_idx & 1) ?
        k_encoder_rope_inv_freq_odd[dim_idx >> 1] :
        k_encoder_rope_inv_freq_even[dim_idx >> 1];
    float theta = row_pos[row_idx] * inv_freq;
    float cos_v = cosf(theta);
    float sin_v = sinf(theta);

    if (q != NULL) {
        __nv_bfloat16 *q_head = q + (size_t)row_idx * q_stride + head_idx * head_dim;
        float q0 = __bfloat162float(q_head[dim_idx]);
        float q1 = __bfloat162float(q_head[dim_idx + half]);
        q_head[dim_idx] = __float2bfloat16(q0 * cos_v - q1 * sin_v);
        q_head[dim_idx + half] = __float2bfloat16(q1 * cos_v + q0 * sin_v);
    }

    if (k != NULL) {
        __nv_bfloat16 *k_head = k + (size_t)row_idx * k_stride + head_idx * head_dim;
        float k0 = __bfloat162float(k_head[dim_idx]);
        float k1 = __bfloat162float(k_head[dim_idx + half]);
        k_head[dim_idx] = __float2bfloat16(k0 * cos_v - k1 * sin_v);
        k_head[dim_idx + half] = __float2bfloat16(k1 * cos_v + k0 * sin_v);
    }
}

static void encoder_rope_batched_strided(__nv_bfloat16 *q,
                                         __nv_bfloat16 *k,
                                         int total_rows,
                                         int num_heads,
                                         int head_dim,
                                         int q_stride,
                                         int k_stride) {
    if (total_rows <= 0) {
        return;
    }

    dim3 block(head_dim);
    dim3 grid(num_heads, total_rows);
    encoder_rope_batched_strided_kernel<<<grid, block>>>(q, k, d_encoder_row_pos,
                                                         total_rows, num_heads, head_dim,
                                                         q_stride, k_stride, 0.5f);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void encoder_copy_strided_hidden_kernel(const __nv_bfloat16 *src,
                                                   __nv_bfloat16 *dst,
                                                   int rows,
                                                   int hidden_dim,
                                                   int src_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * hidden_dim;
    if (idx >= total) {
        return;
    }

    int row = idx / hidden_dim;
    int col = idx - row * hidden_dim;
    dst[idx] = src[(size_t)row * src_stride + col];
}

static void encoder_copy_strided_hidden(const __nv_bfloat16 *src,
                                        __nv_bfloat16 *dst,
                                        int rows,
                                        int hidden_dim,
                                        int src_stride) {
    int total = rows * hidden_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    encoder_copy_strided_hidden_kernel<<<grid, block>>>(src, dst, rows, hidden_dim, src_stride);
    CUDA_CHECK(cudaGetLastError());
}

static int encoder_qkv_fused_prepare_layer(const WeightStore *ws,
                                           int layer,
                                           int hidden_dim) {
    char name_buf[256];
    size_t weight_bytes;
    size_t bias_bytes;
    __nv_bfloat16 *fused_w = NULL;
    __nv_bfloat16 *fused_b = NULL;

    if (layer < 0 || layer >= ENCODER_MAX_LAYERS || hidden_dim <= 0) {
        return 0;
    }
    if (d_encoder_qkv_fused_w[layer] != NULL && d_encoder_qkv_fused_b[layer] != NULL) {
        return 1;
    }

    snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.q_proj.weight", layer);
    const Tensor *qw = ws_get_const(ws, name_buf);
    snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.q_proj.bias", layer);
    const Tensor *qb = ws_get_const(ws, name_buf);
    snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.k_proj.weight", layer);
    const Tensor *kw = ws_get_const(ws, name_buf);
    snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.v_proj.weight", layer);
    const Tensor *vw = ws_get_const(ws, name_buf);
    snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.v_proj.bias", layer);
    const Tensor *vb = ws_get_const(ws, name_buf);

    if (qw == NULL || kw == NULL || vw == NULL) {
        return 0;
    }

    weight_bytes = (size_t)hidden_dim * (size_t)hidden_dim * sizeof(__nv_bfloat16);
    bias_bytes = (size_t)hidden_dim * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&fused_w, 3 * weight_bytes));
    CUDA_CHECK(cudaMalloc(&fused_b, 3 * bias_bytes));

    CUDA_CHECK(cudaMemcpy(fused_w, qw->data, weight_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(fused_w + (size_t)hidden_dim * hidden_dim,
                          kw->data, weight_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(fused_w + (size_t)2 * hidden_dim * hidden_dim,
                          vw->data, weight_bytes, cudaMemcpyDeviceToDevice));

    if (qb != NULL) {
        CUDA_CHECK(cudaMemcpy(fused_b, qb->data, bias_bytes, cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemset(fused_b, 0, bias_bytes));
    }
    CUDA_CHECK(cudaMemset(fused_b + hidden_dim, 0, bias_bytes));
    if (vb != NULL) {
        CUDA_CHECK(cudaMemcpy(fused_b + 2 * hidden_dim, vb->data, bias_bytes, cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemset(fused_b + 2 * hidden_dim, 0, bias_bytes));
    }

    d_encoder_qkv_fused_w[layer] = fused_w;
    d_encoder_qkv_fused_b[layer] = fused_b;
    return 1;
}

static void encoder_attention_slice_batched(cublasHandle_t handle,
                                             __nv_bfloat16 *q_base,
                                             __nv_bfloat16 *k_base,
                                             __nv_bfloat16 *v_base,
                                             __nv_bfloat16 *out_base,
                                             const int *seq_lens,
                                             const int *seq_offsets,
                                             int num_seqs,
                                             int max_seq_len,
                                             int hidden_dim,
                                             int q_stride,
                                             int k_stride,
                                             int v_stride,
                                             int num_heads,
                                             int head_dim,
                                             int inputs_rope_applied) {
    if (num_seqs <= 0 || max_seq_len <= 0) return;
    if (!encoder_qk_mma_supported()) return;

    if (!inputs_rope_applied) {
        encoder_rope_fast(q_base, k_base,
                                     max_seq_len * num_seqs,
                                     num_heads, head_dim, q_stride, k_stride);
    }

    dim3 block(ENCODER_FA_Q32_THREADS);
    dim3 grid((max_seq_len + ENCODER_FA_Q32_TILE - 1) / ENCODER_FA_Q32_TILE,
              num_heads,
              num_seqs);
    encoder_flash_attention_tile_kernel_pv_mma_q32_k32_batched<<<grid, block>>>(
        q_base, k_base, v_base, out_base,
        seq_lens, seq_offsets, num_seqs, hidden_dim,
        q_stride, k_stride, v_stride, num_heads);
    CUDA_CHECK(cudaGetLastError());
}

static void encoder_attention_slice(cublasHandle_t handle,
                                    __nv_bfloat16 *q_slice,
                                    __nv_bfloat16 *k_slice,
                                    __nv_bfloat16 *v_slice,
                                    __nv_bfloat16 *out_slice,
                                    int seq_len,
                                    int hidden_dim,
                                    int q_stride,
                                    int k_stride,
                                    int v_stride,
                                    int num_heads,
                                    int head_dim,
                                    int inputs_rope_applied) {
    int can_pipeline;
    int used_fa = 0;
    int used_pipeline = 0;
    int want_pv_mma_q32_k32;
    int want_fa2;
    int want_vllm;

    if (seq_len <= 0) {
        return;
    }

    can_pipeline = encoder_fa_pipeline_can_implement(seq_len, hidden_dim, num_heads, head_dim);
    want_fa2 = can_pipeline && encoder_fa_fa2_enabled();
    want_vllm = encoder_fa_vllm_enabled() && flash_fwd_vllm_available();
    want_pv_mma_q32_k32 = can_pipeline && encoder_fa_pv_mma_q32_k32_enabled();

    if (!used_fa && want_vllm) {
        if (!inputs_rope_applied) {
            encoder_rope_maybe_strided(q_slice, k_slice, seq_len, hidden_dim, num_heads,
                                       head_dim, q_stride, k_stride);
        }
        used_pipeline = flash_fwd_vllm_launch(
            (void *)q_slice, (void *)k_slice, (void *)v_slice,
            (void *)out_slice, seq_len, hidden_dim,
            q_stride, k_stride, v_stride, num_heads, head_dim);
        used_fa = used_pipeline;
    }

    if (!used_fa && want_fa2) {
        if (!inputs_rope_applied) {
            encoder_rope_maybe_strided(q_slice, k_slice, seq_len, hidden_dim, num_heads,
                                       head_dim, q_stride, k_stride);
        }
        used_pipeline = encoder_attention_flash_fa2_launch(
            q_slice, k_slice, v_slice, out_slice, seq_len, hidden_dim,
            q_stride, k_stride, v_stride, num_heads, head_dim);
        used_fa = used_pipeline;
    }

    if (!used_fa && want_pv_mma_q32_k32) {
        if (!inputs_rope_applied) {
            encoder_rope_maybe_strided(q_slice, k_slice, seq_len, hidden_dim, num_heads,
                                       head_dim, q_stride, k_stride);
        }
        used_pipeline = encoder_attention_flash_pv_mma_q32_k32_launch(
            q_slice, k_slice, v_slice, out_slice, seq_len, hidden_dim,
            q_stride, k_stride, v_stride, num_heads, head_dim);
        used_fa = used_pipeline;
    }

    if (used_fa) {
        encoder_attention_note_fa(seq_len, num_heads, used_pipeline);
        return;
    }

    CHECK(q_stride == hidden_dim && k_stride == hidden_dim && v_stride == hidden_dim,
          "Strided encoder attention requires Q32_K32 FA path");
    if (!inputs_rope_applied) {
        rope_encoder(q_slice, k_slice, seq_len, num_heads, head_dim, 0.5f);
    }
    encoder_attention_note_cublas(seq_len, num_heads);
    encoder_attention_cublas(handle, q_slice, k_slice, v_slice, out_slice,
                             seq_len, hidden_dim, num_heads, head_dim);
}

static void encoder_forward_internal(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int total_rows,
    int num_seqs,
    const int *seq_lens,
    const int *seq_offsets,
    int max_seq_len,
    const ModelConfig *cfg
) {
    int hidden_dim = cfg->enc_hidden;
    int num_layers = cfg->enc_layers;
    int num_heads = cfg->enc_heads;
    int head_dim = cfg->enc_head_dim;
    int ffn_dim = cfg->enc_ffn;
    char name_buf[256];
    __nv_bfloat16 *layer_out = hidden;

    allocate_encoder_buffers(total_rows, max_seq_len);
    encoder_prepare_row_positions(seq_lens, seq_offsets, total_rows, num_seqs);

    for (int layer = 0; layer < num_layers; layer++) {
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.input_layernorm.weight", layer);
        const Tensor *ln_w = ws_get_const(ws, name_buf);
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.input_layernorm.bias", layer);
        const Tensor *ln_b = ws_get_const(ws, name_buf);
        CHECK(ln_w != NULL, "Weight not found: input_layernorm.weight layer %d", layer);
        layernorm(d_attn_out, layer_out,
                  (__nv_bfloat16 *)ln_w->data,
                  ln_b ? (__nv_bfloat16 *)ln_b->data : NULL,
                  1e-5, total_rows, hidden_dim);

        int use_qkv_fused = encoder_qkv_fused_enabled() &&
                            encoder_fa_pv_mma_q32_k32_enabled() &&
                            d_qkv_buf != NULL &&
                            encoder_qkv_fused_prepare_layer(ws, layer, hidden_dim);
        if (use_qkv_fused) {
            bf16_linear(handle, d_qkv_buf, d_attn_out, d_encoder_qkv_fused_w[layer],
                        d_encoder_qkv_fused_b[layer], total_rows, 3 * hidden_dim, hidden_dim);
            encoder_rope_fast(d_qkv_buf, d_qkv_buf + hidden_dim,
                                         total_rows, num_heads, head_dim,
                                         3 * hidden_dim, 3 * hidden_dim);
        } else {
            snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.q_proj.weight", layer);
            const Tensor *qw = ws_get_const(ws, name_buf);
            snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.q_proj.bias", layer);
            const Tensor *qb = ws_get_const(ws, name_buf);
            bf16_linear(handle, d_q_buf, d_attn_out, (__nv_bfloat16 *)qw->data,
                        qb ? (__nv_bfloat16 *)qb->data : NULL, total_rows, hidden_dim, hidden_dim);

            snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.k_proj.weight", layer);
            const Tensor *kw = ws_get_const(ws, name_buf);
            bf16_linear(handle, d_k_buf, d_attn_out, (__nv_bfloat16 *)kw->data,
                        NULL, total_rows, hidden_dim, hidden_dim);

            snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.v_proj.weight", layer);
            const Tensor *vw = ws_get_const(ws, name_buf);
            snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.v_proj.bias", layer);
            const Tensor *vb = ws_get_const(ws, name_buf);
            bf16_linear(handle, d_v_buf, d_attn_out, (__nv_bfloat16 *)vw->data,
                        vb ? (__nv_bfloat16 *)vb->data : NULL, total_rows, hidden_dim, hidden_dim);
        }

        int use_batched_fa = encoder_fa_mma_batched_enabled() &&
                             encoder_fa_pv_mma_q32_k32_enabled() &&
                             num_seqs > 1;
        int use_batched_fa2 = !use_batched_fa && num_seqs > 1 &&
                              encoder_fa_vllm_enabled() &&
                              flash_fwd_vllm_available();
        if (use_batched_fa) {
            __nv_bfloat16 *q_base = use_qkv_fused ? d_qkv_buf : d_q_buf;
            __nv_bfloat16 *k_base = use_qkv_fused ? d_qkv_buf + hidden_dim : d_k_buf;
            __nv_bfloat16 *v_base = use_qkv_fused ? d_qkv_buf + 2 * hidden_dim : d_v_buf;
            int qkv_stride = use_qkv_fused ? 3 * hidden_dim : hidden_dim;
            encoder_attention_slice_batched(handle, q_base, k_base, v_base, d_attn_out,
                                            d_encoder_seq_lens, d_encoder_seq_offsets,
                                            num_seqs, max_seq_len, hidden_dim,
                                            qkv_stride, qkv_stride, qkv_stride,
                                            num_heads, head_dim, use_qkv_fused);
        } else if (use_batched_fa2) {
            __nv_bfloat16 *q_base = use_qkv_fused ? d_qkv_buf : d_q_buf;
            __nv_bfloat16 *k_base = use_qkv_fused ? d_qkv_buf + hidden_dim : d_k_buf;
            __nv_bfloat16 *v_base = use_qkv_fused ? d_qkv_buf + 2 * hidden_dim : d_v_buf;
            int qkv_stride = use_qkv_fused ? 3 * hidden_dim : hidden_dim;
            if (!use_qkv_fused) {
        encoder_rope_fast(q_base, k_base,
                                              total_rows, num_heads, head_dim,
                                              qkv_stride, qkv_stride);
            }
            int cu_need = MAX_FA_VARLEN_BATCH + 1;
            if (cu_need > d_encoder_cu_seqlens_cap) {
                if (d_encoder_cu_seqlens) { cudaFree(d_encoder_cu_seqlens); d_encoder_cu_seqlens = NULL; }
                CUDA_CHECK(cudaMalloc(&d_encoder_cu_seqlens, (size_t)cu_need * sizeof(int)));
                d_encoder_cu_seqlens_cap = cu_need;
            }
            int seqs_done = 0;
            int fa2_ok = 1;
            while (seqs_done < num_seqs && fa2_ok) {
                int batch_n = num_seqs - seqs_done;
                if (batch_n > MAX_FA_VARLEN_BATCH) batch_n = MAX_FA_VARLEN_BATCH;
                int batch_start_off = seq_offsets[seqs_done];
                int batch_end = seqs_done + batch_n - 1;
                int batch_tokens = seq_offsets[batch_end] + seq_lens[batch_end] - batch_start_off;
                int dense_tma_ok = 0;
                if (encoder_fa_dense_tma_enabled() && flash_fwd_vllm_dense_tma_available()) {
                    int dense_seq_len = seq_lens[seqs_done];
                    int dense_layout = dense_seq_len > 0;
                    for (int i = 0; i < batch_n && dense_layout; i++) {
                        int idx = seqs_done + i;
                        dense_layout = seq_lens[idx] == dense_seq_len &&
                                       seq_offsets[idx] - batch_start_off == i * dense_seq_len;
                    }
                    if (dense_layout) {
                        dense_tma_ok = flash_fwd_vllm_launch_dense_tma(
                            (void *)(q_base + (size_t)batch_start_off * (size_t)qkv_stride),
                            (void *)(k_base + (size_t)batch_start_off * (size_t)qkv_stride),
                            (void *)(v_base + (size_t)batch_start_off * (size_t)qkv_stride),
                            (void *)(d_attn_out + (size_t)batch_start_off * (size_t)hidden_dim),
                            batch_n, dense_seq_len, num_heads, head_dim,
                            qkv_stride, qkv_stride, qkv_stride);
                    }
                }
                if (!dense_tma_ok) {
                int host_cu[MAX_FA_VARLEN_BATCH + 1];
                for (int i = 0; i < batch_n; i++) {
                    host_cu[i] = seq_offsets[seqs_done + i] - batch_start_off;
                }
                host_cu[batch_n] = batch_tokens;
                CUDA_CHECK(cudaMemcpy(d_encoder_cu_seqlens, host_cu,
                                       (size_t)(batch_n + 1) * sizeof(int), cudaMemcpyHostToDevice));
                fa2_ok = flash_fwd_vllm_launch_varlen(
                    (void *)(q_base + (size_t)batch_start_off * (size_t)qkv_stride),
                    (void *)(k_base + (size_t)batch_start_off * (size_t)qkv_stride),
                    (void *)(v_base + (size_t)batch_start_off * (size_t)qkv_stride),
                    (void *)(d_attn_out + (size_t)batch_start_off * (size_t)hidden_dim),
                    batch_tokens, max_seq_len, num_heads, head_dim,
                    qkv_stride, qkv_stride, qkv_stride,
                    batch_n, d_encoder_cu_seqlens);
                }
                seqs_done += batch_n;
            }
            if (!fa2_ok) {
                use_batched_fa2 = 0;
            }
        }
        if (!use_batched_fa && !use_batched_fa2) {
            for (int seq_idx = 0; seq_idx < num_seqs; seq_idx++) {
                int seq_len = seq_lens[seq_idx];
                int seq_offset = seq_offsets[seq_idx];
                __nv_bfloat16 *q_slice = use_qkv_fused ?
                    d_qkv_buf + (size_t)seq_offset * (size_t)(3 * hidden_dim) :
                    d_q_buf + (size_t)seq_offset * hidden_dim;
                __nv_bfloat16 *k_slice = use_qkv_fused ?
                    q_slice + hidden_dim :
                    d_k_buf + (size_t)seq_offset * hidden_dim;
                __nv_bfloat16 *v_slice = use_qkv_fused ?
                    q_slice + (size_t)2 * hidden_dim :
                    d_v_buf + (size_t)seq_offset * hidden_dim;
                __nv_bfloat16 *out_slice = d_attn_out + (size_t) seq_offset * hidden_dim;
                int qkv_stride = use_qkv_fused ? 3 * hidden_dim : hidden_dim;

                encoder_attention_slice(handle, q_slice, k_slice, v_slice, out_slice,
                                        seq_len, hidden_dim,
                                        qkv_stride, qkv_stride, qkv_stride,
                                        num_heads, head_dim, use_qkv_fused);
            }
        }

        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.o_proj.weight", layer);
        const Tensor *ow = ws_get_const(ws, name_buf);
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.self_attn.o_proj.bias", layer);
        const Tensor *ob = ws_get_const(ws, name_buf);
        bf16_linear(handle, d_q_buf, d_attn_out, (__nv_bfloat16 *)ow->data,
                    NULL, total_rows, hidden_dim, hidden_dim);
        residual_add_bias(layer_out, d_q_buf, layer_out,
                          ob ? (__nv_bfloat16 *)ob->data : NULL,
                          total_rows, hidden_dim);

        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.post_attention_layernorm.weight", layer);
        const Tensor *pln_w = ws_get_const(ws, name_buf);
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.post_attention_layernorm.bias", layer);
        const Tensor *pln_b = ws_get_const(ws, name_buf);
        layernorm(d_attn_out, layer_out,
                  (__nv_bfloat16 *)pln_w->data,
                  pln_b ? (__nv_bfloat16 *)pln_b->data : NULL,
                  1e-5, total_rows, hidden_dim);

        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.mlp.fc1.weight", layer);
        const Tensor *fc1w = ws_get_const(ws, name_buf);
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.mlp.fc1.bias", layer);
        const Tensor *fc1b = ws_get_const(ws, name_buf);
        bf16_linear_gelu(handle, d_hidden_buf, d_attn_out, (__nv_bfloat16 *)fc1w->data,
                         fc1b ? (__nv_bfloat16 *)fc1b->data : NULL, total_rows, ffn_dim, hidden_dim);

        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.mlp.fc2.weight", layer);
        const Tensor *fc2w = ws_get_const(ws, name_buf);
        snprintf(name_buf, sizeof(name_buf), "audio_tower.layers.%d.mlp.fc2.bias", layer);
        const Tensor *fc2b = ws_get_const(ws, name_buf);
        bf16_linear(handle, d_q_buf, d_hidden_buf, (__nv_bfloat16 *)fc2w->data,
                    NULL, total_rows, hidden_dim, ffn_dim);
        residual_add_bias(layer_out, d_q_buf, layer_out,
                          fc2b ? (__nv_bfloat16 *)fc2b->data : NULL,
                          total_rows, hidden_dim);
    }

    {
        const Tensor *norm_w = ws_get_const(ws, "audio_tower.norm.weight");
        const Tensor *norm_b = ws_get_const(ws, "audio_tower.norm.bias");
        layernorm(hidden, hidden,
                  (__nv_bfloat16 *)norm_w->data,
                  norm_b ? (__nv_bfloat16 *)norm_b->data : NULL,
                  1e-5, total_rows, hidden_dim);
    }
}

extern "C" void encoder_scratch_profile_reset(void) {
    memset(&encoder_scratch_stats, 0, sizeof(encoder_scratch_stats));
}

extern "C" void encoder_scratch_profile_get(ScratchProfileStats *out_stats) {
    if (out_stats == NULL) {
        return;
    }
    *out_stats = encoder_scratch_stats;
}

extern "C" void encoder_attention_profile_reset(void) {
    memset(&encoder_attention_stats, 0, sizeof(encoder_attention_stats));
}

extern "C" void encoder_attention_profile_get(EncoderAttentionProfileStats *out_stats) {
    if (out_stats == NULL) {
        return;
    }
    *out_stats = encoder_attention_stats;
}

extern "C" void encoder_attention_microbench(void *handle,
                                             int hidden_dim,
                                             int num_heads,
                                             int head_dim,
                                             const int *seq_lens,
                                             int n_seq_lens,
                                             int warmup_iters,
                                             int iters) {
    cublasHandle_t cublas_handle = (cublasHandle_t)handle;
    int max_seq = 0;
    size_t buf_bytes;
    __nv_bfloat16 *q = NULL;
    __nv_bfloat16 *k = NULL;
    __nv_bfloat16 *v = NULL;
    __nv_bfloat16 *out = NULL;
    cudaEvent_t start = NULL;
    cudaEvent_t stop = NULL;
    int pv_mma_q32_k32_blocks_per_sm = 0;
    int fa2_blocks_per_sm = 0;
    int sm_count = 0;
    int pv_mma_q32_k32_ready;
    int fa2_ready;

    if (cublas_handle == NULL || seq_lens == NULL || n_seq_lens <= 0) {
        fprintf(stderr, "ENCODER_ATTN_BENCH invalid args\n");
        return;
    }
    if (hidden_dim <= 0 || num_heads <= 0 || head_dim <= 0) {
        fprintf(stderr, "ENCODER_ATTN_BENCH invalid dims hidden=%d heads=%d head_dim=%d\n",
                hidden_dim, num_heads, head_dim);
        return;
    }
    if (iters <= 0) {
        iters = 1;
    }
    if (warmup_iters < 0) {
        warmup_iters = 0;
    }

    for (int i = 0; i < n_seq_lens; i++) {
        if (seq_lens[i] > max_seq) {
            max_seq = seq_lens[i];
        }
    }
    if (max_seq <= 0) {
        fprintf(stderr, "ENCODER_ATTN_BENCH no positive seq_len\n");
        return;
    }

    pv_mma_q32_k32_ready = encoder_qk_mma_supported();
    fa2_ready = encoder_qk_mma_supported();
    buf_bytes = (size_t)max_seq * (size_t)hidden_dim * sizeof(__nv_bfloat16);

    allocate_encoder_buffers(max_seq, max_seq);

    CUDA_CHECK(cudaMalloc(&q, buf_bytes));
    CUDA_CHECK(cudaMalloc(&k, buf_bytes));
    CUDA_CHECK(cudaMalloc(&v, buf_bytes));
    CUDA_CHECK(cudaMalloc(&out, buf_bytes));
    CUDA_CHECK(cudaMemset(q, 0, buf_bytes));
    CUDA_CHECK(cudaMemset(k, 0, buf_bytes));
    CUDA_CHECK(cudaMemset(v, 0, buf_bytes));
    CUDA_CHECK(cudaMemset(out, 0, buf_bytes));

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    if (pv_mma_q32_k32_ready) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &pv_mma_q32_k32_blocks_per_sm,
            encoder_flash_attention_tile_kernel_pv_mma_q32_k32,
            ENCODER_FA_Q32_THREADS,
            0));
    }
    if (fa2_ready) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &fa2_blocks_per_sm,
            encoder_flash_attention_tile_kernel_fa2,
            ENCODER_FA_FA2_THREADS,
            0));
    }
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

    fprintf(stderr,
            "ENCODER_ATTN_BENCH config hidden=%d heads=%d head_dim=%d warmup=%d iters=%d sm_count=%d "
            "pv_mma_q32_k32_ready=%d fa2_ready=%d occ_blocks_per_sm(pv_mma_q32_k32=%d,fa2=%d)\n",
            hidden_dim, num_heads, head_dim, warmup_iters, iters, sm_count,
            pv_mma_q32_k32_ready, fa2_ready,
            pv_mma_q32_k32_blocks_per_sm, fa2_blocks_per_sm);

    for (int idx = 0; idx < n_seq_lens; idx++) {
        int seq = seq_lens[idx];
        float cublas_ms = 0.0f;
        float cublas_rope_ms = 0.0f;
        float pv_mma_q32_k32_ms = -1.0f;
        float fa2_ms = -1.0f;
        float speedup_pv_mma_q32_k32 = 0.0f;
        float speedup_fa2 = 0.0f;

        if (seq <= 0 || seq > max_seq) {
            continue;
        }

        for (int i = 0; i < warmup_iters; i++) {
            encoder_attention_cublas(cublas_handle, q, k, v, out, seq, hidden_dim, num_heads, head_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            encoder_attention_cublas(cublas_handle, q, k, v, out, seq, hidden_dim, num_heads, head_dim);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));
        cublas_ms /= (float)iters;

        for (int i = 0; i < warmup_iters; i++) {
            rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
            encoder_attention_cublas(cublas_handle, q, k, v, out, seq, hidden_dim, num_heads, head_dim);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
            encoder_attention_cublas(cublas_handle, q, k, v, out, seq, hidden_dim, num_heads, head_dim);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&cublas_rope_ms, start, stop));
        cublas_rope_ms /= (float)iters;

        if (pv_mma_q32_k32_ready) {
            for (int i = 0; i < warmup_iters; i++) {
                rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
                (void)encoder_attention_flash_pv_mma_q32_k32_launch(q, k, v, out, seq, hidden_dim,
                                                                     hidden_dim, hidden_dim, hidden_dim,
                                                                     num_heads, head_dim);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++) {
                rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
                (void)encoder_attention_flash_pv_mma_q32_k32_launch(q, k, v, out, seq, hidden_dim,
                                                                     hidden_dim, hidden_dim, hidden_dim,
                                                                     num_heads, head_dim);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&pv_mma_q32_k32_ms, start, stop));
            pv_mma_q32_k32_ms /= (float)iters;
        }

        if (fa2_ready) {
            for (int i = 0; i < warmup_iters; i++) {
                rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
                (void)encoder_attention_flash_fa2_launch(q, k, v, out, seq, hidden_dim,
                                                         hidden_dim, hidden_dim, hidden_dim,
                                                         num_heads, head_dim);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++) {
                rope_encoder(q, k, seq, num_heads, head_dim, 0.5f);
                (void)encoder_attention_flash_fa2_launch(q, k, v, out, seq, hidden_dim,
                                                         hidden_dim, hidden_dim, hidden_dim,
                                                         num_heads, head_dim);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&fa2_ms, start, stop));
            fa2_ms /= (float)iters;
        }

        if (pv_mma_q32_k32_ms > 0.0f) {
            speedup_pv_mma_q32_k32 = cublas_rope_ms / pv_mma_q32_k32_ms;
        }
        if (fa2_ms > 0.0f) {
            speedup_fa2 = cublas_rope_ms / fa2_ms;
        }

        fprintf(stderr,
                "ENCODER_ATTN_BENCH seq=%d cublas_ms=%.4f cublas_rope_ms=%.4f "
                "pv_mma_q32_k32_ms=%.4f pv_mma_q32_k32_speedup=%.3fx "
                "fa2_ms=%.4f fa2_speedup=%.3fx\n",
                seq, cublas_ms, cublas_rope_ms,
                pv_mma_q32_k32_ms, speedup_pv_mma_q32_k32,
                fa2_ms, speedup_fa2);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (start != NULL) {
        cudaEventDestroy(start);
    }
    if (stop != NULL) {
        cudaEventDestroy(stop);
    }
    if (q != NULL) {
        cudaFree(q);
    }
    if (k != NULL) {
        cudaFree(k);
    }
    if (v != NULL) {
        cudaFree(v);
    }
    if (out != NULL) {
        cudaFree(out);
    }
}

extern "C" void encoder_cleanup(void) {
    free_encoder_buffers();
}

/* softmax handled by softmax_inplace from kernels.cu */

extern "C"
void encoder_forward(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int seq_len,
    const ModelConfig *cfg
) {
    const int seq_offsets[1] = {0};
    encoder_forward_internal(handle, ws, hidden,
                             seq_len, 1, &seq_len, seq_offsets,
                             seq_len, cfg);
}

extern "C"
void encoder_forward_packed(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int total_rows,
    int num_seqs,
    const int *seq_lens,
    const int *seq_offsets,
    const ModelConfig *cfg
) {
    int max_seq_len = 0;

    for (int i = 0; i < num_seqs; i++) {
        if (seq_lens[i] > max_seq_len) {
            max_seq_len = seq_lens[i];
        }
    }

    encoder_forward_internal(handle, ws, hidden,
                             total_rows, num_seqs, seq_lens, seq_offsets,
                             max_seq_len, cfg);
}
