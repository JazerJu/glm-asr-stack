#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "kernels.cuh"
#include "../include/types.h"

extern "C" void cublas_restore_workspace(void *handle);

#define WARP_SIZE 32
#define DECODER_FA_Q_TILE 32
#define DECODER_FA_KV_TILE 32
#define DECODER_FA_HEAD_DIM 128
#define DECODER_FA_MMA_TILE 16
#define DECODER_FA_QK_WARPS 2
#define DECODER_FA_PV_N_TILE 16
#define DECODER_FA_PV_N_WARPS (DECODER_FA_HEAD_DIM / DECODER_FA_PV_N_TILE)
#define DECODER_FA_PV_WARPS (DECODER_FA_QK_WARPS * DECODER_FA_PV_N_WARPS)
#define DECODER_FA_THREADS (DECODER_FA_PV_WARPS * WARP_SIZE)
#define DECODER_SOFTMAX_THREADS 256

static const char *g_dump_decoder_dir = NULL;
static int g_use_cublas_decoder_attention = -1;
static int g_decoder_flash_attention_supported = -1;

static int get_use_cublas_decoder_attention(void) {
    if (g_use_cublas_decoder_attention < 0) {
        const char *env = getenv("FA_FORCEALIGNER_USE_CUBLAS_DECODER");
        g_use_cublas_decoder_attention = (env && atoi(env)) ? 1 : 0;
    }
    return g_use_cublas_decoder_attention;
}

static int get_decoder_flash_attention_supported(void) {
    if (g_decoder_flash_attention_supported < 0) {
        int device = 0;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        g_decoder_flash_attention_supported = (prop.major >= 8) ? 1 : 0;
    }
    return g_decoder_flash_attention_supported;
}

static __device__ __forceinline__ float decoder_fa_fast_exp(float x) {
    return __expf(x);
}

static void dump_bf16_dec(const char *name, const bf16_t *dev_ptr, size_t count, cudaStream_t stream) {
    if (!g_dump_decoder_dir) return;
    static char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", g_dump_decoder_dir, name);
    void *host = malloc(count * sizeof(bf16_t));
    if (!host) return;
    cudaMemcpyAsync(host, dev_ptr, count * sizeof(bf16_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(host, sizeof(bf16_t), count, f); fclose(f); }
    free(host);
}

static void init_dec_dump_dir() {
    if (!g_dump_decoder_dir) {
        g_dump_decoder_dir = getenv("FA_DUMP_DECODER_DIR");
        if (g_dump_decoder_dir) {
            mkdir(g_dump_decoder_dir, 0755);
            fprintf(stderr, "[dump] Decoder dumps -> %s\n", g_dump_decoder_dir);
        }
    }
}

static bf16_t *d_hidden_states = NULL;
static bf16_t *d_scratch = NULL;
static bf16_t *d_attn_scores = NULL;
static int scratch_seq_capacity = 0;
static int attn_score_seq_capacity = 0;

static inline const Tensor *must_get_weight(const WeightStore *ws, const char *name) {
    const Tensor *t = ws_get_const(ws, name);
    CHECK(t != NULL, "Missing text decoder weight: %s", name);
    return t;
}

__global__ void elementwise_mul_kernel(__nv_bfloat16 *x,
                                       const __nv_bfloat16 *y,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float xv = __bfloat162float(x[idx]);
    float yv = __bfloat162float(y[idx]);
    x[idx] = __float2bfloat16(xv * yv);
}

__global__ void silu_mul_inplace_kernel(__nv_bfloat16 *x,
                                        const __nv_bfloat16 *y,
                                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float xv = __bfloat162float(x[idx]);
    const __nv_bfloat16 silu = __float2bfloat16(xv / (1.0f + expf(-xv)));
    const float yv = __bfloat162float(y[idx]);
    x[idx] = __float2bfloat16(__bfloat162float(silu) * yv);
}

__device__ inline float decoder_warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ inline float decoder_warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    return val;
}

__global__ void causal_softmax_inplace_kernel(__nv_bfloat16 *scores, int seq_len) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16 *row_ptr = scores + (size_t)row * seq_len;

    float local_max = -INFINITY;
    for (int col = tid; col <= row; col += blockDim.x) {
        float v = __bfloat162float(row_ptr[col]);
        local_max = fmaxf(local_max, v);
    }
    local_max = decoder_warp_reduce_max(local_max);

    __shared__ float warp_max[DECODER_SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_max[tid / WARP_SIZE] = local_max;
    }
    __syncthreads();

    float row_max = (tid < blockDim.x / WARP_SIZE) ? warp_max[tid] : -INFINITY;
    if (tid < WARP_SIZE) {
        row_max = decoder_warp_reduce_max(row_max);
    }

    __shared__ float max_shared;
    if (tid == 0) {
        max_shared = row_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = tid; col <= row; col += blockDim.x) {
        float v = __bfloat162float(row_ptr[col]);
        local_sum += expf(v - max_shared);
    }
    local_sum = decoder_warp_reduce_sum(local_sum);

    __shared__ float warp_sum[DECODER_SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sum[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    float row_sum = (tid < blockDim.x / WARP_SIZE) ? warp_sum[tid] : 0.0f;
    if (tid < WARP_SIZE) {
        row_sum = decoder_warp_reduce_sum(row_sum);
    }

    __shared__ float inv_sum;
    if (tid == 0) {
        inv_sum = 1.0f / row_sum;
    }
    __syncthreads();

    for (int col = tid; col <= row; col += blockDim.x) {
        const float v = __bfloat162float(row_ptr[col]);
        row_ptr[col] = __float2bfloat16(expf(v - max_shared) * inv_sum);
    }
    for (int col = row + 1 + tid; col < seq_len; col += blockDim.x) {
        row_ptr[col] = __float2bfloat16(0.0f);
    }
}

__global__ void head_rmsnorm_inplace_kernel(__nv_bfloat16 *x,
                                            const __nv_bfloat16 *weight,
                                            int seq_len,
                                            int num_heads,
                                            int head_dim,
                                            float eps) {
    int head = blockIdx.x;
    int seq = blockIdx.y;
    int tid = threadIdx.x;
    if (head >= num_heads || seq >= seq_len) return;

    __nv_bfloat16 *row = x + ((size_t)seq * num_heads + head) * head_dim;
    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += WARP_SIZE) {
        float v = __bfloat162float(row[i]);
        sum_sq += v * v;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int i = tid; i < head_dim; i += WARP_SIZE) {
        float v = __bfloat162float(row[i]);
        float w = __bfloat162float(weight[i]);
        row[i] = __float2bfloat16(v * inv_rms * w);
    }
}

static void ensure_text_decoder_buffers(int seq_len, const AlignerConfig *cfg, bool need_cublas_attention_workspace, cudaStream_t stream) {
    size_t hidden_elems = (size_t)seq_len * cfg->dec_hidden;
    size_t q_hidden_elems = (size_t)seq_len * (size_t)(cfg->dec_heads * cfg->dec_head_dim);
    size_t kv_hidden_elems = (size_t)seq_len * (size_t)(cfg->dec_kv_heads * cfg->dec_head_dim);
    size_t scratch_elems =
        hidden_elems +      /* norm */
        q_hidden_elems +    /* q */
        kv_hidden_elems +   /* k */
        kv_hidden_elems +   /* v */
        q_hidden_elems +    /* attn_out */
        hidden_elems +      /* o_proj */
        (size_t)seq_len * (size_t)cfg->dec_ffn +  /* mlp_gate */
        (size_t)seq_len * (size_t)cfg->dec_ffn +  /* mlp_up */
        hidden_elems;       /* mlp_down */

    if (seq_len > scratch_seq_capacity) {
        if (d_hidden_states != NULL) CUDA_CHECK(cudaFree(d_hidden_states));
        if (d_scratch != NULL) CUDA_CHECK(cudaFree(d_scratch));

        CUDA_CHECK(cudaMalloc(&d_hidden_states, hidden_elems * sizeof(bf16_t)));
        CUDA_CHECK(cudaMalloc(&d_scratch, scratch_elems * sizeof(bf16_t)));

        scratch_seq_capacity = seq_len;
    }

    if (need_cublas_attention_workspace) {
        size_t score_elems = (size_t)seq_len * seq_len;
        if (seq_len > attn_score_seq_capacity) {
            if (d_attn_scores != NULL) CUDA_CHECK(cudaFree(d_attn_scores));
            CUDA_CHECK(cudaMalloc(&d_attn_scores, score_elems * sizeof(bf16_t)));
            attn_score_seq_capacity = seq_len;
        }
    }
}

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
__global__ static void decoder_flash_attention_kernel(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k,
    const __nv_bfloat16 *v,
    __nv_bfloat16 *attn_out,
    int seq_len,
    int q_hidden,
    int kv_hidden,
    int q_per_kv) {
#if !defined(__CUDA_ARCH__)
    (void)q;
    (void)k;
    (void)v;
    (void)attn_out;
    (void)seq_len;
    (void)q_hidden;
    (void)kv_hidden;
    (void)q_per_kv;
#elif (__CUDA_ARCH__ < 800)
    (void)q;
    (void)k;
    (void)v;
    (void)attn_out;
    (void)seq_len;
    (void)q_hidden;
    (void)kv_hidden;
    (void)q_per_kv;
#else
    extern __shared__ __nv_bfloat16 sh_mem[];
    __nv_bfloat16 *sh_q = sh_mem;
    __nv_bfloat16 *sh_k = sh_q + DECODER_FA_Q_TILE * DECODER_FA_HEAD_DIM;
    __nv_bfloat16 *sh_v = sh_k + DECODER_FA_KV_TILE * DECODER_FA_HEAD_DIM;
    __nv_bfloat16 *sh_score = sh_v + DECODER_FA_KV_TILE * DECODER_FA_HEAD_DIM;
    __nv_bfloat16 *sh_out = sh_score + DECODER_FA_Q_TILE * DECODER_FA_KV_TILE;
    __shared__ float sh_m[DECODER_FA_Q_TILE];
    __shared__ float sh_l[DECODER_FA_Q_TILE];
    __shared__ float sh_alpha[DECODER_FA_Q_TILE];

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int kv_head = blockIdx.x;
    const int q_tile_idx = blockIdx.y;
    const int q_start = q_tile_idx * DECODER_FA_Q_TILE;
    const float qk_scale = 1.0f / sqrtf((float)DECODER_FA_HEAD_DIM);
    const int row_block = warp / DECODER_FA_PV_N_WARPS;
    const int pv_row_base = row_block * DECODER_FA_MMA_TILE;
    const int pv_col_base = (warp & (DECODER_FA_PV_N_WARPS - 1)) * DECODER_FA_PV_N_TILE;

    for (int q_local = 0; q_local < q_per_kv; ++q_local) {
        const int q_head = kv_head * q_per_kv + q_local;
        const __nv_bfloat16 *q_head_ptr = q + q_head * DECODER_FA_HEAD_DIM;
        const __nv_bfloat16 *k_head_ptr = k + kv_head * DECODER_FA_HEAD_DIM;
        const __nv_bfloat16 *v_head_ptr = v + kv_head * DECODER_FA_HEAD_DIM;
        __nv_bfloat16 *out_head_ptr = attn_out + q_head * DECODER_FA_HEAD_DIM;

        for (int idx = threadIdx.x; idx < DECODER_FA_Q_TILE * DECODER_FA_HEAD_DIM; idx += blockDim.x) {
            const int tq = idx / DECODER_FA_HEAD_DIM;
            const int td = idx % DECODER_FA_HEAD_DIM;
            const int q_row = q_start + tq;
            __nv_bfloat16 q_val = __float2bfloat16(0.0f);
            if (q_row < seq_len) {
                q_val = q_head_ptr[(size_t)q_row * q_hidden + td];
            }
            sh_q[idx] = q_val;
        }
        if (threadIdx.x < DECODER_FA_Q_TILE) {
            const int q_row = q_start + threadIdx.x;
            sh_m[threadIdx.x] = (q_row < seq_len) ? -INFINITY : 0.0f;
            sh_l[threadIdx.x] = 0.0f;
            sh_alpha[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                               DECODER_FA_MMA_TILE,
                               DECODER_FA_MMA_TILE,
                               DECODER_FA_MMA_TILE,
                               __nv_bfloat16,
                               nvcuda::wmma::row_major> q_frag[DECODER_FA_HEAD_DIM / DECODER_FA_MMA_TILE];
        if (warp < DECODER_FA_QK_WARPS) {
            const int row_base = warp * DECODER_FA_MMA_TILE;
#pragma unroll
            for (int d = 0; d < DECODER_FA_HEAD_DIM; d += DECODER_FA_MMA_TILE) {
                nvcuda::wmma::load_matrix_sync(q_frag[d / DECODER_FA_MMA_TILE],
                                               &sh_q[row_base * DECODER_FA_HEAD_DIM + d],
                                               DECODER_FA_HEAD_DIM);
            }
        }

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                               DECODER_FA_MMA_TILE,
                               DECODER_FA_PV_N_TILE,
                               DECODER_FA_MMA_TILE,
                               float> pv_frag;
        if (warp < DECODER_FA_PV_WARPS) {
            nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        }

        for (int kv_start = 0; kv_start < seq_len; kv_start += DECODER_FA_KV_TILE) {
            int tile_len = seq_len - kv_start;
            if (tile_len > DECODER_FA_KV_TILE) {
                tile_len = DECODER_FA_KV_TILE;
            }

            for (int idx = threadIdx.x; idx < DECODER_FA_KV_TILE * DECODER_FA_HEAD_DIM; idx += blockDim.x) {
                sh_k[idx] = __float2bfloat16(0.0f);
                sh_v[idx] = __float2bfloat16(0.0f);
            }
            for (int idx = threadIdx.x; idx < DECODER_FA_Q_TILE * DECODER_FA_KV_TILE; idx += blockDim.x) {
                sh_score[idx] = __float2bfloat16(0.0f);
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < tile_len * DECODER_FA_HEAD_DIM; idx += blockDim.x) {
                const int tk = idx / DECODER_FA_HEAD_DIM;
                const int td = idx % DECODER_FA_HEAD_DIM;
                const int kv_row = kv_start + tk;
                sh_k[tk * DECODER_FA_HEAD_DIM + td] = k_head_ptr[(size_t)kv_row * kv_hidden + td];
                sh_v[tk * DECODER_FA_HEAD_DIM + td] = v_head_ptr[(size_t)kv_row * kv_hidden + td];
            }
            __syncthreads();

            if (warp < DECODER_FA_QK_WARPS) {
                const int row_base = warp * DECODER_FA_MMA_TILE;
                const int lane_group = lane >> 2;
                const unsigned row_group_mask = 0xFu << (lane_group * 4);
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_MMA_TILE,
                                       __nv_bfloat16,
                                       nvcuda::wmma::col_major> k_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_MMA_TILE,
                                       float> score_frag[DECODER_FA_KV_TILE / DECODER_FA_MMA_TILE];

#pragma unroll
                for (int sub_idx = 0; sub_idx < DECODER_FA_KV_TILE / DECODER_FA_MMA_TILE; ++sub_idx) {
                    const int sub = sub_idx * DECODER_FA_MMA_TILE;
                    nvcuda::wmma::fill_fragment(score_frag[sub_idx], 0.0f);
                    if (sub >= tile_len) {
                        continue;
                    }
#pragma unroll
                    for (int d = 0; d < DECODER_FA_HEAD_DIM; d += DECODER_FA_MMA_TILE) {
                        nvcuda::wmma::load_matrix_sync(k_frag,
                                                       &sh_k[sub * DECODER_FA_HEAD_DIM + d],
                                                       DECODER_FA_HEAD_DIM);
                        nvcuda::wmma::mma_sync(score_frag[sub_idx],
                                               q_frag[d / DECODER_FA_MMA_TILE],
                                               k_frag,
                                               score_frag[sub_idx]);
                    }
                }

                float row_max_low = -INFINITY;
                float row_max_high = -INFINITY;
#pragma unroll
                for (int sub_idx = 0; sub_idx < DECODER_FA_KV_TILE / DECODER_FA_MMA_TILE; ++sub_idx) {
                    const int sub = sub_idx * DECODER_FA_MMA_TILE;
#pragma unroll
                    for (int frag_idx = 0; frag_idx < score_frag[sub_idx].num_elements; ++frag_idx) {
                        const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                        const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                        const int q_row = q_start + row_base + local_row;
                        const int k_col = kv_start + sub + local_col;
                        float score = -INFINITY;
                        if (local_col + sub < tile_len && q_row < seq_len && k_col <= q_row) {
                            score = score_frag[sub_idx].x[frag_idx] * qk_scale;
                        }
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

                const float old_m_low = sh_m[row_base + lane_group];
                const float old_l_low = sh_l[row_base + lane_group];
                const float old_m_high = sh_m[row_base + lane_group + 8];
                const float old_l_high = sh_l[row_base + lane_group + 8];
                const float new_m_low = fmaxf(old_m_low, row_max_low);
                const float new_m_high = fmaxf(old_m_high, row_max_high);
                float p_sum_low = 0.0f;
                float p_sum_high = 0.0f;

#pragma unroll
                for (int sub_idx = 0; sub_idx < DECODER_FA_KV_TILE / DECODER_FA_MMA_TILE; ++sub_idx) {
                    const int sub = sub_idx * DECODER_FA_MMA_TILE;
#pragma unroll
                    for (int frag_idx = 0; frag_idx < score_frag[sub_idx].num_elements; ++frag_idx) {
                        const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                        const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                        const int q_row = q_start + row_base + local_row;
                        const int tk = sub + local_col;
                        const int k_col = kv_start + tk;
                        float p = 0.0f;
                        if (tk < tile_len && q_row < seq_len && k_col <= q_row) {
                            const float new_m = (local_row < 8) ? new_m_low : new_m_high;
                            p = decoder_fa_fast_exp(score_frag[sub_idx].x[frag_idx] * qk_scale - new_m);
                        }
                        sh_score[(row_base + local_row) * DECODER_FA_KV_TILE + tk] = __float2bfloat16_rn(p);
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
                    const float alpha_low = decoder_fa_fast_exp(old_m_low - new_m_low);
                    const float alpha_high = decoder_fa_fast_exp(old_m_high - new_m_high);
                    sh_m[row_base + lane_group] = new_m_low;
                    sh_l[row_base + lane_group] = old_l_low * alpha_low + p_sum_low;
                    sh_alpha[row_base + lane_group] = alpha_low;
                    sh_m[row_base + lane_group + 8] = new_m_high;
                    sh_l[row_base + lane_group + 8] = old_l_high * alpha_high + p_sum_high;
                    sh_alpha[row_base + lane_group + 8] = alpha_high;
                }
            }
            __syncthreads();

            if (warp < DECODER_FA_PV_WARPS) {
                if (kv_start != 0) {
                    for (int frag_idx = 0; frag_idx < pv_frag.num_elements; ++frag_idx) {
                        const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                        pv_frag.x[frag_idx] *= sh_alpha[pv_row_base + local_row];
                    }
                }

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_PV_N_TILE,
                                       DECODER_FA_MMA_TILE,
                                       __nv_bfloat16,
                                       nvcuda::wmma::row_major> p_frag;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                       DECODER_FA_MMA_TILE,
                                       DECODER_FA_PV_N_TILE,
                                       DECODER_FA_MMA_TILE,
                                       __nv_bfloat16,
                                       nvcuda::wmma::row_major> v_frag;
                for (int sub = 0; sub < DECODER_FA_KV_TILE; sub += DECODER_FA_MMA_TILE) {
                    nvcuda::wmma::load_matrix_sync(p_frag,
                                                   &sh_score[pv_row_base * DECODER_FA_KV_TILE + sub],
                                                   DECODER_FA_KV_TILE);
                    nvcuda::wmma::load_matrix_sync(v_frag,
                                                   &sh_v[sub * DECODER_FA_HEAD_DIM + pv_col_base],
                                                   DECODER_FA_HEAD_DIM);
                    nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x < DECODER_FA_Q_TILE) {
            const int q_row = q_start + threadIdx.x;
            sh_alpha[threadIdx.x] = (q_row < seq_len && sh_l[threadIdx.x] > 0.0f) ? __frcp_rn(sh_l[threadIdx.x]) : 0.0f;
        }
        __syncthreads();

        if (warp < DECODER_FA_PV_WARPS) {
            for (int frag_idx = 0; frag_idx < pv_frag.num_elements; ++frag_idx) {
                const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                const int row = pv_row_base + local_row;
                const int col = pv_col_base + local_col;
                sh_out[row * DECODER_FA_HEAD_DIM + col] = __float2bfloat16_rn(pv_frag.x[frag_idx] * sh_alpha[row]);
            }
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < DECODER_FA_Q_TILE * DECODER_FA_HEAD_DIM; idx += blockDim.x) {
            const int tq = idx / DECODER_FA_HEAD_DIM;
            const int td = idx % DECODER_FA_HEAD_DIM;
            const int q_row = q_start + tq;
            if (q_row < seq_len) {
                out_head_ptr[(size_t)q_row * q_hidden + td] = sh_out[idx];
            }
        }
        __syncthreads();
    }
#endif
}
#endif

#if 0
static void attention_prefill(cublasHandle_t handle,
                              cudaStream_t stream,
                              __nv_bfloat16 *attn_out,
                              __nv_bfloat16 *q,
                              __nv_bfloat16 *k,
                              __nv_bfloat16 *v,
                              int seq_len,
                              const AlignerConfig *cfg) {
    const float qk_scale = 1.0f / sqrtf((float)cfg->dec_head_dim);
    const float one = 1.0f;
    const float zero = 0.0f;
    int q_per_kv = cfg->dec_heads / cfg->dec_kv_heads;
    int hidden = cfg->dec_heads * cfg->dec_head_dim;
    int kv_hidden = cfg->dec_kv_heads * cfg->dec_head_dim;
    int head_dim = cfg->dec_head_dim;
    size_t score_elems = (size_t)seq_len * seq_len;

    /* Use cublasGemmStridedBatchedEx for QK and PV GEMMs.
     * On Blackwell sm_120, cublasGemmEx fails intermittently for
     * repeated calls with non-zero base pointer offsets (GQA head strides).
     * The batched API path handles this correctly. */
    for (int kvh = 0; kvh < cfg->dec_kv_heads; ++kvh) {
        const __nv_bfloat16 *k_head = k + kvh * head_dim;
        const __nv_bfloat16 *v_head = v + kvh * head_dim;

        for (int q_local = 0; q_local < q_per_kv; ++q_local) {
            int qh = kvh * q_per_kv + q_local;
            const __nv_bfloat16 *q_head = q + qh * head_dim;
            __nv_bfloat16 *score = (__nv_bfloat16 *)d_attn_scores;
            __nv_bfloat16 *out_head = attn_out + qh * head_dim;

            cublasStatus_t st = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, head_dim,
                &qk_scale,
                k_head, CUDA_R_16BF, kv_hidden, (long long)0,
                q_head, CUDA_R_16BF, hidden, (long long)0,
                &zero,
                score, CUDA_R_16BF, seq_len, (long long)0,
                1,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            CHECK(st == CUBLAS_STATUS_SUCCESS, "QK GEMM failed for head %d (kvh=%d)", qh, kvh);

            add_causal_mask_kernel<<<(int)((score_elems + 255) / 256), 256, 0, stream>>>(
                score, (const __nv_bfloat16 *)d_causal_mask, (int)score_elems);
            CUDA_CHECK(cudaGetLastError());
            softmax_inplace((bf16_t *)score, seq_len, seq_len, stream);

            st = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, seq_len, seq_len,
                &one,
                v_head, CUDA_R_16BF, kv_hidden, (long long)0,
                score, CUDA_R_16BF, seq_len, (long long)0,
                &zero,
                out_head, CUDA_R_16BF, hidden, (long long)0,
                1,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            CHECK(st == CUBLAS_STATUS_SUCCESS, "PV GEMM failed for head %d (kvh=%d)", qh, kvh);
        }
    }
}
#endif

static void attention_prefill_cublas(cublasHandle_t handle,
                                     cudaStream_t stream,
                                     __nv_bfloat16 *attn_out,
                                     __nv_bfloat16 *q,
                                     __nv_bfloat16 *k,
                                     __nv_bfloat16 *v,
                                     int seq_len,
                                     const AlignerConfig *cfg) {
    const float qk_scale = 1.0f / sqrtf((float)cfg->dec_head_dim);
    const float one = 1.0f;
    const float zero = 0.0f;
    int q_per_kv = cfg->dec_heads / cfg->dec_kv_heads;
    int hidden = cfg->dec_heads * cfg->dec_head_dim;
    int kv_hidden = cfg->dec_kv_heads * cfg->dec_head_dim;
    int head_dim = cfg->dec_head_dim;
    for (int kvh = 0; kvh < cfg->dec_kv_heads; ++kvh) {
        const __nv_bfloat16 *k_head = k + kvh * head_dim;
        const __nv_bfloat16 *v_head = v + kvh * head_dim;

        for (int q_local = 0; q_local < q_per_kv; ++q_local) {
            int qh = kvh * q_per_kv + q_local;
            const __nv_bfloat16 *q_head = q + qh * head_dim;
            __nv_bfloat16 *score = (__nv_bfloat16 *)d_attn_scores;
            __nv_bfloat16 *out_head = attn_out + qh * head_dim;

            cublasStatus_t st = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                seq_len, seq_len, head_dim,
                &qk_scale,
                k_head, CUDA_R_16BF, kv_hidden, (long long)0,
                q_head, CUDA_R_16BF, hidden, (long long)0,
                &zero,
                score, CUDA_R_16BF, seq_len, (long long)0,
                1,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            CHECK(st == CUBLAS_STATUS_SUCCESS, "QK GEMM failed for head %d (kvh=%d)", qh, kvh);

            causal_softmax_inplace_kernel<<<seq_len, DECODER_SOFTMAX_THREADS, 0, stream>>>(score, seq_len);
            CUDA_CHECK(cudaGetLastError());

            st = cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, seq_len, seq_len,
                &one,
                v_head, CUDA_R_16BF, kv_hidden, (long long)0,
                score, CUDA_R_16BF, seq_len, (long long)0,
                &zero,
                out_head, CUDA_R_16BF, hidden, (long long)0,
                1,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT);
            CHECK(st == CUBLAS_STATUS_SUCCESS, "PV GEMM failed for head %d (kvh=%d)", qh, kvh);
        }
    }
}

static void decoder_attention_prefill(cublasHandle_t handle,
                                      cudaStream_t stream,
                                      __nv_bfloat16 *attn_out,
                                      __nv_bfloat16 *q,
                                      __nv_bfloat16 *k,
                                      __nv_bfloat16 *v,
                                      int seq_len,
                                      const AlignerConfig *cfg) {
    const int q_hidden = cfg->dec_heads * cfg->dec_head_dim;
    const int kv_hidden = cfg->dec_kv_heads * cfg->dec_head_dim;

    if (cfg->dec_heads <= 0 ||
        cfg->dec_kv_heads <= 0 ||
        (cfg->dec_heads % cfg->dec_kv_heads) != 0) {
        attention_prefill_cublas(handle, stream, attn_out, q, k, v, seq_len, cfg);
        return;
    }

    const int q_per_kv = cfg->dec_heads / cfg->dec_kv_heads;

    if (get_use_cublas_decoder_attention() ||
        !get_decoder_flash_attention_supported() ||
        seq_len < DECODER_FA_Q_TILE ||
        (seq_len % DECODER_FA_Q_TILE) != 0 ||
        cfg->dec_head_dim != DECODER_FA_HEAD_DIM ||
        q_per_kv <= 0) {
        attention_prefill_cublas(handle, stream, attn_out, q, k, v, seq_len, cfg);
        return;
    }

    const dim3 block(DECODER_FA_THREADS);
    const dim3 grid(cfg->dec_kv_heads, (seq_len + DECODER_FA_Q_TILE - 1) / DECODER_FA_Q_TILE);
    const size_t shmem_bytes =
        (size_t)(DECODER_FA_Q_TILE * DECODER_FA_HEAD_DIM +
                 DECODER_FA_KV_TILE * DECODER_FA_HEAD_DIM +
                 DECODER_FA_KV_TILE * DECODER_FA_HEAD_DIM +
                 DECODER_FA_Q_TILE * DECODER_FA_KV_TILE +
                 DECODER_FA_Q_TILE * DECODER_FA_HEAD_DIM) * sizeof(__nv_bfloat16);
    decoder_flash_attention_kernel<<<grid, block, shmem_bytes, stream>>>(
        q, k, v, attn_out, seq_len, q_hidden, kv_hidden, q_per_kv);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void cuda_text_decoder_forward(
    bf16_t *logits,
    const bf16_t *input_embeds,
    const int *input_ids,
    int seq_len,
    const WeightStore *ws,
    const AlignerConfig *cfg,
    cublasHandle_t handle,
    cudaStream_t stream) {
    char name[160];
    int hidden;
    int kv_hidden;
    int q_hidden;
    int total_hidden;
    init_dec_dump_dir();
    int total_ffn;
    __nv_bfloat16 *norm_buf;
    __nv_bfloat16 *q_buf;
    __nv_bfloat16 *k_buf;
    __nv_bfloat16 *v_buf;
    __nv_bfloat16 *attn_out_buf;
    __nv_bfloat16 *o_proj_buf;
    __nv_bfloat16 *mlp_gate_buf;
    __nv_bfloat16 *mlp_up_buf;
    __nv_bfloat16 *mlp_down_buf;
    cudaStream_t exec_stream = stream;

    (void)input_ids;

    CHECK(logits != NULL, "logits must not be null");
    CHECK(input_embeds != NULL, "input_embeds must not be null");
    CHECK(ws != NULL, "weight store must not be null");
    CHECK(cfg != NULL, "config must not be null");
    CHECK(seq_len > 0, "seq_len must be positive");

    hidden = cfg->dec_hidden;
    kv_hidden = cfg->dec_kv_heads * cfg->dec_head_dim;
    q_hidden = cfg->dec_heads * cfg->dec_head_dim;
    total_hidden = seq_len * hidden;
    total_ffn = seq_len * cfg->dec_ffn;

    const int q_per_kv = (cfg->dec_kv_heads > 0) ? (cfg->dec_heads / cfg->dec_kv_heads) : 0;
    const bool valid_gqa = (cfg->dec_heads > 0 &&
                            cfg->dec_kv_heads > 0 &&
                            (cfg->dec_heads % cfg->dec_kv_heads) == 0);
    const bool need_cublas_attention_workspace =
        !valid_gqa ||
        get_use_cublas_decoder_attention() ||
        !get_decoder_flash_attention_supported() ||
        seq_len < DECODER_FA_Q_TILE ||
        (seq_len % DECODER_FA_Q_TILE) != 0 ||
        cfg->dec_head_dim != DECODER_FA_HEAD_DIM ||
        q_per_kv <= 0;

    ensure_text_decoder_buffers(seq_len, cfg, need_cublas_attention_workspace, exec_stream);
    CUDA_CHECK(cudaMemcpyAsync(d_hidden_states, input_embeds,
                               (size_t)total_hidden * sizeof(bf16_t),
                               cudaMemcpyDeviceToDevice, exec_stream));
    CHECK(cublasSetStream(handle, exec_stream) == CUBLAS_STATUS_SUCCESS,
          "cublasSetStream failed");
    cublas_restore_workspace(handle);

    norm_buf = (__nv_bfloat16 *)d_scratch;
    q_buf = norm_buf + (size_t)seq_len * hidden;
    k_buf = q_buf + (size_t)seq_len * q_hidden;
    v_buf = k_buf + (size_t)seq_len * kv_hidden;
    attn_out_buf = v_buf + (size_t)seq_len * kv_hidden;
    o_proj_buf = attn_out_buf + (size_t)seq_len * q_hidden;
    mlp_gate_buf = o_proj_buf + (size_t)seq_len * hidden;
    mlp_up_buf = mlp_gate_buf + (size_t)seq_len * cfg->dec_ffn;
    mlp_down_buf = mlp_up_buf + (size_t)seq_len * cfg->dec_ffn;

    {
        const Tensor *tw = must_get_weight(ws, "thinker.model.layers.0.input_layernorm.weight");
        rmsnorm((bf16_t *)norm_buf, d_hidden_states,
                (const bf16_t *)tw->data,
                hidden, seq_len, exec_stream);
    }

    for (int layer = 0; layer < cfg->dec_layers; ++layer) {
        const Tensor *tw;

        bool dump_l0 = (layer == 0 && g_dump_decoder_dir != NULL);
        if (dump_l0) {
            dump_bf16_dec("dec_l0_norm_in", norm_buf, (size_t) total_hidden, exec_stream);
        }

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)q_buf, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, q_hidden, hidden, handle, exec_stream);

        if (dump_l0) {
            dump_bf16_dec("dec_l0_q_proj", q_buf, (size_t) seq_len * q_hidden, exec_stream);
        }

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.q_norm.weight", layer);
        tw = must_get_weight(ws, name);
        head_rmsnorm_inplace_kernel<<<dim3(cfg->dec_heads, seq_len), WARP_SIZE, 0, exec_stream>>>(
            q_buf, (const __nv_bfloat16 *)tw->data,
            seq_len, cfg->dec_heads, cfg->dec_head_dim, cfg->dec_rms_eps);
        CUDA_CHECK(cudaGetLastError());

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)k_buf, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, kv_hidden, hidden, handle, exec_stream);

        if (dump_l0) {
            dump_bf16_dec("dec_l0_k_proj", k_buf, (size_t) seq_len * kv_hidden, exec_stream);
        }

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.k_norm.weight", layer);
        tw = must_get_weight(ws, name);
        head_rmsnorm_inplace_kernel<<<dim3(cfg->dec_kv_heads, seq_len), WARP_SIZE, 0, exec_stream>>>(
            k_buf, (const __nv_bfloat16 *)tw->data,
            seq_len, cfg->dec_kv_heads, cfg->dec_head_dim, cfg->dec_rms_eps);
        CUDA_CHECK(cudaGetLastError());

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.v_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)v_buf, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, kv_hidden, hidden, handle, exec_stream);

        if (dump_l0) {
            dump_bf16_dec("dec_l0_v_proj", v_buf, (size_t) seq_len * kv_hidden, exec_stream);
        }

        rope_forward((bf16_t *)q_buf, (bf16_t *)k_buf,
                     seq_len, cfg->dec_heads, cfg->dec_kv_heads,
                     cfg->dec_head_dim, cfg->dec_rope_theta, exec_stream);

        if (dump_l0) {
            dump_bf16_dec("dec_l0_q_rope", q_buf, (size_t) seq_len * q_hidden, exec_stream);
            dump_bf16_dec("dec_l0_k_rope", k_buf, (size_t) seq_len * kv_hidden, exec_stream);
        }

        decoder_attention_prefill(handle, exec_stream, attn_out_buf, q_buf, k_buf, v_buf, seq_len, cfg);

        if (dump_l0) {
            dump_bf16_dec("dec_l0_attn_out", attn_out_buf, (size_t) seq_len * q_hidden, exec_stream);
        }

        snprintf(name, sizeof(name), "thinker.model.layers.%d.self_attn.o_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)o_proj_buf, (const bf16_t *)attn_out_buf, (const bf16_t *)tw->data,
                    seq_len, hidden, q_hidden, handle, exec_stream);
        snprintf(name, sizeof(name), "thinker.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = must_get_weight(ws, name);
        fused_add_residual_rmsnorm((bf16_t *)norm_buf, d_hidden_states, (const bf16_t *)o_proj_buf,
                                   (const bf16_t *)tw->data,
                                   hidden, seq_len, exec_stream);

        if (layer == 0 && g_dump_decoder_dir) {
            dump_bf16_dec("dec_l0_o_proj", o_proj_buf, (size_t) total_hidden, exec_stream);
            dump_bf16_dec("dec_l0_post_attn_res", d_hidden_states, (size_t) total_hidden, exec_stream);
        }

        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.gate_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)mlp_gate_buf, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, cfg->dec_ffn, hidden, handle, exec_stream);

        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.up_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)mlp_up_buf, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, cfg->dec_ffn, hidden, handle, exec_stream);

        silu_mul_inplace_kernel<<<(total_ffn + 255) / 256, 256, 0, exec_stream>>>(mlp_gate_buf, mlp_up_buf, total_ffn);
        CUDA_CHECK(cudaGetLastError());

        snprintf(name, sizeof(name), "thinker.model.layers.%d.mlp.down_proj.weight", layer);
        tw = must_get_weight(ws, name);
        bf16_linear((bf16_t *)mlp_down_buf, (const bf16_t *)mlp_gate_buf, (const bf16_t *)tw->data,
                    seq_len, hidden, cfg->dec_ffn, handle, exec_stream);
        if (layer + 1 < cfg->dec_layers) {
            snprintf(name, sizeof(name), "thinker.model.layers.%d.input_layernorm.weight", layer + 1);
            tw = must_get_weight(ws, name);
            fused_add_residual_rmsnorm((bf16_t *)norm_buf, d_hidden_states, (const bf16_t *)mlp_down_buf,
                                       (const bf16_t *)tw->data,
                                       hidden, seq_len, exec_stream);
        } else {
            add_residual(d_hidden_states, d_hidden_states, (const bf16_t *)mlp_down_buf,
                         total_hidden, exec_stream);
        }
        if (g_dump_decoder_dir) {
            static char layer_name[32];
            snprintf(layer_name, sizeof(layer_name), "dec_layer%02d", layer);
            dump_bf16_dec(layer_name, d_hidden_states, (size_t) total_hidden, exec_stream);
        }
    }

    if (g_dump_decoder_dir) {
        dump_bf16_dec("dec_input_embeds", input_embeds, (size_t) total_hidden, exec_stream);
    }

    {
        const Tensor *tw = must_get_weight(ws, "thinker.model.norm.weight");
        rmsnorm((bf16_t *)norm_buf, d_hidden_states,
                (const bf16_t *)tw->data,
                hidden, seq_len, exec_stream);
    }

    if (g_dump_decoder_dir) {
        dump_bf16_dec("dec_final_norm", norm_buf, (size_t) total_hidden, exec_stream);
    }

    {
        const Tensor *tw = must_get_weight(ws, "thinker.lm_head.weight");
        bf16_linear(logits, (const bf16_t *)norm_buf, (const bf16_t *)tw->data,
                    seq_len, cfg->classify_num, hidden, handle, exec_stream);
    }

    if (g_dump_decoder_dir) {
        dump_bf16_dec("dec_logits", logits, (size_t) seq_len * cfg->classify_num, exec_stream);
    }
}
