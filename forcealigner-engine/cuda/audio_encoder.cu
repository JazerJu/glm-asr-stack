#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "kernels.cuh"
#include "types.h"

extern "C" void cublas_restore_workspace(void *handle);

static const char *g_dump_encoder_dir = NULL;

static void dump_bf16(const char *name, const bf16_t *dev_ptr, size_t count, cudaStream_t stream) {
    if (!g_dump_encoder_dir) return;
    static char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", g_dump_encoder_dir, name);
    void *host = malloc(count * sizeof(bf16_t));
    if (!host) return;
    CUDA_CHECK(cudaMemcpyAsync(host, dev_ptr, count * sizeof(bf16_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(host, sizeof(bf16_t), count, f); fclose(f); }
    free(host);
}

#define AUDIO_LN_EPS 1.0e-5f
#define WARP_SIZE 32
#define AUDIO_ENC_FA_HEAD_DIM 64
#define AUDIO_ENC_FA_HEAD_PAIRS (AUDIO_ENC_FA_HEAD_DIM / 2)
#define AUDIO_ENC_FA_HEAD_OCTETS (AUDIO_ENC_FA_HEAD_PAIRS / 4)
#define AUDIO_ENC_FA_Q32_TILE 32
#define AUDIO_ENC_FA_Q32_ROW_BLOCK 16
#define AUDIO_ENC_FA_PV_MMA_SCORE_TILE 16
#define AUDIO_ENC_FA_PV_MMA_N_TILE 16
#define AUDIO_ENC_FA_PV_MMA_N_WARPS (AUDIO_ENC_FA_HEAD_DIM / AUDIO_ENC_FA_PV_MMA_N_TILE)
#define AUDIO_ENC_FA_Q32_PV_WARPS (2 * AUDIO_ENC_FA_PV_MMA_N_WARPS)
#define AUDIO_ENC_QK_MMA_TILE 16
#define AUDIO_ENC_FA_Q32_THREADS (8 * WARP_SIZE)
#define AUDIO_ENC_FA_Q32_K32_K_TILE 32
#define AUDIO_ENC_FA_Q32_K32_V_PAD 4
#define AUDIO_ENC_FA_PV_MMA_K_STRIDE (AUDIO_ENC_FA_HEAD_DIM + 8)
#define AUDIO_ENC_FA_Q32_K32_K_STRIDE AUDIO_ENC_FA_PV_MMA_K_STRIDE
#define AUDIO_ENC_FA_Q32_K32_PROB_STRIDE (AUDIO_ENC_FA_Q32_K32_K_TILE + 8)

static bf16_t *d_audio_pos_emb = NULL;
static int g_audio_pos_rows = 0;
static int g_audio_pos_dim = 0;

static bf16_t *d_conv_a = NULL;
static size_t g_conv_a_elems = 0;
static bf16_t *d_conv_b = NULL;
static size_t g_conv_b_elems = 0;
static bf16_t *d_conv_chunk = NULL;
static size_t g_conv_chunk_elems = 0;
static bf16_t *d_conv_cols = NULL;
static size_t g_conv_cols_elems = 0;
static bf16_t *d_conv_flat = NULL;
static size_t g_conv_flat_elems = 0;

static bf16_t *d_hidden = NULL;
static size_t g_hidden_elems = 0;
static bf16_t *d_norm = NULL;
static size_t g_norm_elems = 0;
static bf16_t *d_q = NULL;
static size_t g_q_elems = 0;
static bf16_t *d_k = NULL;
static size_t g_k_elems = 0;
static bf16_t *d_v = NULL;
static size_t g_v_elems = 0;
static bf16_t *d_qkv = NULL;
static size_t g_qkv_elems = 0;
static bf16_t *d_attn_merge = NULL;
static size_t g_attn_merge_elems = 0;
static bf16_t *d_ffn = NULL;
static size_t g_ffn_elems = 0;
static bf16_t *d_proj = NULL;
static size_t g_proj_elems = 0;
static bf16_t *d_qk_buf = NULL;
static size_t g_qk_buf_elems = 0;

static int g_use_cublas_attention = -1; /* -1=uninitialized, 0=wmma, 1=cublas */

static int get_use_cublas_attention(void) {
    if (g_use_cublas_attention < 0) {
        const char *env = getenv("FA_FORCEALIGNER_USE_CUBLAS");
        g_use_cublas_attention = (env && atoi(env)) ? 1 : 0;
    }
    return g_use_cublas_attention;
}

static void init_dump_dir(void) {
    if (!g_dump_encoder_dir) {
        g_dump_encoder_dir = getenv("FA_DUMP_ENCODER_DIR");
        if (g_dump_encoder_dir) {
            mkdir(g_dump_encoder_dir, 0755);
            fprintf(stderr, "[dump] Encoder dumps -> %s\n", g_dump_encoder_dir);
        }
    }
}

static __device__ __forceinline__ float audio_encoder_fa_fast_exp(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
#else
    return __expf(x);
#endif
}

__global__ __launch_bounds__(AUDIO_ENC_FA_Q32_THREADS, 4)
static void flash_attention_q32_k32_kernel(
    const bf16_t *q,
    const bf16_t *k,
    const bf16_t *v,
    bf16_t *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads
) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 800)
    (void) q;
    (void) k;
    (void) v;
    (void) out;
    (void) seq_len;
    (void) hidden_dim;
    (void) q_stride;
    (void) k_stride;
    (void) v_stride;
    (void) num_heads;
#else
    __shared__ __align__(16) bf16_t sh_q[AUDIO_ENC_FA_Q32_TILE][AUDIO_ENC_FA_HEAD_DIM];
    __shared__ __align__(16) bf16_t sh_k[AUDIO_ENC_FA_Q32_K32_K_TILE][AUDIO_ENC_FA_Q32_K32_K_STRIDE];
    __shared__ __align__(16) __nv_bfloat162 sh_v[AUDIO_ENC_FA_Q32_K32_K_TILE][AUDIO_ENC_FA_HEAD_PAIRS + AUDIO_ENC_FA_Q32_K32_V_PAD];
    __shared__ __align__(16) bf16_t sh_prob[AUDIO_ENC_FA_Q32_TILE][AUDIO_ENC_FA_Q32_K32_PROB_STRIDE];
    __shared__ float sh_m[AUDIO_ENC_FA_Q32_TILE];
    __shared__ float sh_l[AUDIO_ENC_FA_Q32_TILE];
    __shared__ float sh_alpha[AUDIO_ENC_FA_Q32_TILE];

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int q_start = blockIdx.x * AUDIO_ENC_FA_Q32_TILE;
    const int head = blockIdx.y;
    const int v_ld = (AUDIO_ENC_FA_HEAD_PAIRS + AUDIO_ENC_FA_Q32_K32_V_PAD) * 2;

    if (head >= num_heads) {
        return;
    }

    for (int idx = threadIdx.x; idx < AUDIO_ENC_FA_Q32_TILE * AUDIO_ENC_FA_HEAD_DIM; idx += blockDim.x) {
        const int tq = idx / AUDIO_ENC_FA_HEAD_DIM;
        const int td = idx % AUDIO_ENC_FA_HEAD_DIM;
        const int q_row = q_start + tq;
        bf16_t q_val = __float2bfloat16(0.0f);
        if (q_row < seq_len) {
            const float qf = __bfloat162float(q[(size_t) q_row * q_stride + head * AUDIO_ENC_FA_HEAD_DIM + td]);
            q_val = __float2bfloat16_rn(qf * 0.125f);
        }
        sh_q[tq][td] = q_val;
    }
    if (threadIdx.x < AUDIO_ENC_FA_Q32_TILE) {
        sh_m[threadIdx.x] = -INFINITY;
        sh_l[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                           AUDIO_ENC_FA_Q32_ROW_BLOCK,
                           AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                           AUDIO_ENC_QK_MMA_TILE,
                           bf16_t,
                           nvcuda::wmma::row_major> q_frag[AUDIO_ENC_FA_HEAD_DIM / AUDIO_ENC_QK_MMA_TILE];
    if (warp < 2) {
        const int row_base = warp * AUDIO_ENC_FA_Q32_ROW_BLOCK;
#pragma unroll
        for (int d = 0; d < AUDIO_ENC_FA_HEAD_DIM; d += AUDIO_ENC_QK_MMA_TILE) {
            nvcuda::wmma::load_matrix_sync(q_frag[d / AUDIO_ENC_QK_MMA_TILE],
                                           &sh_q[row_base][d],
                                           AUDIO_ENC_FA_HEAD_DIM);
        }
    }

    const int pv_row_base = (warp / AUDIO_ENC_FA_PV_MMA_N_WARPS) * AUDIO_ENC_FA_Q32_ROW_BLOCK;
    const int pv_n_col = (warp & (AUDIO_ENC_FA_PV_MMA_N_WARPS - 1)) * AUDIO_ENC_FA_PV_MMA_N_TILE;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                           AUDIO_ENC_FA_Q32_ROW_BLOCK,
                           AUDIO_ENC_FA_PV_MMA_N_TILE,
                           AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                           float> pv_frag;
    nvcuda::wmma::fill_fragment(pv_frag, 0.0f);

    for (int k_start = 0; k_start < seq_len; k_start += AUDIO_ENC_FA_Q32_K32_K_TILE) {
        int tile_len = seq_len - k_start;
        if (tile_len > AUDIO_ENC_FA_Q32_K32_K_TILE) {
            tile_len = AUDIO_ENC_FA_Q32_K32_K_TILE;
        }
        const int tile_elems = tile_len * AUDIO_ENC_FA_HEAD_OCTETS;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
            const int tk = idx / AUDIO_ENC_FA_HEAD_OCTETS;
            const int td = idx % AUDIO_ENC_FA_HEAD_OCTETS;
            const int k_row = k_start + tk;
            const int bf16_col = td * 8;
            const int pair_col = td * 4;
            const bf16_t *k_ptr = k + (size_t) k_row * k_stride + head * AUDIO_ENC_FA_HEAD_DIM;
            const bf16_t *v_ptr = v + (size_t) k_row * v_stride + head * AUDIO_ENC_FA_HEAD_DIM;
            const uint4 *k_ptr_u4 = reinterpret_cast<const uint4 *>(k_ptr);
            const uint4 *v_ptr_u4 = reinterpret_cast<const uint4 *>(v_ptr);
            bf16_t *sh_k_dst = &sh_k[tk][bf16_col];
            __nv_bfloat162 *sh_v_dst = &sh_v[tk][pair_col];
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned) __cvta_generic_to_shared(sh_k_dst)), "l"(k_ptr_u4 + td));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"((unsigned) __cvta_generic_to_shared(sh_v_dst)), "l"(v_ptr_u4 + td));
        }
        asm volatile("cp.async.commit_group;\n" : :);
        asm volatile("cp.async.wait_group 0;\n" : :);
        __syncthreads();

        if (warp < 2) {
            const int row_base = warp * AUDIO_ENC_FA_Q32_ROW_BLOCK;
            const int lane_group = lane >> 2;
            const unsigned row_group_mask = 0xFu << (lane_group * 4);
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   AUDIO_ENC_FA_Q32_ROW_BLOCK,
                                   AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                                   AUDIO_ENC_QK_MMA_TILE,
                                   bf16_t,
                                   nvcuda::wmma::col_major> b_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,
                                   AUDIO_ENC_FA_Q32_ROW_BLOCK,
                                   AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                                   AUDIO_ENC_QK_MMA_TILE,
                                   float> c_frag[AUDIO_ENC_FA_Q32_K32_K_TILE / AUDIO_ENC_FA_PV_MMA_SCORE_TILE];
#pragma unroll
            for (int sub_idx = 0; sub_idx < AUDIO_ENC_FA_Q32_K32_K_TILE / AUDIO_ENC_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                const int sub = sub_idx * AUDIO_ENC_FA_PV_MMA_SCORE_TILE;
                nvcuda::wmma::fill_fragment(c_frag[sub_idx], 0.0f);
                if (sub >= tile_len) {
                    continue;
                }
                for (int d = 0; d < AUDIO_ENC_FA_HEAD_DIM; d += AUDIO_ENC_QK_MMA_TILE) {
                    nvcuda::wmma::load_matrix_sync(b_frag, &sh_k[sub][d], AUDIO_ENC_FA_Q32_K32_K_STRIDE);
                    nvcuda::wmma::mma_sync(c_frag[sub_idx], q_frag[d / AUDIO_ENC_QK_MMA_TILE], b_frag, c_frag[sub_idx]);
                }
            }

            float row_max_low = -INFINITY;
            float row_max_high = -INFINITY;
#pragma unroll
            for (int sub_idx = 0; sub_idx < AUDIO_ENC_FA_Q32_K32_K_TILE / AUDIO_ENC_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                const int sub = sub_idx * AUDIO_ENC_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    const int tk = sub + local_col;
                    const float score = (tk < tile_len) ? c_frag[sub_idx].x[frag_idx] : -INFINITY;
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
            for (int sub_idx = 0; sub_idx < AUDIO_ENC_FA_Q32_K32_K_TILE / AUDIO_ENC_FA_PV_MMA_SCORE_TILE; sub_idx++) {
                const int sub = sub_idx * AUDIO_ENC_FA_PV_MMA_SCORE_TILE;
#pragma unroll
                for (int frag_idx = 0; frag_idx < c_frag[sub_idx].num_elements; frag_idx++) {
                    const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
                    const int tk = sub + local_col;
                    float p = 0.0f;
                    if (tk < tile_len) {
                        const float new_m = (local_row < 8) ? new_m_low : new_m_high;
                        p = audio_encoder_fa_fast_exp(c_frag[sub_idx].x[frag_idx] - new_m);
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
                const float alpha_low = audio_encoder_fa_fast_exp(old_m_low - new_m_low);
                const float alpha_high = audio_encoder_fa_fast_exp(old_m_high - new_m_high);
                sh_m[row_base + lane_group] = new_m_low;
                sh_l[row_base + lane_group] = old_l_low * alpha_low + p_sum_low;
                sh_alpha[row_base + lane_group] = alpha_low;
                sh_m[row_base + lane_group + 8] = new_m_high;
                sh_l[row_base + lane_group + 8] = old_l_high * alpha_high + p_sum_high;
                sh_alpha[row_base + lane_group + 8] = alpha_high;
            }
        }
        __syncthreads();

        if (warp < AUDIO_ENC_FA_Q32_PV_WARPS) {
            const bf16_t *sh_v_bf16 = reinterpret_cast<const bf16_t *>(&sh_v[0][0]);
            if (k_start != 0) {
                for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
                    const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
                    pv_frag.x[frag_idx] *= sh_alpha[pv_row_base + local_row];
                }
            }
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,
                                   AUDIO_ENC_FA_Q32_ROW_BLOCK,
                                   AUDIO_ENC_FA_PV_MMA_N_TILE,
                                   AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                                   bf16_t,
                                   nvcuda::wmma::row_major> p_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,
                                   AUDIO_ENC_FA_Q32_ROW_BLOCK,
                                   AUDIO_ENC_FA_PV_MMA_N_TILE,
                                   AUDIO_ENC_FA_PV_MMA_SCORE_TILE,
                                   bf16_t,
                                   nvcuda::wmma::row_major> v_frag;
            for (int sub = 0; sub < AUDIO_ENC_FA_Q32_K32_K_TILE; sub += AUDIO_ENC_FA_PV_MMA_SCORE_TILE) {
                nvcuda::wmma::load_matrix_sync(p_frag, &sh_prob[pv_row_base][sub], AUDIO_ENC_FA_Q32_K32_PROB_STRIDE);
                nvcuda::wmma::load_matrix_sync(v_frag, &sh_v_bf16[sub * v_ld + pv_n_col], v_ld);
                nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < AUDIO_ENC_FA_Q32_TILE) {
        sh_alpha[threadIdx.x] = __frcp_rn(sh_l[threadIdx.x]);
    }
    __syncthreads();

    if (warp < AUDIO_ENC_FA_Q32_PV_WARPS) {
        for (int frag_idx = 0; frag_idx < pv_frag.num_elements; frag_idx++) {
            const int local_row = (lane >> 2) + ((frag_idx & 2) ? 8 : 0);
            const int local_col = ((lane & 3) << 1) + (frag_idx & 1) + ((frag_idx & 4) ? 8 : 0);
            const int row = pv_row_base + local_row;
            const int col = pv_n_col + local_col;
            const int q_idx = q_start + row;
            if (q_idx < seq_len) {
                out[(size_t) q_idx * hidden_dim + head * AUDIO_ENC_FA_HEAD_DIM + col] =
                    __float2bfloat16_rn(pv_frag.x[frag_idx] * sh_alpha[row]);
            }
        }
    }
#endif
}

// Simple non-fused attention for tiny chunks where WMMA tile sizes exceed seq_len
__global__ static void tiny_attention_kernel(
    const bf16_t *q,
    const bf16_t *k,
    const bf16_t *v,
    bf16_t *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads,
    int head_dim
) {
    int total = seq_len * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int col = idx % head_dim;
    int t = idx / head_dim;
    int head = t % num_heads;
    int qi = t / num_heads;

    if (seq_len == 1) {
        // Self-attention of 1 token: output = V (identity)
        float vv = __bfloat162float(v[head * head_dim + col]);
        out[(size_t)qi * hidden_dim + head * head_dim + col] = __float2bfloat16(vv);
    } else {
        float total_sum = 0.0f;
        float acc = 0.0f;
        for (int kj = 0; kj < seq_len; kj++) {
            float qk = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float qv = __bfloat162float(q[(size_t)qi * q_stride + head * head_dim + d]);
                float kv = __bfloat162float(k[(size_t)kj * k_stride + head * head_dim + d]);
                qk += qv * kv;
            }
            float score = expf(qk * 0.125f);
            float vv = __bfloat162float(v[(size_t)kj * v_stride + head * head_dim + col]);
            acc += score * vv;
            total_sum += score;
        }
        out[(size_t)qi * hidden_dim + head * head_dim + col] = __float2bfloat16(acc / (total_sum + 1e-8f));
    }
}

static void flash_attention_q32_k32(
    const bf16_t *q,
    const bf16_t *k,
    const bf16_t *v,
    bf16_t *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    CHECK(q != NULL && k != NULL && v != NULL && out != NULL, "audio encoder FA received null input");
    CHECK(head_dim == AUDIO_ENC_FA_HEAD_DIM, "audio encoder FA expects head_dim=64, got %d", head_dim);
    CHECK(seq_len > 0 && hidden_dim > 0 && num_heads > 0, "audio encoder FA invalid shape seq=%d hidden=%d heads=%d", seq_len, hidden_dim, num_heads);
    CHECK(q_stride >= hidden_dim && k_stride >= hidden_dim && v_stride >= hidden_dim,
          "audio encoder FA invalid strides q=%d k=%d v=%d hidden=%d", q_stride, k_stride, v_stride, hidden_dim);

    // Fallback to tiny attention kernel for chunks smaller than the WMMA tile size
    if (seq_len < 8) {
        int total = seq_len * num_heads * head_dim;
        tiny_attention_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            q, k, v, out, seq_len, hidden_dim, q_stride, k_stride, v_stride, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 block(AUDIO_ENC_FA_Q32_THREADS);
    const dim3 grid((seq_len + AUDIO_ENC_FA_Q32_TILE - 1) / AUDIO_ENC_FA_Q32_TILE, num_heads);
    flash_attention_q32_k32_kernel<<<grid, block, 0, stream>>>(
        q, k, v, out, seq_len, hidden_dim, q_stride, k_stride, v_stride, num_heads);
    CUDA_CHECK(cudaGetLastError());
}

static bf16_t **g_fused_qkv_weights = NULL;
static bf16_t **g_fused_qkv_biases = NULL;
static const void **g_fused_q_src = NULL;
static const void **g_fused_k_src = NULL;
static const void **g_fused_v_src = NULL;
static const void **g_fused_q_bias_src = NULL;
static const void **g_fused_k_bias_src = NULL;
static const void **g_fused_v_bias_src = NULL;
static int g_fused_qkv_layers = 0;
static int g_fused_qkv_hidden = 0;

static void ensure_bf16_buffer(bf16_t **ptr, size_t *capacity_elems, size_t needed_elems) {
    if (needed_elems == 0) {
        return;
    }
    if (*capacity_elems >= needed_elems && *ptr != NULL) {
        return;
    }
    if (*ptr != NULL) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = NULL;
        *capacity_elems = 0;
    }
    CUDA_CHECK(cudaMalloc(ptr, needed_elems * sizeof(bf16_t)));
    *capacity_elems = needed_elems;
}

static void free_fused_qkv_cache(void) {
    if (g_fused_qkv_weights != NULL) {
        for (int i = 0; i < g_fused_qkv_layers; i++) {
            if (g_fused_qkv_weights[i] != NULL) {
                CUDA_CHECK(cudaFree(g_fused_qkv_weights[i]));
            }
            if (g_fused_qkv_biases[i] != NULL) {
                CUDA_CHECK(cudaFree(g_fused_qkv_biases[i]));
            }
        }
        free(g_fused_qkv_weights);
        free(g_fused_qkv_biases);
        free(g_fused_q_src);
        free(g_fused_k_src);
        free(g_fused_v_src);
        free(g_fused_q_bias_src);
        free(g_fused_k_bias_src);
        free(g_fused_v_bias_src);
    }

    g_fused_qkv_weights = NULL;
    g_fused_qkv_biases = NULL;
    g_fused_q_src = NULL;
    g_fused_k_src = NULL;
    g_fused_v_src = NULL;
    g_fused_q_bias_src = NULL;
    g_fused_k_bias_src = NULL;
    g_fused_v_bias_src = NULL;
    g_fused_qkv_layers = 0;
    g_fused_qkv_hidden = 0;
}

static void ensure_fused_qkv_cache_storage(int layers, int hidden) {
    if (g_fused_qkv_weights != NULL && g_fused_qkv_layers == layers && g_fused_qkv_hidden == hidden) {
        return;
    }

    free_fused_qkv_cache();

    g_fused_qkv_weights = (bf16_t **) calloc((size_t) layers, sizeof(bf16_t *));
    g_fused_qkv_biases = (bf16_t **) calloc((size_t) layers, sizeof(bf16_t *));
    g_fused_q_src = (const void **) calloc((size_t) layers, sizeof(void *));
    g_fused_k_src = (const void **) calloc((size_t) layers, sizeof(void *));
    g_fused_v_src = (const void **) calloc((size_t) layers, sizeof(void *));
    g_fused_q_bias_src = (const void **) calloc((size_t) layers, sizeof(void *));
    g_fused_k_bias_src = (const void **) calloc((size_t) layers, sizeof(void *));
    g_fused_v_bias_src = (const void **) calloc((size_t) layers, sizeof(void *));

    CHECK(g_fused_qkv_weights != NULL, "failed to allocate fused QKV weight pointer cache");
    CHECK(g_fused_qkv_biases != NULL, "failed to allocate fused QKV bias pointer cache");
    CHECK(g_fused_q_src != NULL && g_fused_k_src != NULL && g_fused_v_src != NULL,
          "failed to allocate fused QKV source cache");
    CHECK(g_fused_q_bias_src != NULL && g_fused_k_bias_src != NULL && g_fused_v_bias_src != NULL,
          "failed to allocate fused QKV bias source cache");

    g_fused_qkv_layers = layers;
    g_fused_qkv_hidden = hidden;
}

static void ensure_fused_qkv_layer(
    int layer,
    int hidden,
    const Tensor *q_w,
    const Tensor *q_b,
    const Tensor *k_w,
    const Tensor *k_b,
    const Tensor *v_w,
    const Tensor *v_b,
    cudaStream_t stream
) {
    const size_t weight_elems = (size_t) 3 * hidden * hidden;
    const size_t proj_weight_elems = (size_t) hidden * hidden;
    const size_t bias_elems = (size_t) 3 * hidden;
    const size_t proj_bias_elems = (size_t) hidden;
    const size_t proj_weight_bytes = proj_weight_elems * sizeof(bf16_t);
    const size_t proj_bias_bytes = proj_bias_elems * sizeof(bf16_t);
    const bool need_refresh =
        g_fused_qkv_weights[layer] == NULL ||
        g_fused_qkv_biases[layer] == NULL ||
        g_fused_q_src[layer] != q_w->data ||
        g_fused_k_src[layer] != k_w->data ||
        g_fused_v_src[layer] != v_w->data ||
        g_fused_q_bias_src[layer] != q_b->data ||
        g_fused_k_bias_src[layer] != k_b->data ||
        g_fused_v_bias_src[layer] != v_b->data;

    CHECK(tensor_numel(q_w) == (int64_t) proj_weight_elems, "unexpected q_proj.weight size in layer %d", layer);
    CHECK(tensor_numel(k_w) == (int64_t) proj_weight_elems, "unexpected k_proj.weight size in layer %d", layer);
    CHECK(tensor_numel(v_w) == (int64_t) proj_weight_elems, "unexpected v_proj.weight size in layer %d", layer);
    CHECK(tensor_numel(q_b) == (int64_t) proj_bias_elems, "unexpected q_proj.bias size in layer %d", layer);
    CHECK(tensor_numel(k_b) == (int64_t) proj_bias_elems, "unexpected k_proj.bias size in layer %d", layer);
    CHECK(tensor_numel(v_b) == (int64_t) proj_bias_elems, "unexpected v_proj.bias size in layer %d", layer);

    if (g_fused_qkv_weights[layer] == NULL) {
        CUDA_CHECK(cudaMalloc(&g_fused_qkv_weights[layer], weight_elems * sizeof(bf16_t)));
    }
    if (g_fused_qkv_biases[layer] == NULL) {
        CUDA_CHECK(cudaMalloc(&g_fused_qkv_biases[layer], bias_elems * sizeof(bf16_t)));
    }
    if (!need_refresh) {
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_weights[layer], q_w->data,
                               proj_weight_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_weights[layer] + proj_weight_elems, k_w->data,
                               proj_weight_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_weights[layer] + proj_weight_elems * 2, v_w->data,
                               proj_weight_bytes, cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_biases[layer], q_b->data,
                               proj_bias_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_biases[layer] + proj_bias_elems, k_b->data,
                               proj_bias_bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(g_fused_qkv_biases[layer] + proj_bias_elems * 2, v_b->data,
                               proj_bias_bytes, cudaMemcpyDeviceToDevice, stream));

    g_fused_q_src[layer] = q_w->data;
    g_fused_k_src[layer] = k_w->data;
    g_fused_v_src[layer] = v_w->data;
    g_fused_q_bias_src[layer] = q_b->data;
    g_fused_k_bias_src[layer] = k_b->data;
    g_fused_v_bias_src[layer] = v_b->data;
}

static int conv_out_size(int in_size, int kernel, int stride, int padding) {
    return (in_size + 2 * padding - kernel) / stride + 1;
}

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static const Tensor *require_weight(const WeightStore *ws, const char *name) {
    const Tensor *t = ws_get_const(ws, name);
    CHECK(t != NULL, "missing audio encoder weight: %s", name);
    CHECK(t->data != NULL, "weight has no device data: %s", name);
    return t;
}

static void make_layer_weight_name(char *out, size_t out_size, int layer, const char *suffix) {
    snprintf(out, out_size, "thinker.audio_tower.layers.%d.%s", layer, suffix);
}

static void ensure_position_embeddings(const AlignerConfig *cfg, cudaStream_t stream) {
    int rows = cfg->max_source_positions;
    int dim = cfg->enc_d_model;
    int half = dim / 2;
    float log_increment;
    bf16_t *host;

    CHECK(dim % 2 == 0, "audio position embedding dim must be even, got %d", dim);
    if (d_audio_pos_emb != NULL && g_audio_pos_rows == rows && g_audio_pos_dim == dim) {
        return;
    }
    if (d_audio_pos_emb != NULL) {
        CUDA_CHECK(cudaFree(d_audio_pos_emb));
        d_audio_pos_emb = NULL;
    }

    host = (bf16_t *) malloc((size_t) rows * (size_t) dim * sizeof(bf16_t));
    CHECK(host != NULL, "failed to allocate host positional embedding buffer");
    log_increment = logf(10000.0f) / (float) (half - 1);
    for (int pos = 0; pos < rows; pos++) {
        for (int i = 0; i < half; i++) {
            float inv_timescale = expf(-log_increment * (float) i);
            float v = (float) pos * inv_timescale;
            host[(size_t) pos * dim + i] = __float2bfloat16(sinf(v));
            host[(size_t) pos * dim + half + i] = __float2bfloat16(cosf(v));
        }
    }

    CUDA_CHECK(cudaMalloc(&d_audio_pos_emb, (size_t) rows * (size_t) dim * sizeof(bf16_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_audio_pos_emb, host,
                               (size_t) rows * (size_t) dim * sizeof(bf16_t),
                               cudaMemcpyHostToDevice, stream));
    free(host);
    g_audio_pos_rows = rows;
    g_audio_pos_dim = dim;
}

__global__ static void im2col_2d_kernel(
    bf16_t *cols,
    const bf16_t *input,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_spatial = out_h * out_w;
    int K = channels * kernel_h * kernel_w;
    int total = out_spatial * K;
    if (idx >= total) {
        return;
    }

    int k_idx = idx % K;
    int out_idx = idx / K;
    int ow = out_idx % out_w;
    int oh = out_idx / out_w;

    int kw = k_idx % kernel_w;
    int kh = (k_idx / kernel_w) % kernel_h;
    int c = k_idx / (kernel_h * kernel_w);

    int ih = oh * stride + kh - padding;
    int iw = ow * stride + kw - padding;
    float v = 0.0f;

    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
        int in_offset = (c * in_h + ih) * in_w + iw;
        v = __bfloat162float(input[in_offset]);
    }
    cols[idx] = __float2bfloat16(v);
}

__global__ static void conv_flat_to_chw_kernel(
    bf16_t *out,
    const bf16_t *flat,
    int out_channels,
    int out_h,
    int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_channels * out_h * out_w;
    if (idx >= total) {
        return;
    }
    int ow = idx % out_w;
    int t = idx / out_w;
    int oh = t % out_h;
    int oc = t / out_h;
    int flat_row = oh * out_w + ow;
    out[idx] = flat[flat_row * out_channels + oc];
}

__global__ static void flatten_conv3_kernel(
    bf16_t *out,
    const bf16_t *in,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feature_dim = channels * height;
    int total = width * feature_dim;
    if (idx >= total) {
        return;
    }
    int f = idx % feature_dim;
    int w = idx / feature_dim;
    int h = f % height;
    int c = f / height;
    out[idx] = in[(c * height + h) * width + w];
}

__global__ static void add_position_embedding_kernel(bf16_t *x, const bf16_t *pos, int rows, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * dim;
    if (idx >= total) {
        return;
    }
    int row = idx / dim;
    int col = idx % dim;
    float xv = __bfloat162float(x[idx]);
    float pv = __bfloat162float(pos[row * dim + col]);
    x[idx] = __float2bfloat16(xv + pv);
}

__global__ static void split_qkv_kernel(bf16_t *q, bf16_t *k, bf16_t *v, const bf16_t *qkv, int seq_len, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden;
    if (idx >= total) {
        return;
    }
    int s = idx / hidden;
    int d = idx % hidden;
    size_t base = (size_t) s * (size_t) (3 * hidden) + d;
    q[idx] = qkv[base];
    k[idx] = qkv[base + hidden];
    v[idx] = qkv[base + 2 * hidden];
}

static void conv2d_forward(
    bf16_t *out,
    const bf16_t *input,
    const bf16_t *weight,
    const bf16_t *bias,
    int in_channels,
    int out_channels,
    int in_h,
    int in_w,
    int stride,
    int padding,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    int kernel_h = 3;
    int kernel_w = 3;
    int out_h = conv_out_size(in_h, kernel_h, stride, padding);
    int out_w = conv_out_size(in_w, kernel_w, stride, padding);
    int out_spatial = out_h * out_w;
    int K = in_channels * kernel_h * kernel_w;
    int total_cols = out_spatial * K;
    int total_out = out_channels * out_h * out_w;

    ensure_bf16_buffer(&d_conv_cols, &g_conv_cols_elems, (size_t) total_cols);
    ensure_bf16_buffer(&d_conv_flat, &g_conv_flat_elems, (size_t) out_spatial * out_channels);

    im2col_2d_kernel<<<(total_cols + 255) / 256, 256, 0, stream>>>(
        d_conv_cols, input, in_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride, padding);
    CUDA_CHECK(cudaGetLastError());

    if (bias != nullptr) {
        bf16_linear_bias(d_conv_flat, d_conv_cols, weight, bias,
                         out_spatial, out_channels, K, handle, stream);
    } else {
        bf16_linear(d_conv_flat, d_conv_cols, weight, out_spatial, out_channels, K, handle, stream);
    }

    conv_flat_to_chw_kernel<<<(total_out + 255) / 256, 256, 0, stream>>>(
        out, d_conv_flat, out_channels, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
}

/* cuBLAS GEMM-based attention: QK^T / sqrt(d) -> softmax -> PV
 * Drop-in replacement for flash_attention_q32_k32 using cuBLAS batched GEMM.
 * Q, K, V are [seq_len, hidden_dim] row-major with per-head packing:
 *   row * stride + head * head_dim + d
 * This matches the cuBLAS fallback in Infer-engine encoder.cu. */
static void cublas_attention(
    const bf16_t *q,
    const bf16_t *k,
    const bf16_t *v,
    bf16_t *out,
    int seq_len,
    int hidden_dim,
    int q_stride,
    int k_stride,
    int v_stride,
    int num_heads,
    int head_dim,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    CHECK(q != NULL && k != NULL && v != NULL && out != NULL, "cublas_attention received null input");
    CHECK(seq_len > 0 && hidden_dim > 0 && num_heads > 0, "cublas_attention invalid shape");
    CHECK(q_stride == hidden_dim && k_stride == hidden_dim && v_stride == hidden_dim,
          "cublas_attention requires contiguous strides (stride==hidden), got q=%d k=%d v=%d hidden=%d",
          q_stride, k_stride, v_stride, hidden_dim);

    /* Allocate QK score buffer: [num_heads, seq_len, seq_len] */
    size_t qk_elems = (size_t) num_heads * seq_len * seq_len;
    ensure_bf16_buffer(&d_qk_buf, &g_qk_buf_elems, qk_elems);

    float alpha = 1.0f / sqrtf((float) head_dim);
    float beta = 0.0f;
    float one = 1.0f;

    /* QK^T: [seq_len, seq_len] = alpha * K^T @ Q per head
     * cuBLAS column-major: K^T has shape [seq_len, head_dim], Q has shape [head_dim, seq_len]
     * K is stored row-major [seq_len, hidden] → in col-major [hidden, seq_len], leading dim = hidden
     * For per-head slice: lda = hidden (row stride), strideA = head_dim (actual rows per head) */
    cublasStatus_t st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T,  /* op(A) = K^T → [head_dim, seq_len] -> [seq_len, head_dim] */
        CUBLAS_OP_N,  /* op(B) = Q   → [head_dim, seq_len] */
        seq_len, seq_len, head_dim,
        &alpha,
        k, CUDA_R_16BF, k_stride, head_dim,
        q, CUDA_R_16BF, q_stride, head_dim,
        &beta,
        d_qk_buf, CUDA_R_16BF, seq_len, (long long) seq_len * seq_len,
        num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublas_attention QK GEMM failed: %d", (int) st);

    /* Softmax over last dim (key dimension) for each head */
    softmax_inplace(d_qk_buf, num_heads * seq_len, seq_len, stream);

    /* PV: output[head_dim, seq_len] = V[head_dim, seq_len] @ scores[seq_len, seq_len]
     * V stored row-major [seq_len, hidden] → col-major [hidden, seq_len], leading dim = hidden
     * Per-head slice: lda = hidden, actual rows per head = head_dim
     * Scores: [seq_len, seq_len], ldc = seq_len, stride = seq_len*seq_len
     * Output: col-major [hidden, seq_len], ldc = hidden, stride = hidden*seq_len
     *   → when read back row-major: [seq_len, hidden] which matches our layout */
    st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        head_dim, seq_len, seq_len,
        &one,
        v, CUDA_R_16BF, v_stride, head_dim,
        d_qk_buf, CUDA_R_16BF, seq_len, (long long) seq_len * seq_len,
        &beta,
        out, CUDA_R_16BF, hidden_dim, (long long) hidden_dim * seq_len,
        num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
    );
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublas_attention PV GEMM failed: %d", (int) st);

    /* Restore cuBLAS stream after batched GEMM (sm_120 cublasGemmStridedBatchedEx bug workaround) */
    cublasSetStream(handle, stream);
}

static void encoder_attention(
    bf16_t *out,
    const bf16_t *x,
    const bf16_t *residual,
    int layer,
    const Tensor *q_w,
    const Tensor *q_b,
    const Tensor *k_w,
    const Tensor *k_b,
    const Tensor *v_w,
    const Tensor *v_b,
    const Tensor *o_w,
    const Tensor *o_b,
    const AlignerConfig *cfg,
    int seq_len,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    int hidden = cfg->enc_d_model;
    int heads = cfg->enc_heads;
    int head_dim = cfg->enc_head_dim;
    int hidden_elems = seq_len * hidden;
    int qkv_elems = seq_len * hidden * 3;

    ensure_fused_qkv_cache_storage(cfg->enc_layers, hidden);
    ensure_fused_qkv_layer(layer, hidden, q_w, q_b, k_w, k_b, v_w, v_b, stream);

    ensure_bf16_buffer(&d_q, &g_q_elems, (size_t) hidden_elems);
    ensure_bf16_buffer(&d_k, &g_k_elems, (size_t) hidden_elems);
    ensure_bf16_buffer(&d_v, &g_v_elems, (size_t) hidden_elems);
    ensure_bf16_buffer(&d_qkv, &g_qkv_elems, (size_t) qkv_elems);
    ensure_bf16_buffer(&d_attn_merge, &g_attn_merge_elems, (size_t) hidden_elems);

    bf16_linear_bias(d_qkv, x, g_fused_qkv_weights[layer], g_fused_qkv_biases[layer],
                     seq_len, hidden * 3, hidden, handle, stream);

    split_qkv_kernel<<<(hidden_elems + 255) / 256, 256, 0, stream>>>(d_q, d_k, d_v, d_qkv, seq_len, hidden);
    CUDA_CHECK(cudaGetLastError());

    if (get_use_cublas_attention()) {
        cublas_attention(d_q, d_k, d_v, d_attn_merge,
                         seq_len, hidden, hidden, hidden, hidden,
                         heads, head_dim, handle, stream);
    } else {
        flash_attention_q32_k32(d_q, d_k, d_v, d_attn_merge,
                                seq_len, hidden, hidden, hidden, hidden,
                                heads, head_dim, stream);
    }

    bf16_linear_bias_residual(out, d_attn_merge, (const bf16_t *) o_w->data, (const bf16_t *) o_b->data,
                              residual, seq_len, hidden, hidden, handle, stream);
}

extern "C" {

void cuda_audio_encoder_forward(
    bf16_t *output,
    const bf16_t *mel,
    int mel_T,
    const WeightStore *ws,
    const AlignerConfig *cfg,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    init_dump_dir();
    int conv1_h;
    int conv1_w;
    int conv2_h;
    int conv2_w;
    int conv3_h;
    int conv3_w;
    int chunk_size;
    int chunk_count;
    int conv_out_dim;
    int frontend_rows;
    int infer_ratio;
    int window_aftercnn;
    int infer_starts[32];
    int infer_lens[32];
    int num_infer_chunks;
    int seq_len;
    size_t conv1_elems;
    size_t conv2_elems;
    size_t conv3_elems;
    char name[160];

    CHECK(output != NULL, "audio encoder output is null");
    CHECK(mel != NULL, "audio encoder mel is null");
    CHECK(ws != NULL, "audio encoder weight store is null");
    CHECK(cfg != NULL, "audio encoder config is null");
    CHECK(mel_T > 0, "audio encoder mel_T must be > 0, got %d", mel_T);
    CHECK(cfg->n_window > 0, "audio encoder n_window must be > 0, got %d", cfg->n_window);
    CHECK(cfg->n_window_infer > 0, "audio encoder n_window_infer must be > 0, got %d", cfg->n_window_infer);

    chunk_size = cfg->n_window * 2;
    CHECK(chunk_size > 0, "audio encoder chunk size must be > 0, got %d", chunk_size);
    chunk_count = ceil_div(mel_T, chunk_size);
    CHECK(chunk_count > 0, "audio encoder chunk count must be > 0, got %d", chunk_count);

    conv1_h = conv_out_size(cfg->num_mel_bins, 3, 2, 1);
    conv1_w = conv_out_size(chunk_size, 3, 2, 1);
    conv2_h = conv_out_size(conv1_h, 3, 2, 1);
    conv2_w = conv_out_size(conv1_w, 3, 2, 1);
    conv3_h = conv_out_size(conv2_h, 3, 2, 1);
    conv3_w = conv_out_size(conv2_w, 3, 2, 1);
    conv_out_dim = cfg->downsample_hidden * conv3_h;
    frontend_rows = conv3_w;
    infer_ratio = cfg->n_window_infer / (cfg->n_window * 2);
    window_aftercnn = conv3_w * infer_ratio;
    num_infer_chunks = 0;
    seq_len = 0;

    CHECK(infer_ratio > 0,
          "audio encoder infer ratio must be > 0, got %d from n_window_infer=%d n_window=%d",
          infer_ratio, cfg->n_window_infer, cfg->n_window);
    CHECK(window_aftercnn > 0,
          "audio encoder window_aftercnn must be > 0, got %d", window_aftercnn);

    for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
        int chunk_start = chunk_idx * chunk_size;
        int chunk_len = mel_T - chunk_start;
        int chunk_seq_len;
        if (chunk_len > chunk_size) {
            chunk_len = chunk_size;
        }
        chunk_seq_len = conv_out_size(chunk_len, 3, 2, 1);
        chunk_seq_len = conv_out_size(chunk_seq_len, 3, 2, 1);
        chunk_seq_len = conv_out_size(chunk_seq_len, 3, 2, 1);
        seq_len += chunk_seq_len;
    }

    CHECK(conv3_h > 0 && seq_len > 0, "invalid audio conv output shape H=%d W=%d", conv3_h, seq_len);
    CHECK(seq_len <= cfg->max_source_positions,
          "audio sequence length %d exceeds max_source_positions %d", seq_len, cfg->max_source_positions);

    {
        int off = 0;
        while (off < seq_len) {
            int remaining = seq_len - off;
            int clen = (remaining >= window_aftercnn) ? window_aftercnn : remaining;
            CHECK(num_infer_chunks < (int) (sizeof(infer_starts) / sizeof(infer_starts[0])),
                  "audio encoder infer chunk count exceeds limit %zu",
                  sizeof(infer_starts) / sizeof(infer_starts[0]));
            infer_starts[num_infer_chunks] = off;
            infer_lens[num_infer_chunks] = clen;
            num_infer_chunks++;
            off += clen;
        }
    }

    ensure_bf16_buffer(&d_conv_chunk, &g_conv_chunk_elems, (size_t) cfg->num_mel_bins * chunk_size);
    conv1_elems = (size_t) cfg->downsample_hidden * conv1_h * conv1_w;
    conv2_elems = (size_t) cfg->downsample_hidden * conv2_h * conv2_w;
    conv3_elems = (size_t) cfg->downsample_hidden * conv3_h * conv3_w;

    ensure_bf16_buffer(&d_conv_a, &g_conv_a_elems, conv1_elems > conv3_elems ? conv1_elems : conv3_elems);
    ensure_bf16_buffer(&d_conv_b, &g_conv_b_elems, conv2_elems);
    ensure_bf16_buffer(&d_hidden, &g_hidden_elems, (size_t) seq_len * cfg->enc_d_model);
    ensure_bf16_buffer(&d_norm, &g_norm_elems, (size_t) seq_len * cfg->enc_d_model);
    ensure_bf16_buffer(&d_ffn, &g_ffn_elems, (size_t) seq_len * cfg->enc_ffn_dim);
    ensure_bf16_buffer(&d_proj, &g_proj_elems, (size_t) (frontend_rows > seq_len ? frontend_rows : seq_len) * cfg->enc_d_model);
    ensure_position_embeddings(cfg, stream);

    cublasSetStream(handle, stream);
    cublas_restore_workspace(handle);

    dump_bf16("mel_input", mel, (size_t) cfg->num_mel_bins * mel_T, stream);

    {
        const bf16_t *conv1_wt = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d1.weight")->data;
        const bf16_t *conv1_bias = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d1.bias")->data;
        const bf16_t *conv2_wt = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d2.weight")->data;
        const bf16_t *conv2_bias = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d2.bias")->data;
        const bf16_t *conv3_wt = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d3.weight")->data;
        const bf16_t *conv3_bias = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv2d3.bias")->data;
        const bf16_t *conv_out_wt = (const bf16_t *) require_weight(ws, "thinker.audio_tower.conv_out.weight")->data;
        int seq_offset = 0;

        for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            int chunk_start = chunk_idx * chunk_size;
            int chunk_len = mel_T - chunk_start;
            int chunk_hidden_elems;
            if (chunk_len > chunk_size) {
                chunk_len = chunk_size;
            }

            chunk_hidden_elems = conv_out_size(chunk_len, 3, 2, 1);
            chunk_hidden_elems = conv_out_size(chunk_hidden_elems, 3, 2, 1);
            chunk_hidden_elems = conv_out_size(chunk_hidden_elems, 3, 2, 1);

            CUDA_CHECK(cudaMemsetAsync(d_conv_chunk, 0,
                                       (size_t) cfg->num_mel_bins * chunk_size * sizeof(bf16_t),
                                       stream));
            CUDA_CHECK(cudaMemcpy2DAsync(
                d_conv_chunk,
                (size_t) chunk_size * sizeof(bf16_t),
                mel + chunk_start,
                (size_t) mel_T * sizeof(bf16_t),
                (size_t) chunk_len * sizeof(bf16_t),
                (size_t) cfg->num_mel_bins,
                cudaMemcpyDeviceToDevice,
                stream));

            conv2d_forward(d_conv_a, d_conv_chunk,
                           conv1_wt, conv1_bias,
                           1, cfg->downsample_hidden, cfg->num_mel_bins, chunk_size,
                           2, 1, handle, stream);
            gelu_forward(d_conv_a, d_conv_a, (int) conv1_elems, stream);

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_conv1_gelu", d_conv_a, conv1_elems, stream);
            }

            conv2d_forward(d_conv_b, d_conv_a,
                           conv2_wt, conv2_bias,
                           cfg->downsample_hidden, cfg->downsample_hidden, conv1_h, conv1_w,
                           2, 1, handle, stream);
            gelu_forward(d_conv_b, d_conv_b, (int) conv2_elems, stream);

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_conv2_gelu", d_conv_b, conv2_elems, stream);
            }

            conv2d_forward(d_conv_a, d_conv_b,
                           conv3_wt, conv3_bias,
                           cfg->downsample_hidden, cfg->downsample_hidden, conv2_h, conv2_w,
                           2, 1, handle, stream);
            gelu_forward(d_conv_a, d_conv_a, (int) conv3_elems, stream);

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_conv3_gelu", d_conv_a, conv3_elems, stream);
            }

            flatten_conv3_kernel<<<((size_t) conv3_w * conv_out_dim + 255) / 256, 256, 0, stream>>>(
                d_conv_flat, d_conv_a, cfg->downsample_hidden, conv3_h, conv3_w);
            CUDA_CHECK(cudaGetLastError());

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_flattened", d_conv_flat, (size_t) conv3_w * conv_out_dim, stream);
            }

            bf16_linear(d_proj, d_conv_flat, conv_out_wt,
                        conv3_w, cfg->enc_d_model, conv_out_dim, handle, stream);

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_conv_out", d_proj, (size_t) conv3_w * cfg->enc_d_model, stream);
            }

            add_position_embedding_kernel<<<((size_t) conv3_w * cfg->enc_d_model + 255) / 256, 256, 0, stream>>>(
                d_proj, d_audio_pos_emb, conv3_w, cfg->enc_d_model);
            CUDA_CHECK(cudaGetLastError());

            if (g_dump_encoder_dir && chunk_idx == 0) {
                dump_bf16("chunk0_after_posemb", d_proj, (size_t) conv3_w * cfg->enc_d_model, stream);
            }

            CUDA_CHECK(cudaMemcpyAsync(
                d_hidden + (size_t) seq_offset * cfg->enc_d_model,
                d_proj,
                (size_t) chunk_hidden_elems * cfg->enc_d_model * sizeof(bf16_t),
                cudaMemcpyDeviceToDevice,
                stream));
            seq_offset += chunk_hidden_elems;
        }

        CHECK(seq_offset == seq_len,
              "audio encoder chunked frontend produced %d frames, expected %d",
              seq_offset, seq_len);
    }

    dump_bf16("after_conv", d_hidden, (size_t) seq_len * cfg->enc_d_model, stream);

    for (int layer = 0; layer < cfg->enc_layers; layer++) {
        const Tensor *sa_ln_w;
        const Tensor *sa_ln_b;
        const Tensor *q_w;
        const Tensor *q_b;
        const Tensor *k_w;
        const Tensor *k_b;
        const Tensor *v_w;
        const Tensor *v_b;
        const Tensor *o_w;
        const Tensor *o_b;
        const Tensor *ffn_ln_w;
        const Tensor *ffn_ln_b;
        const Tensor *fc1_w;
        const Tensor *fc1_b;
        const Tensor *fc2_w;
        const Tensor *fc2_b;

        make_layer_weight_name(name, sizeof(name), layer, "self_attn_layer_norm.weight");
        sa_ln_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn_layer_norm.bias");
        sa_ln_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.q_proj.weight");
        q_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.q_proj.bias");
        q_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.k_proj.weight");
        k_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.k_proj.bias");
        k_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.v_proj.weight");
        v_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.v_proj.bias");
        v_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.out_proj.weight");
        o_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "self_attn.out_proj.bias");
        o_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "final_layer_norm.weight");
        ffn_ln_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "final_layer_norm.bias");
        ffn_ln_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "fc1.weight");
        fc1_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "fc1.bias");
        fc1_b = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "fc2.weight");
        fc2_w = require_weight(ws, name);
        make_layer_weight_name(name, sizeof(name), layer, "fc2.bias");
        fc2_b = require_weight(ws, name);

        for (int ic = 0; ic < num_infer_chunks; ic++) {
            int i_len = infer_lens[ic];
            size_t i_offset = (size_t) infer_starts[ic] * cfg->enc_d_model;
            bf16_t *ih = d_hidden + i_offset;
            bf16_t *ip = d_proj + i_offset;
            bool dump_layer0_chunk0 = (layer == 0 && ic == 0 && g_dump_encoder_dir != NULL);

            layernorm(d_norm, ih,
                      (const bf16_t *) sa_ln_w->data,
                      (const bf16_t *) sa_ln_b->data,
                      cfg->enc_d_model, i_len, stream);
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_ln1", d_norm, (size_t) i_len * cfg->enc_d_model, stream);
            }

            encoder_attention(ip, d_norm, ih,
                              layer,
                              q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
                              cfg, i_len, handle, stream);
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_qkv", d_qkv, (size_t) i_len * 3 * cfg->enc_d_model, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_attn", d_attn_merge, (size_t) i_len * cfg->enc_d_model, stream);
            }
            {
                bf16_t *tmp = ih;
                ih = ip;
                ip = tmp;
            }
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_outproj", ih, (size_t) i_len * cfg->enc_d_model, stream);
            }

            layernorm(d_norm, ih,
                      (const bf16_t *) ffn_ln_w->data,
                      (const bf16_t *) ffn_ln_b->data,
                      cfg->enc_d_model, i_len, stream);
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_ln2", d_norm, (size_t) i_len * cfg->enc_d_model, stream);
            }

            bf16_linear_bias_gelu(d_ffn, d_norm,
                                  (const bf16_t *) fc1_w->data,
                                  (const bf16_t *) fc1_b->data,
                                  i_len, cfg->enc_ffn_dim, cfg->enc_d_model, handle, stream);
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_fc1_gelu", d_ffn, (size_t) i_len * cfg->enc_ffn_dim, stream);
            }
            bf16_linear_bias_residual(ip, d_ffn,
                                      (const bf16_t *) fc2_w->data,
                                      (const bf16_t *) fc2_b->data,
                                      ih,
                                      i_len, cfg->enc_d_model, cfg->enc_ffn_dim, handle, stream);
            if (dump_layer0_chunk0) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                dump_bf16("layer00_post_fc2_res", ip, (size_t) i_len * cfg->enc_d_model, stream);
            }
            {
                bf16_t *tmp = ih;
                ih = ip;
                ip = tmp;
            }
        }
        if (g_dump_encoder_dir) {
            static char layer_name[32];
            snprintf(layer_name, sizeof(layer_name), "layer%02d", layer);
            dump_bf16(layer_name, d_hidden, (size_t) seq_len * cfg->enc_d_model, stream);
        }
    }

    layernorm(d_norm, d_hidden,
              (const bf16_t *) require_weight(ws, "thinker.audio_tower.ln_post.weight")->data,
              (const bf16_t *) require_weight(ws, "thinker.audio_tower.ln_post.bias")->data,
              cfg->enc_d_model, seq_len, stream);

    bf16_linear_bias(d_proj, d_norm,
                     (const bf16_t *) require_weight(ws, "thinker.audio_tower.proj1.weight")->data,
                     (const bf16_t *) require_weight(ws, "thinker.audio_tower.proj1.bias")->data,
                     seq_len, cfg->enc_output_dim, cfg->enc_d_model, handle, stream);
    gelu_forward(d_proj, d_proj, seq_len * cfg->enc_output_dim, stream);

    dump_bf16("after_proj1_gelu", d_proj, (size_t) seq_len * cfg->enc_output_dim, stream);

    bf16_linear_bias(output, d_proj,
                     (const bf16_t *) require_weight(ws, "thinker.audio_tower.proj2.weight")->data,
                     (const bf16_t *) require_weight(ws, "thinker.audio_tower.proj2.bias")->data,
                     seq_len, cfg->enc_output_dim, cfg->enc_output_dim, handle, stream);

    dump_bf16("encoder_output", output, (size_t) seq_len * cfg->enc_output_dim, stream);
}

}
