#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "../include/types.h"

#define WARP_SIZE 32
#define NEG_INF_F (-1.0e30f)

static __device__ __forceinline__ float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float block_reduce_sum(float val)
{
    __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp = threadIdx.x >> 5;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

static __device__ __forceinline__ float warp_reduce_sum_f32(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float warp_reduce_sum_broadcast(float val)
{
    val = warp_reduce_sum_f32(val);
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

static __device__ __forceinline__ size_t paged_kv_offset(
    int physical_block,
    int token_offset,
    int kv_head,
    int dim,
    int block_size,
    int num_kv_heads,
    int head_dim)
{
    return (((size_t)physical_block * block_size + token_offset) * num_kv_heads + kv_head) * head_dim + dim;
}

__global__ void reshape_and_cache_kernel(
    const bf16_t *key,
    const bf16_t *value,
    bf16_t *key_cache,
    bf16_t *value_cache,
    const int64_t *slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * num_kv_heads * head_dim;
    if (idx >= total) return;

    int token_idx = idx / (num_kv_heads * head_dim);
    int kv_head = (idx / head_dim) % num_kv_heads;
    int dim = idx % head_dim;

    int64_t slot = slot_mapping[token_idx];
    int physical_block = (int)(slot / block_size);
    int token_offset = (int)(slot % block_size);
    size_t cache_idx = paged_kv_offset(physical_block, token_offset, kv_head, dim,
                                       block_size, num_kv_heads, head_dim);

    key_cache[cache_idx] = key[idx];
    value_cache[cache_idx] = value[idx];
}

__global__ void paged_decode_attention_kernel(
    bf16_t *out,
    const bf16_t *query,
    const bf16_t *key_cache,
    const bf16_t *value_cache,
    const int *block_tables,
    const int *seq_lens,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale)
{
    int q_head = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_idx >= num_seqs || q_head >= num_q_heads) return;

    int seq_len = seq_lens[seq_idx];
    bf16_t *out_ptr = out + ((size_t)seq_idx * num_q_heads + q_head) * head_dim;
    if (seq_len <= 0) {
        if (tid < head_dim) {
            out_ptr[tid] = __float2bfloat16(0.0f);
        }
        return;
    }

    int q_per_kv = num_q_heads / num_kv_heads;
    int kv_head = q_head / q_per_kv;
    const bf16_t *q_ptr = query + ((size_t)seq_idx * num_q_heads + q_head) * head_dim;
    const int *block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;

    __shared__ float shared_score;

    float running_max = NEG_INF_F;
    float running_sum = 0.0f;
    float acc = 0.0f;
    int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int logical_block = 0; logical_block < num_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        int block_token_base = logical_block * block_size;
        int block_token_count = seq_len - block_token_base;
        if (block_token_count > block_size) {
            block_token_count = block_size;
        }

        for (int token_offset = 0; token_offset < block_token_count; token_offset++) {
            float partial = 0.0f;
            if (physical_block >= 0) {
                for (int d = tid; d < head_dim; d += blockDim.x) {
                    size_t cache_idx = paged_kv_offset(physical_block, token_offset, kv_head, d,
                                                       block_size, num_kv_heads, head_dim);
                    float qv = __bfloat162float(q_ptr[d]);
                    float kv = __bfloat162float(key_cache[cache_idx]);
                    partial += qv * kv;
                }
            }

            float score = block_reduce_sum(partial) * scale;
            if (tid == 0) {
                shared_score = (physical_block >= 0) ? score : NEG_INF_F;
            }
            __syncthreads();

            {
                float score_val = shared_score;
                float new_max = fmaxf(running_max, score_val);
                float prev_scale = (running_max == NEG_INF_F) ? 0.0f : expf(running_max - new_max);
                float score_scale = (score_val == NEG_INF_F) ? 0.0f : expf(score_val - new_max);

                if (physical_block >= 0 && tid < head_dim) {
                    size_t value_idx = paged_kv_offset(physical_block, token_offset, kv_head, tid,
                                                       block_size, num_kv_heads, head_dim);
                    float vv = __bfloat162float(value_cache[value_idx]);
                    acc = acc * prev_scale + score_scale * vv;
                } else if (tid < head_dim) {
                    acc *= prev_scale;
                }

                running_sum = running_sum * prev_scale + score_scale;
                running_max = new_max;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        shared_score = running_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / (shared_score + 1e-9f);
    if (tid < head_dim) {
        out_ptr[tid] = __float2bfloat16(acc * inv_sum);
    }
}

__global__ void paged_decode_attention_grouped_kernel(
    bf16_t *out,
    const bf16_t *query,
    const bf16_t *key_cache,
    const bf16_t *value_cache,
    const int *block_tables,
    const int *seq_lens,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale)
{
    int kv_head = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    int q_per_kv = num_q_heads / num_kv_heads;
    int threads_per_group = head_dim;
    int q_sub = tid / threads_per_group;
    int dim = tid % threads_per_group;
    int q_head = kv_head * q_per_kv + q_sub;
    int lane = dim & (WARP_SIZE - 1);
    int warp_in_group = dim / WARP_SIZE;

    __shared__ float sh_k[256];
    __shared__ float sh_v[256];
    __shared__ float sh_group_warp_sums[64];
    __shared__ float sh_score[16];
    __shared__ float sh_prev_scale[16];
    __shared__ float sh_score_scale[16];
    __shared__ float sh_running_sum[16];
    __shared__ float sh_running_max[16];
    __shared__ float sh_inv_sum[16];

    if (seq_idx >= num_seqs || kv_head >= num_kv_heads || q_sub >= q_per_kv) return;

    int seq_len = seq_lens[seq_idx];
    bf16_t *out_ptr = out + ((size_t)seq_idx * num_q_heads + q_head) * head_dim;
    if (seq_len <= 0) {
        out_ptr[dim] = __float2bfloat16(0.0f);
        return;
    }

    const bf16_t *q_ptr = query + ((size_t)seq_idx * num_q_heads + q_head) * head_dim;
    const int *block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;
    float q_cached = __bfloat162float(q_ptr[dim]);
    float running_max = NEG_INF_F;
    float running_sum = 0.0f;
    float acc = 0.0f;
    int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int logical_block = 0; logical_block < num_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        int block_token_base = logical_block * block_size;
        int block_token_count = seq_len - block_token_base;
        if (block_token_count > block_size) {
            block_token_count = block_size;
        }

        for (int token_offset = 0; token_offset < block_token_count; token_offset++) {
            if (q_sub == 0 && dim < head_dim) {
                if (physical_block >= 0) {
                    size_t cache_idx = paged_kv_offset(physical_block, token_offset, kv_head, dim,
                                                       block_size, num_kv_heads, head_dim);
                    sh_k[dim] = __bfloat162float(key_cache[cache_idx]);
                    sh_v[dim] = __bfloat162float(value_cache[cache_idx]);
                } else {
                    sh_k[dim] = 0.0f;
                    sh_v[dim] = 0.0f;
                }
            }
            __syncthreads();

            float partial = (physical_block >= 0) ? (q_cached * sh_k[dim]) : 0.0f;
            partial = warp_reduce_sum_f32(partial);
            if (lane == 0) {
                sh_group_warp_sums[q_sub * 4 + warp_in_group] = partial;
            }
            __syncthreads();

            if (warp_in_group == 0) {
                float score = (lane < 4) ? sh_group_warp_sums[q_sub * 4 + lane] : 0.0f;
                score = warp_reduce_sum_f32(score) * scale;
                if (lane == 0) {
                    sh_score[q_sub] = (physical_block >= 0) ? score : NEG_INF_F;
                }
            }
            __syncthreads();

            if (dim == 0) {
                float score_val = sh_score[q_sub];
                float new_max = fmaxf(running_max, score_val);
                float prev_scale = (running_max == NEG_INF_F) ? 0.0f : expf(running_max - new_max);
                float score_scale = (score_val == NEG_INF_F) ? 0.0f : expf(score_val - new_max);
                sh_prev_scale[q_sub] = prev_scale;
                sh_score_scale[q_sub] = score_scale;
                sh_running_sum[q_sub] = running_sum * prev_scale + score_scale;
                sh_running_max[q_sub] = new_max;
            }
            __syncthreads();

            if (physical_block >= 0) {
                acc = acc * sh_prev_scale[q_sub] + sh_score_scale[q_sub] * sh_v[dim];
            } else {
                acc *= sh_prev_scale[q_sub];
            }

            running_sum = sh_running_sum[q_sub];
            running_max = sh_running_max[q_sub];
            __syncthreads();
        }
    }

    if (dim == 0) {
        sh_inv_sum[q_sub] = 1.0f / (running_sum + 1e-9f);
    }
    __syncthreads();
    out_ptr[dim] = __float2bfloat16(acc * sh_inv_sum[q_sub]);
}

__global__ void paged_decode_attention_gqa4_h128_warp_kernel(
    bf16_t *out,
    const bf16_t *query,
    const bf16_t *key_cache,
    const bf16_t *value_cache,
    const int *block_tables,
    const int *seq_lens,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    float scale)
{
    int kv_head = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    int q_sub = tid >> 5;
    int lane = tid & (WARP_SIZE - 1);
    int q_head = kv_head * 4 + q_sub;

    if (seq_idx >= num_seqs || kv_head >= num_kv_heads || q_sub >= 4 || q_head >= num_q_heads) {
        return;
    }

    int seq_len = seq_lens[seq_idx];
    bf16_t *out_ptr = out + ((size_t)seq_idx * num_q_heads + q_head) * 128;
    if (seq_len <= 0) {
        out_ptr[lane] = __float2bfloat16(0.0f);
        out_ptr[lane + 32] = __float2bfloat16(0.0f);
        out_ptr[lane + 64] = __float2bfloat16(0.0f);
        out_ptr[lane + 96] = __float2bfloat16(0.0f);
        return;
    }

    const bf16_t *q_ptr = query + ((size_t)seq_idx * num_q_heads + q_head) * 128;
    const int *block_table = block_tables + (size_t)seq_idx * max_blocks_per_seq;

    float q0 = __bfloat162float(q_ptr[lane]);
    float q1 = __bfloat162float(q_ptr[lane + 32]);
    float q2 = __bfloat162float(q_ptr[lane + 64]);
    float q3 = __bfloat162float(q_ptr[lane + 96]);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float running_max = NEG_INF_F;
    float running_sum = 0.0f;
    int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int logical_block = 0; logical_block < num_blocks; logical_block++) {
        int physical_block = block_table[logical_block];
        int block_token_base = logical_block * block_size;
        int block_token_count = seq_len - block_token_base;
        if (block_token_count > block_size) {
            block_token_count = block_size;
        }

        for (int token_offset = 0; token_offset < block_token_count; token_offset++) {
            float partial = 0.0f;
            float v0 = 0.0f;
            float v1 = 0.0f;
            float v2 = 0.0f;
            float v3 = 0.0f;

            if (physical_block >= 0) {
                size_t base = (((size_t)physical_block * block_size + token_offset) * num_kv_heads + kv_head) * 128;
                float k0 = __bfloat162float(key_cache[base + lane]);
                float k1 = __bfloat162float(key_cache[base + lane + 32]);
                float k2 = __bfloat162float(key_cache[base + lane + 64]);
                float k3 = __bfloat162float(key_cache[base + lane + 96]);
                v0 = __bfloat162float(value_cache[base + lane]);
                v1 = __bfloat162float(value_cache[base + lane + 32]);
                v2 = __bfloat162float(value_cache[base + lane + 64]);
                v3 = __bfloat162float(value_cache[base + lane + 96]);
                partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            }

            float score = warp_reduce_sum_broadcast(partial) * scale;
            if (physical_block < 0) {
                score = NEG_INF_F;
            }

            float new_max = fmaxf(running_max, score);
            float prev_scale = (running_max == NEG_INF_F) ? 0.0f : expf(running_max - new_max);
            float score_scale = (score == NEG_INF_F) ? 0.0f : expf(score - new_max);

            acc0 = acc0 * prev_scale + score_scale * v0;
            acc1 = acc1 * prev_scale + score_scale * v1;
            acc2 = acc2 * prev_scale + score_scale * v2;
            acc3 = acc3 * prev_scale + score_scale * v3;
            running_sum = running_sum * prev_scale + score_scale;
            running_max = new_max;
        }
    }

    float inv_sum = 1.0f / (running_sum + 1e-9f);
    out_ptr[lane] = __float2bfloat16(acc0 * inv_sum);
    out_ptr[lane + 32] = __float2bfloat16(acc1 * inv_sum);
    out_ptr[lane + 64] = __float2bfloat16(acc2 * inv_sum);
    out_ptr[lane + 96] = __float2bfloat16(acc3 * inv_sum);
}

extern "C" void reshape_and_cache(
    uint16_t *key_cache,
    uint16_t *value_cache,
    const uint16_t *key,
    const uint16_t *value,
    const int64_t *slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size)
{
    int total = num_tokens * num_kv_heads * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    reshape_and_cache_kernel<<<grid, block>>>(
        (const bf16_t *)key,
        (const bf16_t *)value,
        (bf16_t *)key_cache,
        (bf16_t *)value_cache,
        slot_mapping,
        num_tokens,
        num_kv_heads,
        head_dim,
        block_size);
}

extern "C" void paged_decode_attention(
    uint16_t *out,
    const uint16_t *query,
    const uint16_t *key_cache,
    const uint16_t *value_cache,
    const int *block_tables,
    const int *seq_lens,
    int num_seqs,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq)
{
    float scale = 1.0f / sqrtf((float)head_dim);
    int q_per_kv = num_q_heads / num_kv_heads;

    if (head_dim == 128 && q_per_kv == 4) {
        dim3 grid(num_kv_heads, num_seqs);
        paged_decode_attention_gqa4_h128_warp_kernel<<<grid, 128>>>(
            (bf16_t *)out,
            (const bf16_t *)query,
            (const bf16_t *)key_cache,
            (const bf16_t *)value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            block_size,
            max_blocks_per_seq,
            scale);
    } else if (head_dim <= 256 && q_per_kv <= 16 && head_dim % WARP_SIZE == 0 &&
        head_dim * q_per_kv <= 1024) {
        dim3 grid(num_kv_heads, num_seqs);
        int block = head_dim * q_per_kv;
        paged_decode_attention_grouped_kernel<<<grid, block>>>(
            (bf16_t *)out,
            (const bf16_t *)query,
            (const bf16_t *)key_cache,
            (const bf16_t *)value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            scale);
    } else {
        dim3 grid(num_q_heads, num_seqs);
        int block = 128;
        paged_decode_attention_kernel<<<grid, block>>>(
            (bf16_t *)out,
            (const bf16_t *)query,
            (const bf16_t *)key_cache,
            (const bf16_t *)value_cache,
            block_tables,
            seq_lens,
            num_seqs,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            scale);
    }
}
