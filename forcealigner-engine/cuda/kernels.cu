#include "kernels.cuh"
#include "cuda_kernels.h"

#include <cmath>
#include <vector>

namespace {

static void *g_cublas_workspace = nullptr;
static size_t g_cublas_workspace_size = 0;
static bf16_t *g_rope_sincos = nullptr;
static int g_rope_cached_half_dim = 0;
static int g_rope_cached_max_seq = 0;
static float g_rope_cached_theta = 0;
static constexpr int kRopeMaxSeqLen = 8192;

}

extern "C" void cublas_set_workspace(void *ptr, size_t size) {
    g_cublas_workspace = ptr;
    g_cublas_workspace_size = size;
}

extern "C" void cublas_restore_workspace(void *handle) {
    if (g_cublas_workspace && g_cublas_workspace_size) {
        cublasSetWorkspace(static_cast<cublasHandle_t>(handle), g_cublas_workspace, g_cublas_workspace_size);
    }
}

namespace {

static void restore_workspace(cublasHandle_t handle) {
    if (g_cublas_workspace && g_cublas_workspace_size) {
        cublasSetWorkspace(handle, g_cublas_workspace, g_cublas_workspace_size);
    }
}

constexpr int WARP_SIZE = 32;
constexpr int NORM_THREADS = 256;
constexpr int EW_THREADS = 256;
constexpr int SOFTMAX_THREADS = 256;
constexpr float kNormEps = 1e-6f;
constexpr float kNegInf = -1.0e30f;

__device__ inline float bf16_to_float(bf16_t v) {
    return __bfloat162float(v);
}

__device__ inline bf16_t float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    return val;
}

__device__ inline int warp_reduce_argmax(float &val, int idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        const float other_val = __shfl_down_sync(0xffffffffu, val, offset);
        const int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

__global__ void rmsnorm_kernel(bf16_t *out, const bf16_t *x, const bf16_t *weight, int hidden, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bf16_t *x_row = x + static_cast<size_t>(row) * hidden;
    bf16_t *out_row = out + static_cast<size_t>(row) * hidden;

    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = bf16_to_float(x_row[i]);
        sum += v * v;
    }
    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[NORM_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    float total = (tid < blockDim.x / WARP_SIZE) ? warp_sums[tid] : 0.0f;
    if (tid < WARP_SIZE) {
        total = warp_reduce_sum(total);
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / hidden + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += blockDim.x) {
        const float xv = bf16_to_float(x_row[i]);
        const float wv = bf16_to_float(weight[i]);
        out_row[i] = float_to_bf16(xv * inv_rms * wv);
    }
}

__global__ void fused_add_residual_rmsnorm_kernel(bf16_t *out, bf16_t *hidden_inout, const bf16_t *residual,
                                                  const bf16_t *weight, int hidden, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    bf16_t *hidden_row = hidden_inout + static_cast<size_t>(row) * hidden;
    const bf16_t *residual_row = residual + static_cast<size_t>(row) * hidden;
    bf16_t *out_row = out + static_cast<size_t>(row) * hidden;

    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float updated = bf16_to_float(hidden_row[i]) + bf16_to_float(residual_row[i]);
        hidden_row[i] = float_to_bf16(updated);
        sum += updated * updated;
    }
    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[NORM_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    float total = (tid < blockDim.x / WARP_SIZE) ? warp_sums[tid] : 0.0f;
    if (tid < WARP_SIZE) {
        total = warp_reduce_sum(total);
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(total / hidden + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += blockDim.x) {
        const float updated = bf16_to_float(hidden_row[i]);
        const float wv = bf16_to_float(weight[i]);
        out_row[i] = float_to_bf16(updated * inv_rms * wv);
    }
}

template <bool HasBias>
__global__ void layernorm_kernel(bf16_t *out, const bf16_t *x, const bf16_t *weight, const bf16_t *bias, int hidden, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bf16_t *x_row = x + static_cast<size_t>(row) * hidden;
    bf16_t *out_row = out + static_cast<size_t>(row) * hidden;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = bf16_to_float(x_row[i]);
        sum += v;
        sq_sum += v * v;
    }
    sum = warp_reduce_sum(sum);
    sq_sum = warp_reduce_sum(sq_sum);

    __shared__ float warp_sum[NORM_THREADS / WARP_SIZE];
    __shared__ float warp_sq_sum[NORM_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sum[tid / WARP_SIZE] = sum;
        warp_sq_sum[tid / WARP_SIZE] = sq_sum;
    }
    __syncthreads();

    float total_sum = (tid < blockDim.x / WARP_SIZE) ? warp_sum[tid] : 0.0f;
    float total_sq_sum = (tid < blockDim.x / WARP_SIZE) ? warp_sq_sum[tid] : 0.0f;
    if (tid < WARP_SIZE) {
        total_sum = warp_reduce_sum(total_sum);
        total_sq_sum = warp_reduce_sum(total_sq_sum);
    }

    __shared__ float mean;
    __shared__ float inv_std;
    if (tid == 0) {
        mean = total_sum / hidden;
        const float var = total_sq_sum / hidden - mean * mean;
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    for (int i = tid; i < hidden; i += blockDim.x) {
        const float xv = bf16_to_float(x_row[i]);
        const float wv = bf16_to_float(weight[i]);
        float outv = (xv - mean) * inv_std * wv;
        if constexpr (HasBias) {
            outv += bf16_to_float(bias[i]);
        }
        out_row[i] = float_to_bf16(outv);
    }
}

__global__ void rope_kernel(bf16_t *q, bf16_t *k, int seq_len, int heads, int kv_heads, int head_dim, const bf16_t *table) {
    const int half_dim = head_dim / 2;
    const int total_q = seq_len * heads * half_dim;
    const int total = total_q + seq_len * kv_heads * half_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    bf16_t *base = nullptr;
    int linear_idx = idx;
    int local_heads = heads;
    if (idx < total_q) {
        base = q;
    } else {
        base = k;
        linear_idx = idx - total_q;
        local_heads = kv_heads;
    }

    const int d = linear_idx % half_dim;
    const int head_idx = (linear_idx / half_dim) % local_heads;
    const int seq = linear_idx / (half_dim * local_heads);
    const int offset = (seq * local_heads + head_idx) * head_dim;
    const int tidx = (seq * half_dim + d) * 2;
    const float c = __bfloat162float(table[tidx]);
    const float s = __bfloat162float(table[tidx + 1]);

    const float x_lo = bf16_to_float(base[offset + d]);
    const float x_hi = bf16_to_float(base[offset + half_dim + d]);
    base[offset + d] = float_to_bf16(x_lo * c - x_hi * s);
    base[offset + half_dim + d] = float_to_bf16(x_lo * s + x_hi * c);
}

__global__ void gelu_kernel(bf16_t *out, const bf16_t *x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float xv = bf16_to_float(x[idx]);
    const float inner = 0.7978845608028654f * (xv + 0.044715f * xv * xv * xv);
    out[idx] = float_to_bf16(0.5f * xv * (1.0f + tanhf(inner)));
}

__global__ void silu_kernel(bf16_t *out, const bf16_t *x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float xv = bf16_to_float(x[idx]);
    out[idx] = float_to_bf16(xv / (1.0f + expf(-xv)));
}

__global__ void add_bias_kernel(bf16_t *C, const bf16_t *bias, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int col = idx % cols;
    C[idx] = float_to_bf16(bf16_to_float(C[idx]) + bf16_to_float(bias[col]));
}

__global__ void add_bias_residual_kernel(bf16_t *C, const bf16_t *bias, const bf16_t *residual, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int col = idx % cols;
    const float value = bf16_to_float(C[idx]) + bf16_to_float(bias[col]) + bf16_to_float(residual[idx]);
    C[idx] = float_to_bf16(value);
}

__global__ void add_residual_kernel(bf16_t *C, const bf16_t *residual, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    C[idx] = float_to_bf16(bf16_to_float(C[idx]) + bf16_to_float(residual[idx]));
}

__global__ void bias_gelu_kernel(bf16_t *C, const bf16_t *bias, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int col = idx % cols;
    const float xv = bf16_to_float(C[idx]) + bf16_to_float(bias[col]);
    const float inner = 0.7978845608028654f * (xv + 0.044715f * xv * xv * xv);
    C[idx] = float_to_bf16(0.5f * xv * (1.0f + tanhf(inner)));
}

__global__ void softmax_kernel(bf16_t *x, int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    bf16_t *row_ptr = x + static_cast<size_t>(row) * cols;

    float local_max = kNegInf;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, bf16_to_float(row_ptr[i]));
    }
    local_max = warp_reduce_max(local_max);

    __shared__ float warp_max[SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_max[tid / WARP_SIZE] = local_max;
    }
    __syncthreads();

    float row_max = (tid < blockDim.x / WARP_SIZE) ? warp_max[tid] : kNegInf;
    if (tid < WARP_SIZE) {
        row_max = warp_reduce_max(row_max);
    }

    __shared__ float max_shared;
    if (tid == 0) {
        max_shared = row_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_sum += expf(bf16_to_float(row_ptr[i]) - max_shared);
    }
    local_sum = warp_reduce_sum(local_sum);

    __shared__ float warp_sum[SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sum[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    float row_sum = (tid < blockDim.x / WARP_SIZE) ? warp_sum[tid] : 0.0f;
    if (tid < WARP_SIZE) {
        row_sum = warp_reduce_sum(row_sum);
    }

    __shared__ float inv_sum;
    if (tid == 0) {
        inv_sum = 1.0f / row_sum;
    }
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        const float val = expf(bf16_to_float(row_ptr[i]) - max_shared) * inv_sum;
        row_ptr[i] = float_to_bf16(val);
    }
}

__global__ void argmax_kernel(int *out, const bf16_t *x, int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bf16_t *row_ptr = x + static_cast<size_t>(row) * cols;

    float best_val = kNegInf;
    int best_idx = 0;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float v = bf16_to_float(row_ptr[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx = warp_reduce_argmax(best_val, best_idx);

    __shared__ float warp_vals[SOFTMAX_THREADS / WARP_SIZE];
    __shared__ int warp_idxs[SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_vals[tid / WARP_SIZE] = best_val;
        warp_idxs[tid / WARP_SIZE] = best_idx;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        best_val = (tid < blockDim.x / WARP_SIZE) ? warp_vals[tid] : kNegInf;
        best_idx = (tid < blockDim.x / WARP_SIZE) ? warp_idxs[tid] : 0;
        best_idx = warp_reduce_argmax(best_val, best_idx);
        if (tid == 0) {
            out[row] = best_idx;
        }
    }
}

__global__ void argmax_indexed_kernel(int *out, const bf16_t *x, const int *row_indices, int cols) {
    const int out_row = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = row_indices[out_row];
    const bf16_t *row_ptr = x + static_cast<size_t>(row) * cols;

    float best_val = kNegInf;
    int best_idx = 0;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float v = bf16_to_float(row_ptr[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx = warp_reduce_argmax(best_val, best_idx);

    __shared__ float warp_vals[SOFTMAX_THREADS / WARP_SIZE];
    __shared__ int warp_idxs[SOFTMAX_THREADS / WARP_SIZE];
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_vals[tid / WARP_SIZE] = best_val;
        warp_idxs[tid / WARP_SIZE] = best_idx;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        best_val = (tid < blockDim.x / WARP_SIZE) ? warp_vals[tid] : kNegInf;
        best_idx = (tid < blockDim.x / WARP_SIZE) ? warp_idxs[tid] : 0;
        best_idx = warp_reduce_argmax(best_val, best_idx);
        if (tid == 0) {
            out[out_row] = best_idx;
        }
    }
}

__global__ void embedding_lookup_kernel(bf16_t *out, const bf16_t *weight, const int *ids, int seq_len, int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_len * hidden;
    if (idx >= total) return;
    const int seq = idx / hidden;
    const int h = idx % hidden;
    out[idx] = weight[static_cast<size_t>(ids[seq]) * hidden + h];
}

__global__ void scatter_copy_kernel(bf16_t *dst, const bf16_t *src, const int *indices, int total, int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int row = idx / hidden;
    const int col = idx % hidden;
    dst[static_cast<size_t>(indices[row]) * hidden + col] = src[idx];
}

__global__ void add_residual_kernel(bf16_t *out, const bf16_t *a, const bf16_t *b, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = float_to_bf16(bf16_to_float(a[idx]) + bf16_to_float(b[idx]));
}

__global__ void repeat_kv_kernel(bf16_t *out, const bf16_t *kv, int batch, int kv_heads, int n_rep, int seq_len, int head_dim) {
    const int total_heads = kv_heads * n_rep;
    const int total = batch * total_heads * seq_len * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    const int d = tmp % head_dim;
    tmp /= head_dim;
    const int seq = tmp % seq_len;
    tmp /= seq_len;
    const int head = tmp % total_heads;
    const int batch_idx = tmp / total_heads;
    const int src_head = head / n_rep;

    const size_t src_idx = (((static_cast<size_t>(batch_idx) * kv_heads + src_head) * seq_len) + seq) * head_dim + d;
    out[idx] = kv[src_idx];
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace

void rmsnorm(bf16_t *out, const bf16_t *x, const bf16_t *weight, int hidden, int n, cudaStream_t stream) {
    rmsnorm_kernel<<<n, NORM_THREADS, 0, stream>>>(out, x, weight, hidden, kNormEps);
    CUDA_CHECK(cudaGetLastError());
}

void fused_add_residual_rmsnorm(bf16_t *out, bf16_t *hidden_inout, const bf16_t *residual,
                                const bf16_t *weight, int hidden, int n, cudaStream_t stream) {
    fused_add_residual_rmsnorm_kernel<<<n, NORM_THREADS, 0, stream>>>(out, hidden_inout, residual, weight, hidden, kNormEps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm(bf16_t *out, const bf16_t *x, const bf16_t *weight, const bf16_t *bias, int hidden, int n, cudaStream_t stream) {
    layernorm_kernel<true><<<n, NORM_THREADS, 0, stream>>>(out, x, weight, bias, hidden, kNormEps);
    CUDA_CHECK(cudaGetLastError());
}

void layernorm_no_bias(bf16_t *out, const bf16_t *x, const bf16_t *weight, int hidden, int n, cudaStream_t stream) {
    layernorm_kernel<false><<<n, NORM_THREADS, 0, stream>>>(out, x, weight, nullptr, hidden, kNormEps);
    CUDA_CHECK(cudaGetLastError());
}

void rope_forward(bf16_t *q, bf16_t *k, int seq_len, int heads, int kv_heads, int head_dim, float theta, cudaStream_t stream) {
    CHECK(seq_len <= kRopeMaxSeqLen, "rope_forward seq_len exceeds max cached length: %d > %d", seq_len, kRopeMaxSeqLen);

    const int half_dim = head_dim / 2;
    if (g_rope_sincos == nullptr || g_rope_cached_half_dim != half_dim || g_rope_cached_max_seq != kRopeMaxSeqLen || g_rope_cached_theta != theta) {
        if (g_rope_sincos != nullptr) {
            CUDA_CHECK(cudaFree(g_rope_sincos));
            g_rope_sincos = nullptr;
        }

        std::vector<bf16_t> host_table(static_cast<size_t>(kRopeMaxSeqLen) * half_dim * 2);
        for (int pos = 0; pos < kRopeMaxSeqLen; ++pos) {
            for (int d = 0; d < half_dim; ++d) {
                const float inv_freq = powf(theta, -2.0f * static_cast<float>(d) / (2.0f * half_dim));
                const float angle = pos * inv_freq;
                const size_t tidx = (static_cast<size_t>(pos) * half_dim + d) * 2;
                host_table[tidx] = __float2bfloat16(cosf(angle));
                host_table[tidx + 1] = __float2bfloat16(sinf(angle));
            }
        }

        const size_t table_bytes = host_table.size() * sizeof(bf16_t);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_rope_sincos), table_bytes));
        CUDA_CHECK(cudaMemcpy(g_rope_sincos, host_table.data(), table_bytes, cudaMemcpyHostToDevice));
        g_rope_cached_half_dim = half_dim;
        g_rope_cached_max_seq = kRopeMaxSeqLen;
        g_rope_cached_theta = theta;
    }

    const int total_pairs = seq_len * (heads + kv_heads) * (head_dim / 2);
    rope_kernel<<<ceil_div(total_pairs, EW_THREADS), EW_THREADS, 0, stream>>>(q, k, seq_len, heads, kv_heads, head_dim, g_rope_sincos);
    CUDA_CHECK(cudaGetLastError());
}

void gelu_forward(bf16_t *out, const bf16_t *x, int n, cudaStream_t stream) {
    gelu_kernel<<<ceil_div(n, EW_THREADS), EW_THREADS, 0, stream>>>(out, x, n);
    CUDA_CHECK(cudaGetLastError());
}

void silu_forward(bf16_t *out, const bf16_t *x, int n, cudaStream_t stream) {
    silu_kernel<<<ceil_div(n, EW_THREADS), EW_THREADS, 0, stream>>>(out, x, n);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_linear(bf16_t *C, const bf16_t *A, const bf16_t *B, int M, int N, int K, cublasHandle_t handle, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed in bf16_linear: %d M=%d N=%d K=%d", static_cast<int>(status), M, N, K);
}

void bf16_linear_bias(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias, int M, int N, int K, cublasHandle_t handle, cudaStream_t stream) {
    bf16_linear(C, A, B, M, N, K, handle, stream);
    const int total = M * N;
    add_bias_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(C, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_linear_bias_residual(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias, const bf16_t *residual, int M, int N, int K, cublasHandle_t handle, cudaStream_t stream) {
    bf16_linear(C, A, B, M, N, K, handle, stream);
    const int total = M * N;
    add_bias_residual_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(C, bias, residual, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_linear_bias_gelu(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias, int M, int N, int K, cublasHandle_t handle, cudaStream_t stream) {
    bf16_linear(C, A, B, M, N, K, handle, stream);
    const int total = M * N;
    bias_gelu_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(C, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_inplace(bf16_t *x, int rows, int cols, cudaStream_t stream) {
    softmax_kernel<<<rows, SOFTMAX_THREADS, 0, stream>>>(x, cols);
    CUDA_CHECK(cudaGetLastError());
}

void argmax_last_dim(int *out, const bf16_t *x, int rows, int cols, cudaStream_t stream) {
    argmax_kernel<<<rows, SOFTMAX_THREADS, 0, stream>>>(out, x, cols);
    CUDA_CHECK(cudaGetLastError());
}

void argmax_indexed_last_dim(int *out, const bf16_t *x, const int *row_indices, int rows, int cols, cudaStream_t stream) {
    argmax_indexed_kernel<<<rows, SOFTMAX_THREADS, 0, stream>>>(out, x, row_indices, cols);
    CUDA_CHECK(cudaGetLastError());
}

void embedding_lookup(bf16_t *out, const bf16_t *weight, const int *ids, int seq_len, int hidden, cudaStream_t stream) {
    const int total = seq_len * hidden;
    embedding_lookup_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(out, weight, ids, seq_len, hidden);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_copy(bf16_t *dst, const bf16_t *src, const int *indices, int n, int hidden, cudaStream_t stream) {
    const int total = n * hidden;
    scatter_copy_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(dst, src, indices, total, hidden);
    CUDA_CHECK(cudaGetLastError());
}

void add_residual(bf16_t *out, const bf16_t *a, const bf16_t *b, int n, cudaStream_t stream) {
    add_residual_kernel<<<ceil_div(n, EW_THREADS), EW_THREADS, 0, stream>>>(out, a, b, n);
    CUDA_CHECK(cudaGetLastError());
}

void repeat_kv(bf16_t *out, const bf16_t *kv, int batch, int kv_heads, int n_rep, int seq_len, int head_dim, cudaStream_t stream) {
    const int total = batch * kv_heads * n_rep * seq_len * head_dim;
    repeat_kv_kernel<<<ceil_div(total, EW_THREADS), EW_THREADS, 0, stream>>>(out, kv, batch, kv_heads, n_rep, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" {

void cuda_rmsnorm(uint16_t *out, const uint16_t *x, const uint16_t *weight, int hidden, int n, void *stream) {
    rmsnorm(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(x), reinterpret_cast<const bf16_t *>(weight), hidden, n, static_cast<cudaStream_t>(stream));
}

void cuda_fused_add_residual_rmsnorm(uint16_t *out, uint16_t *hidden_inout, const uint16_t *residual,
                                     const uint16_t *weight, int hidden, int n, void *stream) {
    fused_add_residual_rmsnorm(reinterpret_cast<bf16_t *>(out), reinterpret_cast<bf16_t *>(hidden_inout),
                               reinterpret_cast<const bf16_t *>(residual), reinterpret_cast<const bf16_t *>(weight),
                               hidden, n, static_cast<cudaStream_t>(stream));
}

void cuda_layernorm(uint16_t *out, const uint16_t *x, const uint16_t *weight, const uint16_t *bias, int hidden, int n, void *stream) {
    layernorm(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(x), reinterpret_cast<const bf16_t *>(weight), reinterpret_cast<const bf16_t *>(bias), hidden, n, static_cast<cudaStream_t>(stream));
}

void cuda_rope_forward(uint16_t *q, uint16_t *k, int seq_len, int heads, int kv_heads, int head_dim, float theta, void *stream) {
    rope_forward(reinterpret_cast<bf16_t *>(q), reinterpret_cast<bf16_t *>(k), seq_len, heads, kv_heads, head_dim, theta, static_cast<cudaStream_t>(stream));
}

void cuda_gelu_forward(uint16_t *out, const uint16_t *x, int n, void *stream) {
    gelu_forward(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(x), n, static_cast<cudaStream_t>(stream));
}

void cuda_silu_forward(uint16_t *out, const uint16_t *x, int n, void *stream) {
    silu_forward(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(x), n, static_cast<cudaStream_t>(stream));
}

void cuda_bf16_linear(uint16_t *C, const uint16_t *A, const uint16_t *B, int M, int N, int K, void *handle, void *stream) {
    bf16_linear(reinterpret_cast<bf16_t *>(C), reinterpret_cast<const bf16_t *>(A), reinterpret_cast<const bf16_t *>(B), M, N, K, static_cast<cublasHandle_t>(handle), static_cast<cudaStream_t>(stream));
}

void cuda_softmax_inplace(uint16_t *x, int rows, int cols, void *stream) {
    softmax_inplace(reinterpret_cast<bf16_t *>(x), rows, cols, static_cast<cudaStream_t>(stream));
}

void cuda_argmax_last_dim(int *out, const uint16_t *x, int rows, int cols, void *stream) {
    argmax_last_dim(out, reinterpret_cast<const bf16_t *>(x), rows, cols, static_cast<cudaStream_t>(stream));
}

void cuda_argmax_indexed_last_dim(int *out, const uint16_t *x, const int *row_indices, int rows, int cols, void *stream) {
    argmax_indexed_last_dim(out, reinterpret_cast<const bf16_t *>(x), row_indices, rows, cols, static_cast<cudaStream_t>(stream));
}

void cuda_embedding_lookup(uint16_t *out, const uint16_t *weight, const int *ids, int seq_len, int hidden, void *stream) {
    embedding_lookup(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(weight), ids, seq_len, hidden, static_cast<cudaStream_t>(stream));
}

void cuda_scatter_copy(uint16_t *dst, const uint16_t *src, const int *indices, int n, int hidden, void *stream) {
    scatter_copy(reinterpret_cast<bf16_t *>(dst), reinterpret_cast<const bf16_t *>(src), indices, n, hidden, static_cast<cudaStream_t>(stream));
}

void cuda_add_residual(uint16_t *out, const uint16_t *a, const uint16_t *b, int n, void *stream) {
    add_residual(reinterpret_cast<bf16_t *>(out), reinterpret_cast<const bf16_t *>(a), reinterpret_cast<const bf16_t *>(b), n, static_cast<cudaStream_t>(stream));
}

}  // extern "C"
