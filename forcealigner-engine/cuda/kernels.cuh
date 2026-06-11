#ifndef FA_KERNELS_CUH
#define FA_KERNELS_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "types.h"

/* RMS normalization: out = weight * x * rsqrt(mean(x^2) + eps) */
void rmsnorm(bf16_t *out, const bf16_t *x, const bf16_t *weight, int hidden, int n, cudaStream_t stream);

/* Fused residual add + RMS normalization: hidden += residual; out = weight * hidden * rsqrt(mean(hidden^2) + eps) */
void fused_add_residual_rmsnorm(bf16_t *out, bf16_t *hidden_inout, const bf16_t *residual,
                                const bf16_t *weight, int hidden, int n, cudaStream_t stream);

/* Layer normalization: out = weight * (x - mean(x)) / sqrt(var(x) + eps) + bias */
void layernorm(bf16_t *out, const bf16_t *x, const bf16_t *weight, const bf16_t *bias, int hidden, int n, cudaStream_t stream);

/* Layer norm without bias */
void layernorm_no_bias(bf16_t *out, const bf16_t *x, const bf16_t *weight, int hidden, int n, cudaStream_t stream);

/* Rotary position embedding: apply RoPE to Q and K */
void rope_forward(bf16_t *q, bf16_t *k, int seq_len, int heads, int kv_heads, int head_dim, float theta, cudaStream_t stream);

/* GELU activation (exact: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))) */
void gelu_forward(bf16_t *out, const bf16_t *x, int n, cudaStream_t stream);

/* SiLU activation: x * sigmoid(x) */
void silu_forward(bf16_t *out, const bf16_t *x, int n, cudaStream_t stream);

/* BF16 GEMM via cuBLAS: C = alpha * A @ B^T + beta * C
 * A: [M, K] row-major, B: [N, K] row-major (transposed), C: [M, N] row-major */
void bf16_linear(bf16_t *C, const bf16_t *A, const bf16_t *B, int M, int N, int K,
                 cublasHandle_t handle, cudaStream_t stream);

/* BF16 GEMM with bias: C = A @ B^T + bias */
void bf16_linear_bias(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias,
                      int M, int N, int K, cublasHandle_t handle, cudaStream_t stream);

/* BF16 GEMM with fused bias and residual add: C = A @ B^T + bias + residual */
void bf16_linear_bias_residual(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias,
                               const bf16_t *residual, int M, int N, int K,
                               cublasHandle_t handle, cudaStream_t stream);

/* BF16 GEMM with fused bias and GELU: C = GELU(A @ B^T + bias) */
void bf16_linear_bias_gelu(bf16_t *C, const bf16_t *A, const bf16_t *B, const bf16_t *bias,
                           int M, int N, int K, cublasHandle_t handle, cudaStream_t stream);

/* Softmax in-place on last dimension: [*, n] */
void softmax_inplace(bf16_t *x, int rows, int cols, cudaStream_t stream);

/* Argmax on last dimension: out[i] = argmax(x[i, :]) */
void argmax_last_dim(int *out, const bf16_t *x, int rows, int cols, cudaStream_t stream);

/* Argmax selected rows from a row-major matrix: out[i] = argmax(x[row_indices[i], :]) */
void argmax_indexed_last_dim(int *out, const bf16_t *x, const int *row_indices, int rows, int cols, cudaStream_t stream);

/* Embedding lookup: out[i] = weight[ids[i]] */
void embedding_lookup(bf16_t *out, const bf16_t *weight, const int *ids, int seq_len, int hidden, cudaStream_t stream);

/* Scatter add: dst[indices[i]] = src[i] */
void scatter_copy(bf16_t *dst, const bf16_t *src, const int *indices, int n, int hidden, cudaStream_t stream);

/* Add residual: out = a + b */
void add_residual(bf16_t *out, const bf16_t *a, const bf16_t *b, int n, cudaStream_t stream);

/* Scale and add: out = a * scale + b (for GQA repeat_kv) */
void repeat_kv(bf16_t *out, const bf16_t *kv, int batch, int kv_heads, int n_rep, int seq_len, int head_dim, cudaStream_t stream);

#endif /* FA_KERNELS_CUH */
