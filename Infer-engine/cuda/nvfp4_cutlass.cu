#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdlib.h>
#include "kernels.cuh"

static cublasLtHandle_t g_lt_handle = NULL;
static void *g_workspace = NULL;
static size_t g_workspace_size = 32 * 1024 * 1024;
static float *g_one_dev = NULL;

static int ensure_nvfp4_resources(void)
{
    cudaError_t err;
    if (g_lt_handle == NULL) {
        if (cublasLtCreate(&g_lt_handle) != CUBLAS_STATUS_SUCCESS) return -1;
    }
    if (g_workspace == NULL) {
        err = cudaMalloc(&g_workspace, g_workspace_size);
        if (err != cudaSuccess) return -1;
    }
    if (g_one_dev == NULL) {
        float one = 1.0f;
        err = cudaMalloc(&g_one_dev, sizeof(float));
        if (err != cudaSuccess) return -1;
        err = cudaMemcpy(g_one_dev, &one, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return -1;
    }
    return 0;
}

static __global__ void quantize_bf16_to_fp4_kernel(
    uint8_t *packed_out,
    uint8_t *scale_out,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *pre_quant_scale,
    float input_scale_inv,
    int rows,
    int cols,
    int padded_rows)
{
    int block_col = blockIdx.x;
    int row = blockIdx.y;
    int lane = threadIdx.x;
    int col = block_col * 16 + lane;

    __shared__ float sh_vals[16];

    float v = 0.0f;
    if (row < rows && col < cols) {
        float pre_scale = pre_quant_scale ? __bfloat162float(pre_quant_scale[col]) : 1.0f;
        v = __bfloat162float(x[row * cols + col]) * pre_scale;
    }
    sh_vals[lane] = fabsf(v);
    __syncthreads();

    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride && sh_vals[lane + stride] > sh_vals[lane]) {
            sh_vals[lane] = sh_vals[lane + stride];
        }
        __syncthreads();
    }

    float scale = sh_vals[0] > 0.0f ? (input_scale_inv * sh_vals[0] / 6.0f) : 0.0f;
    __nv_fp8_e4m3 s_fp8(scale);
    float scale_rounded = scale > 0.0f ? (float)s_fp8 : 0.0f;
    float output_scale = (scale_rounded > 0.0f) ? (input_scale_inv / scale_rounded) : 0.0f;
    if (lane == 0) {
        scale_out[row * (cols / 16) + block_col] = s_fp8.__x;
    }
    __syncthreads();

    if (lane < 8) {
        float pre0 = pre_quant_scale ? __bfloat162float(pre_quant_scale[block_col * 16 + lane * 2]) : 1.0f;
        float pre1 = pre_quant_scale ? __bfloat162float(pre_quant_scale[block_col * 16 + lane * 2 + 1]) : 1.0f;
        __nv_fp4_e2m1 q0((row < rows) ? (__bfloat162float(x[row * cols + block_col * 16 + lane * 2]) * pre0 * output_scale) : 0.0f);
        __nv_fp4_e2m1 q1((row < rows) ? (__bfloat162float(x[row * cols + block_col * 16 + lane * 2 + 1]) * pre1 * output_scale) : 0.0f);
        packed_out[row * (cols / 2) + block_col * 8 + lane] = (q0.__x & 0x0F) | (uint8_t)((q1.__x & 0x0F) << 4);
    }

    if (row >= rows && row < padded_rows && lane == 0) {
        scale_out[row * (cols / 16) + block_col] = 0;
    }
}

static __global__ void swizzle_rowwise_scales_kernel(
    uint8_t *dst,
    const uint8_t *src,
    int rows,
    int cols,
    int cols_padded,
    int cols_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int r = idx / cols;
    int c = idx % cols;
    int tile_y = r / 128;
    int tile_y_rem = r % 128;
    int tile_row_group = tile_y_rem / 32;
    int tile_row = tile_y_rem % 32;
    int tile_x = c / 4;
    int tile_col = c % 4;
    size_t out_idx = (((((size_t)tile_y * cols_tiles + tile_x) * 32 + tile_row) * 4 + tile_row_group) * 4 + tile_col);
    dst[out_idx] = src[idx];
}

extern "C" int nvfp4_cutlass_available(void)
{
    return 1;
}

extern "C" int nvfp4_linear(void *unused_handle,
                             __nv_bfloat16 *out, __nv_bfloat16 *dequant_buf,
                             const __nv_bfloat16 *x,
                             const uint8_t *w_packed, const uint8_t *w_scales,
                             float input_scale,
                             float w_global_scale,
                             const __nv_bfloat16 *pre_quant_scale,
                             const __nv_bfloat16 *bias,
                             int M, int N, int K)
{
    if (M <= 0 || N <= 0 || K <= 0) return -1;
    if (w_scales == NULL) return -1;
    if ((K % 16) != 0) return -1;
    if (input_scale <= 0.0f) return -1;

    int padded_M = ((M + 15) / 16) * 16;
    int act_scale_cols = K / 16;
    int act_scale_cols_padded = ((act_scale_cols + 3) / 4) * 4;
    int act_scale_rows_padded = ((padded_M + 127) / 128) * 128;
    int act_scale_tiles = act_scale_cols_padded / 4;
    int w_scale_cols = K / 16;
    int w_scale_cols_padded = ((w_scale_cols + 3) / 4) * 4;
    int w_scale_rows_padded = ((N + 127) / 128) * 128;
    int w_scale_tiles = w_scale_cols_padded / 4;
    uint8_t *tmp = (uint8_t *)(dequant_buf + (size_t)N * K);
    uint8_t *x_packed = tmp;
    uint8_t *x_scales = x_packed + (size_t)padded_M * (K / 2);
    uint8_t *x_scales_t = x_scales + (size_t)padded_M * act_scale_cols;
    uint8_t *w_scales_t = x_scales_t + (size_t)act_scale_rows_padded * act_scale_cols_padded;
    __nv_bfloat16 *out_tmp = (__nv_bfloat16 *)(w_scales_t + (size_t)w_scale_rows_padded * w_scale_cols_padded);
    __nv_bfloat16 *out_to_use = (padded_M == M) ? out : out_tmp;

    cublasLtMatmulDesc_t matmulDesc = NULL;
    cublasLtMatrixLayout_t layoutA = NULL, layoutB = NULL, layoutC = NULL, layoutD = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    size_t workspaceSize = g_workspace_size;
    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;
    cudaError_t err = cudaSuccess;
    int returnedResults = 0;
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;

    (void)unused_handle;
    (void)bias;

    if (ensure_nvfp4_resources() != 0) goto cleanup;

    err = cudaMemset(x_scales_t, 0, (size_t)act_scale_rows_padded * act_scale_cols_padded);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemset(w_scales_t, 0, (size_t)w_scale_rows_padded * w_scale_cols_padded);
    if (err != cudaSuccess) goto cleanup;

    quantize_bf16_to_fp4_kernel<<<dim3(K / 16, padded_M), 16>>>(x_packed, x_scales, x, pre_quant_scale, 1.0f / input_scale, M, K, padded_M);
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    {
        int total = padded_M * act_scale_cols;
        swizzle_rowwise_scales_kernel<<<(total + 255) / 256, 256>>>(
            x_scales_t, x_scales, padded_M, act_scale_cols, act_scale_cols_padded, act_scale_tiles);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    {
        int total = N * w_scale_cols;
        swizzle_rowwise_scales_kernel<<<(total + 255) / 256, 256>>>(
            w_scales_t, w_scales, N, w_scale_cols, w_scale_cols_padded, w_scale_tiles);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;

    status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    {
        const uint8_t *a_scales = w_scales_t;
        status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scales, sizeof(a_scales));
    }
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    {
        const uint8_t *b_scales = x_scales_t;
        status = cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scales, sizeof(b_scales));
    }
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_4F_E2M1, K, N, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_4F_E2M1, K, padded_M, K);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, padded_M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, padded_M, N);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspaceSize, sizeof(workspaceSize));
    if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;

    status = cublasLtMatmulAlgoGetHeuristic(g_lt_handle, matmulDesc,
                                            layoutA, layoutB, layoutC, layoutD,
                                            preference, 1, &heuristic, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) goto cleanup;

    {
        float alpha = input_scale * w_global_scale;
        float beta = 0.0f;
        if (padded_M != M) {
            err = cudaMemset(out_tmp, 0, (size_t)padded_M * N * sizeof(__nv_bfloat16));
            if (err != cudaSuccess) goto cleanup;
        }
        status = cublasLtMatmul(g_lt_handle, matmulDesc,
                                &alpha,
                                w_packed, layoutA,
                                x_packed, layoutB,
                                &beta,
                                out_to_use, layoutC,
                                out_to_use, layoutD,
                                &heuristic.algo,
                                g_workspace, workspaceSize, 0);
        if (status != CUBLAS_STATUS_SUCCESS) goto cleanup;
    }

    if (padded_M != M) {
        err = cudaMemcpy2D(out, (size_t)N * sizeof(__nv_bfloat16),
                           out_tmp, (size_t)N * sizeof(__nv_bfloat16),
                           (size_t)N * sizeof(__nv_bfloat16), M,
                           cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) goto cleanup;
    }

    status = CUBLAS_STATUS_SUCCESS;

cleanup:
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (layoutA) cublasLtMatrixLayoutDestroy(layoutA);
    if (layoutB) cublasLtMatrixLayoutDestroy(layoutB);
    if (layoutC) cublasLtMatrixLayoutDestroy(layoutC);
    if (layoutD) cublasLtMatrixLayoutDestroy(layoutD);
    if (matmulDesc) cublasLtMatmulDescDestroy(matmulDesc);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}
