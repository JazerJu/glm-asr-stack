/**
 * CUDA Mel Spectrogram Implementation for GLM-ASR
 * Adapted from mel_profile_v3.cu
 * 
 * Target: RTX 5070 Ti (SM120, Blackwell), CUDA 12.8
 * 
 * Input: float *pcm (GPU), int num_samples, float *mel_filters_transposed (GPU)
 * Output: __nv_bfloat16 *mel_out in shape [128, n_frames] (n_frames computed from num_samples)
 * 
 * Parameters: N_FFT=400, HOP=160, N_MELS=128, max N_FRAMES=3000
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cufft.h>
#include <cmath>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/types.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Mel spectrogram parameters
#define N_FFT 400
#define HOP 160
#define N_MELS 128
#define N_FRAMES 3000
#define N_FFT_HALF (N_FFT / 2 + 1)  // 201 for FFT size 400
#define MAX_SAMPLES (N_FRAMES * HOP + N_FFT)  // Capacity cap (unused at runtime now)

#define EPS 1e-10f
#define DB_RANGE 8.0f
#define SCALE 4.0f

#define WARP_SIZE 32
#define MEL_REDUCE_BLOCK 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Device mel filters (passed from host)
__device__ float d_mel_filters[N_MELS * N_FFT_HALF];

// Persistent mel workspace buffers
static float *mel_d_windowed = NULL;
static cufftComplex *mel_d_stft = NULL;
static float *mel_d_mel = NULL;
static float *mel_d_max = NULL;
static float *mel_d_block_maxima = NULL;
static int mel_d_block_maxima_count = 0;
static bool mel_filters_uploaded = false;
static cufftHandle mel_fft_plan = {};
static bool mel_fft_plan_created = false;
static bool mel_buffers_allocated = false;

static void mel_allocate_buffers() {
    if (mel_buffers_allocated) return;

    CUDA_CHECK(cudaMalloc(&mel_d_windowed, (size_t)N_FRAMES * N_FFT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mel_d_stft, (size_t)N_FRAMES * N_FFT_HALF * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&mel_d_mel, (size_t)N_MELS * N_FRAMES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mel_d_max, sizeof(float)));
    mel_d_block_maxima_count = (N_MELS * N_FRAMES + MEL_REDUCE_BLOCK - 1) / MEL_REDUCE_BLOCK;
    CUDA_CHECK(cudaMalloc(&mel_d_block_maxima, (size_t)mel_d_block_maxima_count * sizeof(float)));
    // cuFFT plan for max frames (reused across calls)
    int n[] = {N_FFT};
    cufftResult cr = cufftPlanMany(&mel_fft_plan, 1, n, NULL, 1, N_FFT, NULL, 1, N_FFT_HALF, CUFFT_R2C, N_FRAMES);
    if (cr != CUFFT_SUCCESS) {
        fprintf(stderr, "CUDA error at %s:%d: cuFFT plan creation failed (%d)\n", __FILE__, __LINE__, (int)cr);
        exit(1);
    }
    mel_fft_plan_created = true;
    mel_buffers_allocated = true;
}

extern "C" void mel_cleanup() {
    if (!mel_buffers_allocated) return;

    cudaFree(mel_d_windowed);
    cudaFree(mel_d_stft);
    cudaFree(mel_d_mel);
    cudaFree(mel_d_max);
    cudaFree(mel_d_block_maxima);
    if (mel_fft_plan_created) {
        cufftDestroy(mel_fft_plan);
        mel_fft_plan_created = false;
    }
    mel_d_windowed = NULL;
    mel_d_stft = NULL;
    mel_d_mel = NULL;
    mel_d_max = NULL;
    mel_d_block_maxima = NULL;
    mel_d_block_maxima_count = 0;
    mel_filters_uploaded = false;
    mel_buffers_allocated = false;
}

/* ============================================================================
 * Windowing kernel: Apply Hann window to audio frames
 * ============================================================================ */
__global__ void window_kernel(const float *audio, float *windowed, int num_samples, int n_frames) {
    int f = blockIdx.x;
    int t = threadIdx.x;
    
    if (f >= n_frames || t >= N_FFT) return;
    
    // Periodic Hann window
    float w = 0.5f * (1.0f - cosf(2.0f * M_PI * t / N_FFT));
    int idx = f * HOP + t - N_FFT / 2;
    
    // Pad with zeros if out of bounds
    float val = 0.0f;
    if (idx >= 0 && idx < num_samples) {
        val = audio[idx] * w;
    }
    windowed[f * N_FFT + t] = val;
}

/* ============================================================================
 * Mel spectrogram kernel: Apply mel filterbank to STFT output
 * Uses transposed filters for coalesced memory access
 * ============================================================================ */
__global__ void mel_kernel(const cufftComplex *stft, float *mel, int n_frames) {
    int f = blockIdx.x;
    int m = threadIdx.x;
    
    if (f >= n_frames || m >= N_MELS) return;
    
    float p = 0.0f;
    
    // Unroll loop for better performance
    #pragma unroll 4
    for (int k = 0; k < N_FFT_HALF; k++) {
        float re = stft[f * N_FFT_HALF + k].x;
        float im = stft[f * N_FFT_HALF + k].y;
        
        // Access transposed mel filters: [k * N_MELS + m] for coalesced access
        // Filters are stored as [N_FFT_HALF, N_MELS] = [201, 128]
        p += (re * re + im * im) * d_mel_filters[k * N_MELS + m];
    }
    
    // Store in transposed layout: [m * n_frames + f] for final output
    mel[m * n_frames + f] = log10f(p + EPS);
}

/* ============================================================================
 * Block-level max reduce (pass 1): replaces CUB DeviceReduce::Max
 * Each block finds max in its chunk using warp shuffle
 * ============================================================================ */
__global__ void block_max_kernel(const float *input, float *block_maxima, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float my_max = -FLT_MAX;

    for (int i = tid; i < n; i += stride) {
        my_max = fmaxf(my_max, input[i]);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
    }

    __shared__ float warp_results[MEL_REDUCE_BLOCK / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_results[warp_id] = my_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        my_max = (threadIdx.x < (MEL_REDUCE_BLOCK / WARP_SIZE)) ? warp_results[threadIdx.x] : -FLT_MAX;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
        }
        if (threadIdx.x == 0) {
            block_maxima[blockIdx.x] = my_max;
        }
    }
}

/* ============================================================================
 * Final max reduce (pass 2): single block reduces block maxima to global max
 * ============================================================================ */
__global__ void final_max_kernel(const float *block_maxima, float *global_max, int n_blocks) {
    float my_max = -FLT_MAX;
    for (int i = threadIdx.x; i < n_blocks; i += blockDim.x) {
        my_max = fmaxf(my_max, block_maxima[i]);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, offset));
    }
    if (threadIdx.x == 0) {
        *global_max = my_max;
    }
}

/* ============================================================================
 * Normalize + BF16 conversion (fused kernel, replaces 2 separate launches)
 * ============================================================================ */
__global__ void normalize_bf16_kernel(__nv_bfloat16 *out, const float *mel,
                                       const float *d_global_max, int n_frames) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= n_frames || y >= N_MELS) return;
    
    int idx = y * n_frames + x;
    float val = mel[idx];
    float max_val = *d_global_max;
    
    val = fmaxf(val, max_val - DB_RANGE);
    val = (val + SCALE) / SCALE;
    
    out[idx] = __float2bfloat16(val);
}

/* ============================================================================
 * Main mel spectrogram function
 * 
 * Input:
 *   - pcm: float array on GPU, raw audio samples
 *   - num_samples: number of audio samples
 *   - mel_filters_transposed: float array on GPU, shape [N_FFT_HALF, N_MELS] = [201, 128]
 *                           This is the transposed mel filterbank
 * 
 * Output:
 *   - mel_out: bf16 array on GPU, shape [N_MELS, n_frames]
 *   - out_frames: actual number of frames computed (≤ N_FRAMES)
 * ============================================================================ */
extern "C"
void mel_spectrogram(const float *pcm, int num_samples,
                     const float *mel_filters_transposed,
                     __nv_bfloat16 *mel_out, int *out_frames) {
    mel_allocate_buffers();

    // Match WhisperFeatureExtractor's attention_mask frame count:
    // input_features_mask.sum(-1) = ceil(num_samples / HOP).
    int n_frames = (num_samples + HOP - 1) / HOP;
    if (n_frames < 1) n_frames = 1;
    if (n_frames > N_FRAMES) n_frames = N_FRAMES;  // cap at max capacity
    
    *out_frames = n_frames;
    
    // Upload mel filters once (not per-call)
    if (!mel_filters_uploaded) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_mel_filters, mel_filters_transposed, 
                                       N_FFT_HALF * N_MELS * sizeof(float)));
        mel_filters_uploaded = true;
    }
    
    // 1. Windowing
    window_kernel<<<n_frames, N_FFT>>>(pcm, mel_d_windowed, num_samples, n_frames);
    CUDA_CHECK(cudaGetLastError());
    
    // 2. FFT (reuse persistent plan)
    cufftExecR2C(mel_fft_plan, (cufftReal *)mel_d_windowed, mel_d_stft);
    CUDA_CHECK(cudaGetLastError());
    
    // 3. Mel filterbank: apply mel filters to STFT magnitude squared
    mel_kernel<<<n_frames, N_MELS>>>(mel_d_stft, mel_d_mel, n_frames);
    CUDA_CHECK(cudaGetLastError());
    
    // 4. Two-pass max reduce (replaces CUB DeviceReduce::Max)
    {
        int total = N_MELS * n_frames;
        int n_blocks = (total + MEL_REDUCE_BLOCK - 1) / MEL_REDUCE_BLOCK;
        block_max_kernel<<<n_blocks, MEL_REDUCE_BLOCK>>>(mel_d_mel, mel_d_block_maxima, total);
        CUDA_CHECK(cudaGetLastError());
        if (n_blocks == 1) {
            CUDA_CHECK(cudaMemcpy(mel_d_max, mel_d_block_maxima, sizeof(float), cudaMemcpyDeviceToDevice));
        } else {
            final_max_kernel<<<1, MEL_REDUCE_BLOCK>>>(mel_d_block_maxima, mel_d_max, n_blocks);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    
    // 5. Normalize + convert to BF16 (fused, replaces 2 separate launches)
    dim3 norm_block(32, 8);
    dim3 norm_grid((n_frames + norm_block.x - 1) / norm_block.x, 
                   (N_MELS + norm_block.y - 1) / norm_block.y);
    normalize_bf16_kernel<<<norm_grid, norm_block>>>(mel_out, mel_d_mel, mel_d_max, n_frames);
    CUDA_CHECK(cudaGetLastError());
}
