#include "kernels.cuh"
#include "cuda_kernels.h"

#include <cufft.h>
#include <cub/cub.cuh>

#include <cmath>
#include <vector>

namespace {

constexpr float kMelFloor = 1e-10f;
constexpr float kPi = 3.14159265358979323846f;

#define CUFFT_CHECK(call) do { \
    cufftResult _e = (call); \
    CHECK(_e == CUFFT_SUCCESS, "cuFFT error [%s:%d]: %d", __FILE__, __LINE__, static_cast<int>(_e)); \
} while (0)

bf16_t *g_mel_filterbank = nullptr;
float *g_hann_window = nullptr;
int g_cached_n_fft = 0;
int g_cached_n_mels = 0;
int g_cached_sample_rate = 0;

/* Persistent cuFFT plan cache: avoid cufftPlanMany + cufftDestroy every call */
static cufftHandle g_cufft_plan = 0;
static int g_cached_plan_nfft = 0;
static int g_cached_plan_nframes = 0;

static float *g_framed = nullptr;
static size_t g_framed_elems = 0;
static cufftComplex *g_fft_out = nullptr;
static size_t g_fft_out_elems = 0;
static float *g_power = nullptr;
static size_t g_power_elems = 0;

__device__ inline bf16_t float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void apply_hann_kernel(float *framed, const float *pcm_padded,
                                   const float *window,
                                   int pcm_padded_samples, int num_frames,
                                   int n_fft, int hop_length) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_frames * n_fft;
    if (idx >= total) return;
    const int frame = idx / n_fft;
    const int offset = idx % n_fft;
    const int sample_idx = frame * hop_length + offset - n_fft / 2;
    const float sample = (sample_idx >= 0 && sample_idx < pcm_padded_samples) ? pcm_padded[sample_idx] : 0.0f;
    framed[idx] = sample * window[offset];
}

__global__ void power_spectrum_kernel(float *power, const cufftComplex *fft_out, int bins, int num_frames) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_frames * bins;
    if (idx >= total) return;
    const cufftComplex v = fft_out[idx];
    power[idx] = v.x * v.x + v.y * v.y;
}

__global__ void mel_project_kernel(bf16_t *mel_out, const float *power, const bf16_t *filterbank,
                                   int num_frames, int bins, int n_mels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_frames * n_mels;
    if (idx >= total) return;

    const int frame = idx / n_mels;
    const int mel = idx % n_mels;
    const float *power_row = power + static_cast<size_t>(frame) * bins;
    const bf16_t *filter_row = filterbank + static_cast<size_t>(mel) * bins;

    float acc = 0.0f;
    for (int b = 0; b < bins; ++b) {
        acc += power_row[b] * __bfloat162float(filter_row[b]);
    }
    float log_mel = log10f(fmaxf(acc, kMelFloor));
    mel_out[static_cast<size_t>(mel) * num_frames + frame] = float_to_bf16(log_mel);
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void ensure_float_buffer(float **ptr, size_t *capacity_elems, size_t needed_elems) {
    if (*ptr && *capacity_elems >= needed_elems) {
        return;
    }
    if (*ptr) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(ptr), needed_elems * sizeof(float)));
    *capacity_elems = needed_elems;
}

void ensure_cufft_complex_buffer(cufftComplex **ptr, size_t *capacity_elems, size_t needed_elems) {
    if (*ptr && *capacity_elems >= needed_elems) {
        return;
    }
    if (*ptr) {
        CUDA_CHECK(cudaFree(*ptr));
        *ptr = nullptr;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(ptr), needed_elems * sizeof(cufftComplex)));
    *capacity_elems = needed_elems;
}

inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* Build mel filterbank matching Python's create_mel_filter_bank exactly:
 * - Integer bin indices (not continuous frequency)
 * - Linear ramp between integer bin boundaries
 * - Slaney normalization: enorm = 2.0 / (hz_points[i+2] - hz_points[i])
 */
void ensure_mel_tables(int n_fft, int n_mels, int sample_rate) {
    if (g_mel_filterbank && g_hann_window && g_cached_n_fft == n_fft &&
        g_cached_n_mels == n_mels && g_cached_sample_rate == sample_rate) {
        return;
    }

    if (g_mel_filterbank) {
        CUDA_CHECK(cudaFree(g_mel_filterbank));
        g_mel_filterbank = nullptr;
    }
    if (g_hann_window) {
        CUDA_CHECK(cudaFree(g_hann_window));
        g_hann_window = nullptr;
    }

    /* Hann window */
    std::vector<float> hann(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        hann[i] = 0.5f - 0.5f * cosf(2.0f * kPi * i / n_fft);
    }

    /* Load pre-computed filterbank matching WhisperFeatureExtractor */
    const int bins = n_fft / 2 + 1;
    std::vector<float> filterbank(static_cast<size_t>(n_mels) * bins, 0.0f);
    FILE *ffb = fopen("mel_filterbank.bin", "rb");
    CHECK(ffb != NULL, "failed to open mel_filterbank.bin");
    size_t read_ok = fread(filterbank.data(), sizeof(float), static_cast<size_t>(n_mels) * bins, ffb);
    fclose(ffb);
    CHECK(read_ok == static_cast<size_t>(n_mels) * bins,
          "mel_filterbank.bin size mismatch: expected %zu, got %zu",
          static_cast<size_t>(n_mels) * bins, read_ok);

    /* Convert to bf16 and upload */
    std::vector<bf16_t> fb_bf16(static_cast<size_t>(n_mels) * bins);
    for (size_t k = 0; k < fb_bf16.size(); ++k) {
        fb_bf16[k] = __float2bfloat16(filterbank[k]);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_hann_window), sizeof(float) * hann.size()));
    CUDA_CHECK(cudaMemcpy(g_hann_window, hann.data(), sizeof(float) * hann.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&g_mel_filterbank), sizeof(bf16_t) * fb_bf16.size()));
    CUDA_CHECK(cudaMemcpy(g_mel_filterbank, fb_bf16.data(), sizeof(bf16_t) * fb_bf16.size(), cudaMemcpyHostToDevice));

    g_cached_n_fft = n_fft;
    g_cached_n_mels = n_mels;
    g_cached_sample_rate = sample_rate;
}

/* Convert bf16 mel to float32 for CUB reduce (overlays on power buffer) */
__global__ void bf16_to_float_kernel(float *out, const bf16_t *in, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __bfloat162float(in[idx]);
}

/* Apply Whisper-style postprocess: clamp(x, max-8) then (x+4)/4, write back as bf16 */
__global__ void mel_postprocess_kernel(bf16_t *mel, const float *d_max, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const float mel_max = d_max[0];
    float v = fmaxf(__bfloat162float(mel[idx]), mel_max - 8.0f);
    mel[idx] = __float2bfloat16((v + 4.0f) / 4.0f);
}

}  // namespace

extern "C" void cuda_mel_spectrogram(uint16_t *mel_out, const float *pcm_padded,
                                       int pcm_padded_samples,
                                       int n_fft, int hop_length, int n_mels, int sample_rate,
                                       void *stream_ptr) {
    CHECK(n_fft > 0, "n_fft must be positive");
    CHECK(hop_length > 0, "hop_length must be positive");
    CHECK(n_mels > 0, "n_mels must be positive");
    CHECK(sample_rate > 0, "sample_rate must be positive");

    ensure_mel_tables(n_fft, n_mels, sample_rate);

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    /* Frame count matching torch.stft(center=True):
     * torch.stft with center=True produces signal_length // hop_length frames.
     * Each frame i has its center at i * hop_length, with window spanning
     * [i*hop - n_fft/2, i*hop + n_fft/2 - 1]. Out-of-range samples are zero.
     */
    const int num_frames = (pcm_padded_samples < hop_length)
        ? 1
        : pcm_padded_samples / hop_length;
    const int bins = n_fft / 2 + 1;

    const size_t framed_elems = static_cast<size_t>(num_frames) * n_fft;
    const size_t fft_out_elems = static_cast<size_t>(num_frames) * bins;
    const size_t power_elems = static_cast<size_t>(num_frames) * ((bins > n_mels) ? bins : n_mels);
    ensure_float_buffer(&g_framed, &g_framed_elems, framed_elems);
    ensure_cufft_complex_buffer(&g_fft_out, &g_fft_out_elems, fft_out_elems);
    ensure_float_buffer(&g_power, &g_power_elems, power_elems);

    apply_hann_kernel<<<ceil_div(num_frames * n_fft, 256), 256, 0, stream>>>(
        g_framed, pcm_padded, g_hann_window, pcm_padded_samples, num_frames, n_fft, hop_length);
    CUDA_CHECK(cudaGetLastError());

    /* Persistent cuFFT plan: reuse when n_fft and num_frames are unchanged */
    if (!g_cufft_plan || g_cached_plan_nfft != n_fft || g_cached_plan_nframes != num_frames) {
        if (g_cufft_plan) {
            CUFFT_CHECK(cufftDestroy(g_cufft_plan));
            g_cufft_plan = 0;
        }
        const int rank = 1;
        int n[1] = {n_fft};
        int inembed[1] = {n_fft};
        int onembed[1] = {bins};
        const int istride = 1;
        const int ostride = 1;
        const int idist = n_fft;
        const int odist = bins;
        CUFFT_CHECK(cufftPlanMany(&g_cufft_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, num_frames));
        g_cached_plan_nfft = n_fft;
        g_cached_plan_nframes = num_frames;
    }
    CUFFT_CHECK(cufftSetStream(g_cufft_plan, stream));
    CUFFT_CHECK(cufftExecR2C(g_cufft_plan, g_framed, g_fft_out));

    power_spectrum_kernel<<<ceil_div(num_frames * bins, 256), 256, 0, stream>>>(g_power, g_fft_out, bins, num_frames);
    CUDA_CHECK(cudaGetLastError());

    mel_project_kernel<<<ceil_div(num_frames * n_mels, 256), 256, 0, stream>>>(
        reinterpret_cast<bf16_t *>(mel_out), g_power, g_mel_filterbank, num_frames, bins, n_mels);
    CUDA_CHECK(cudaGetLastError());

    /* Whisper-style postprocess on GPU: clamp to max-8, then (x+4)/4 */
    {
        const int total = num_frames * n_mels;

        /* Persistent CUB temp buffer for DeviceReduce::Max */
        static void *g_cub_temp = nullptr;
        static size_t g_cub_temp_bytes = 0;
        size_t req_bytes = 0;
        cub::DeviceReduce::Max(nullptr, req_bytes, (const float *)nullptr, (float *)nullptr, total);
        if (req_bytes > g_cub_temp_bytes) {
            if (g_cub_temp) CUDA_CHECK(cudaFree(g_cub_temp));
            CUDA_CHECK(cudaMalloc(&g_cub_temp, req_bytes));
            g_cub_temp_bytes = req_bytes;
        }

        /* Must go through float for CUB reduce; re-use power buffer (already float) */
        float *d_mel_f32 = g_power; /* power workspace is sized to cover max(frames*bins, frames*n_mels) */
        /* Temporary buffer for CUB reduce */
        static float *d_max_out = nullptr;
        if (!d_max_out) CUDA_CHECK(cudaMalloc(&d_max_out, sizeof(float)));

        /* Convert bf16 mel to float32 (overlay on power buffer) */
        {
            const int total = num_frames * n_mels;
            bf16_to_float_kernel<<<ceil_div(total, 256), 256, 0, stream>>>(
                d_mel_f32, reinterpret_cast<const bf16_t *>(mel_out), total);
            CUDA_CHECK(cudaGetLastError());
        }

        /* CUB DeviceReduce to find max */
        cub::DeviceReduce::Max(g_cub_temp, req_bytes, d_mel_f32, d_max_out, total, stream);
        CUDA_CHECK(cudaGetLastError());

        /* Apply postprocess: clamp(x, max-8) then (x+4)/4, write back as bf16 */
        mel_postprocess_kernel<<<ceil_div(total, 256), 256, 0, stream>>>(
            reinterpret_cast<bf16_t *>(mel_out), d_max_out, total);
        CUDA_CHECK(cudaGetLastError());
    }
}
