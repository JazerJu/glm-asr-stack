#ifndef FA_CUDA_KERNELS_H
#define FA_CUDA_KERNELS_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_rmsnorm(uint16_t *out, const uint16_t *x, const uint16_t *weight, int hidden, int n, void *stream);
void cuda_layernorm(uint16_t *out, const uint16_t *x, const uint16_t *weight, const uint16_t *bias, int hidden, int n, void *stream);
void cuda_rope_forward(uint16_t *q, uint16_t *k, int seq_len, int heads, int kv_heads, int head_dim, float theta, void *stream);
void cuda_gelu_forward(uint16_t *out, const uint16_t *x, int n, void *stream);
void cuda_silu_forward(uint16_t *out, const uint16_t *x, int n, void *stream);
void cuda_bf16_linear(uint16_t *C, const uint16_t *A, const uint16_t *B, int M, int N, int K, void *handle, void *stream);
void cuda_softmax_inplace(uint16_t *x, int rows, int cols, void *stream);
void cuda_argmax_last_dim(int *out, const uint16_t *x, int rows, int cols, void *stream);
void cuda_argmax_indexed_last_dim(int *out, const uint16_t *x, const int *row_indices, int rows, int cols, void *stream);
void cuda_embedding_lookup(uint16_t *out, const uint16_t *weight, const int *ids, int seq_len, int hidden, void *stream);
void cuda_scatter_copy(uint16_t *dst, const uint16_t *src, const int *indices, int n, int hidden, void *stream);
void cuda_add_residual(uint16_t *out, const uint16_t *a, const uint16_t *b, int n, void *stream);

void cublas_set_workspace(void *ptr, size_t size);
void cublas_restore_workspace(void *handle);

void cuda_mel_spectrogram(uint16_t *mel_out, const float *pcm, int pcm_samples,
                          int n_fft, int hop_length, int n_mels, int sample_rate,
                          void *stream);

void cuda_conv2d_forward(uint16_t *out, const uint16_t *input,
                         const uint16_t *weight, const uint16_t *bias,
                         int N, int C_in, int H_in, int W_in,
                         int C_out, int kH, int kW, int stride, int padding,
                         void *handle, void *stream);

void cuda_audio_encoder_forward(uint16_t *output,
                                const uint16_t *mel, int mel_T,
                                const void *weights,
                                const void *config,
                                void *handle, void *stream);

void cuda_text_decoder_forward(uint16_t *logits,
                               const uint16_t *input_embeds, const int *input_ids,
                               int seq_len,
                               const void *weights,
                               const void *config,
                               void *handle, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* FA_CUDA_KERNELS_H */
