#ifndef GLMASR_KERNELS_CUH
#define GLMASR_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "../include/types.h"
#include "../include/kv_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

void mel_spectrogram(const float *pcm, int num_samples,
                     const float *mel_filters_transposed,
                     __nv_bfloat16 *mel_out, int *out_frames);

void fp4_dequant_to_bf16(__nv_bfloat16 *out,
                         const uint8_t *packed_w,
                         const uint8_t *block_scales,
                         float global_scale,
                         int rows, int cols);

void swizzle_weight_scale(uint8_t *swizzled, const uint8_t *original,
                         int N, int K_blocks);

void rmsnorm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
             const __nv_bfloat16 *weight, float eps, int rows, int dim);

void layernorm(__nv_bfloat16 *out, const __nv_bfloat16 *x,
               const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
               float eps, int rows, int dim);

void gelu_inplace(__nv_bfloat16 *x, int n);
void silu_inplace(__nv_bfloat16 *x, int n);

void silu_elementwise_mul(__nv_bfloat16 *gate, const __nv_bfloat16 *up, int n);

void rope_encoder(__nv_bfloat16 *q, __nv_bfloat16 *k,
                  int seq_len, int num_heads, int head_dim,
                  float partial_factor);

void rope_decoder(__nv_bfloat16 *q, __nv_bfloat16 *k,
                  int pos_offset, int seq_len,
                  int q_heads, int kv_heads, int head_dim);

void rope_decoder_batched(__nv_bfloat16 *buf,
                          const int *cu_seqlens,
                          int num_seqs, int max_seq_len,
                          int n_heads, int head_dim);
void split_packed_qkv_rope_decoder_batched(__nv_bfloat16 *q_dst,
                                           __nv_bfloat16 *k_dst,
                                           __nv_bfloat16 *v_dst,
                                           const __nv_bfloat16 *packed,
                                           const int *cu_seqlens,
                                           int num_seqs, int max_seq_len,
                                           int num_heads, int num_kv_heads,
                                           int head_dim);
void embedding_lookup(__nv_bfloat16 *out, const __nv_bfloat16 *table,
                      const int *ids, int n, int dim);

void residual_add(__nv_bfloat16 *out, const __nv_bfloat16 *a,
                  const __nv_bfloat16 *b, int n);
void residual_add_bias(__nv_bfloat16 *out,
                       const __nv_bfloat16 *a,
                       const __nv_bfloat16 *b,
                       const __nv_bfloat16 *bias,
                       int M, int N);

void concat_4frames(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                    int seq_len, int dim);

void bf16_linear(cublasHandle_t handle,
                 __nv_bfloat16 *out,
                 const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                 const __nv_bfloat16 *bias,
                 int M, int N, int K);
void bf16_linear_bias_residual(cublasHandle_t handle,
                               __nv_bfloat16 *out,
                               const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                               const __nv_bfloat16 *bias,
                               const __nv_bfloat16 *residual,
                               int M, int N, int K);
void bf16_linear_gelu(cublasHandle_t handle,
                      __nv_bfloat16 *out,
                      const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                      const __nv_bfloat16 *bias,
                      int M, int N, int K);
void silu_mul_split_packed(__nv_bfloat16 *out,
                           const __nv_bfloat16 *packed_gate_up,
                           int M, int N);

void fp4_linear(cublasHandle_t handle,
                __nv_bfloat16 *out,
                __nv_bfloat16 *weight_scratch,
                __nv_bfloat16 *prequant_scratch,
                const __nv_bfloat16 *x,
                const uint8_t *w_packed, const uint8_t *w_scales,
                float w_global_scale,
                const __nv_bfloat16 *pre_quant_scale,
                const __nv_bfloat16 *bias,
                int M, int N, int K);

int nvfp4_cutlass_available(void);

int nvfp4_linear(void *handle,
                 __nv_bfloat16 *out, __nv_bfloat16 *dequant_buf,
                 const __nv_bfloat16 *x,
                 const uint8_t *w_packed, const uint8_t *w_scales,
                 float input_scale,
                 float w_global_scale,
                 const __nv_bfloat16 *pre_quant_scale,
                 const __nv_bfloat16 *bias,
                 int M, int N, int K);

void conv1d_forward(__nv_bfloat16 *out, const __nv_bfloat16 *input,
                    const __nv_bfloat16 *weight, const __nv_bfloat16 *bias,
                    int in_ch, int out_ch, int seq_len, int kernel_size,
                    int stride, int padding);

void transpose_2d(__nv_bfloat16 *out, const __nv_bfloat16 *in,
                  int rows, int cols);

void softmax_inplace(__nv_bfloat16 *data, int rows, int cols);

void encoder_forward(cublasHandle_t handle, const WeightStore *ws,
                     __nv_bfloat16 *hidden, int seq_len,
                     const ModelConfig *cfg);
void encoder_forward_packed(cublasHandle_t handle, const WeightStore *ws,
                            __nv_bfloat16 *hidden, int total_rows,
                            int num_seqs, const int *seq_lens,
                            const int *seq_offsets,
                            const ModelConfig *cfg);
void encoder_attention_profile_reset(void);
void encoder_attention_profile_get(EncoderAttentionProfileStats *out_stats);
void encoder_attention_microbench(void *handle,
                                  int hidden_dim, int num_heads, int head_dim,
                                  const int *seq_lens, int n_seq_lens,
                                  int warmup_iters, int iters);

void decoder_prefill(cublasHandle_t handle, const WeightStore *ws,
                     __nv_bfloat16 *hidden, int seq_len,
                     KVCache *kv, __nv_bfloat16 *dequant_buf,
                     const ModelConfig *cfg);

void decoder_cleanup(void);
void decoder_cleanup_all(void);
void decoder_decode_kernel_profile_reset(void);
void decoder_decode_kernel_profile_get(DecodeKernelProfileStats *out_stats);
void decoder_prefill_kernel_profile_reset(void);
void decoder_prefill_kernel_profile_get(PrefillKernelProfileStats *out_stats);

void decoder_decode_step(cublasHandle_t handle, const WeightStore *ws,
                         __nv_bfloat16 *hidden,
                         KVCache *kv, __nv_bfloat16 *dequant_buf,
                         const ModelConfig *cfg);

void decoder_prefill_paged(cublasHandle_t handle, const WeightStore *ws,
                           __nv_bfloat16 *hidden, int seq_len,
                           KVPool *pool, BlockTable *bt,
                           __nv_bfloat16 *dequant_buf,
                           const ModelConfig *cfg);

void decoder_prefill_paged_batched(cublasHandle_t handle, const WeightStore *ws,
                                   __nv_bfloat16 *hidden, int total_tokens,
                                   int num_seqs,
                                   const int *seq_lens,
                                   const int *seq_offsets,
                                   KVPool *pool, BlockTable *bts,
                                   __nv_bfloat16 *dequant_buf,
                                   const ModelConfig *cfg);

void decoder_decode_step_paged(cublasHandle_t handle, const WeightStore *ws,
                                __nv_bfloat16 *hidden,
                                KVPool *pool, BlockTable *bt,
                                int *d_seq_lens, int seq_idx,
                                __nv_bfloat16 *dequant_buf,
                                const ModelConfig *cfg);

void decoder_decode_step_batched(cublasHandle_t handle, const WeightStore *ws,
                                 bf16_t **query_ptrs,
                                 KVPool *pool, BlockTable *bts,
                                 int *d_seq_lens, int *h_seq_lens,
                                 int num_seqs, int *seq_indices,
                                 __nv_bfloat16 *dequant_buf,
                                 const ModelConfig *cfg);

void lm_head_argmax(const __nv_bfloat16 *hidden,
                     const __nv_bfloat16 *lm_weight,
                     int dim, int vocab_size, int *out_token);

void lm_head_argmax_batched(cublasHandle_t handle,
                            const __nv_bfloat16 *hidden,
                            const __nv_bfloat16 *lm_weight,
                            int rows, int dim, int vocab_size,
                            int *out_tokens);

void lm_head_cleanup(void);

void kv_cache_alloc(KVCache *kv, const ModelConfig *cfg, int max_seq);
void kv_cache_free(KVCache *kv);

void reshape_and_cache(uint16_t *key_cache, uint16_t *value_cache,
                       const uint16_t *key, const uint16_t *value,
                       const int64_t *slot_mapping,
                       int num_tokens, int num_kv_heads,
                       int head_dim, int block_size);

void paged_decode_attention(uint16_t *out,
                            const uint16_t *query,
                            const uint16_t *key_cache,
                            const uint16_t *value_cache,
                            const int *block_tables,
                            const int *seq_lens,
                            int num_seqs, int num_q_heads,
                            int num_kv_heads, int head_dim,
                            int block_size, int max_blocks_per_seq);

#ifdef __cplusplus
}
#endif

#endif
