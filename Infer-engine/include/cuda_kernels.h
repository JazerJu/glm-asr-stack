#ifndef GLMASR_CUDA_KERNELS_H
#define GLMASR_CUDA_KERNELS_H

#include <stdint.h>
#include "types.h"
#include "kv_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void mel_spectrogram(const float *pcm, int num_samples,
                            const float *mel_filters_transposed,
                            uint16_t *mel_out, int *out_frames);
extern void mel_cleanup(void);

extern void fp4_dequant_to_bf16(uint16_t *out,
                                const uint8_t *packed_w,
                                const uint8_t *block_scales,
                                float global_scale,
                                int rows, int cols);

extern void rmsnorm(uint16_t *out, const uint16_t *x,
                    const uint16_t *weight, float eps, int rows, int dim);

extern void layernorm(uint16_t *out, const uint16_t *x,
                      const uint16_t *weight, const uint16_t *bias,
                      float eps, int rows, int dim);

extern void gelu_inplace(uint16_t *x, int n);
extern void silu_inplace(uint16_t *x, int n);
extern void silu_elementwise_mul(uint16_t *gate, const uint16_t *up, int n);

extern void rope_encoder(uint16_t *q, uint16_t *k,
                         int seq_len, int num_heads, int head_dim,
                         float partial_factor);

extern void rope_decoder(uint16_t *q, uint16_t *k,
                         int pos_offset, int seq_len,
                         int q_heads, int kv_heads, int head_dim);

extern void embedding_lookup(uint16_t *out, const uint16_t *table,
                             const int *ids, int n, int dim);

extern void residual_add(uint16_t *out, const uint16_t *a,
                         const uint16_t *b, int n);
extern void residual_add_bias(uint16_t *out,
                              const uint16_t *a,
                              const uint16_t *b,
                              const uint16_t *bias,
                              int M, int N);

extern void concat_4frames(uint16_t *out, const uint16_t *in,
                           int seq_len, int dim);

extern void conv1d_forward(uint16_t *out, const uint16_t *input,
                           const uint16_t *weight, const uint16_t *bias,
                           int in_ch, int out_ch, int seq_len, int kernel_size,
                           int stride, int padding);

extern void transpose_2d(uint16_t *out, const uint16_t *in,
                         int rows, int cols);

extern void softmax_inplace(uint16_t *data, int rows, int cols);

extern void bf16_linear(void *handle,
                        uint16_t *out,
                        const uint16_t *x, const uint16_t *w,
                        const uint16_t *bias,
                        int M, int N, int K);
extern void bf16_linear_gelu(void *handle,
                             uint16_t *out,
                             const uint16_t *x, const uint16_t *w,
                             const uint16_t *bias,
                             int M, int N, int K);
extern void silu_mul_split_packed(uint16_t *out,
                                  const uint16_t *packed_gate_up,
                                  int M, int N);

extern void fp4_linear(void *handle,
                       uint16_t *out,
                       uint16_t *weight_scratch,
                       uint16_t *prequant_scratch,
                       const uint16_t *x,
                       const uint8_t *w_packed, const uint8_t *w_scales,
                       float w_global_scale,
                       const uint16_t *pre_quant_scale,
                       const uint16_t *bias,
                       int M, int N, int K);

extern int nvfp4_cutlass_available(void);

extern int nvfp4_linear(void *handle,
                        uint16_t *out, uint16_t *dequant_buf,
                        const uint16_t *x,
                        const uint8_t *w_packed, const uint8_t *w_scales,
                        float input_scale,
                        float w_global_scale,
                        const uint16_t *pre_quant_scale,
                        const uint16_t *bias,
                        int M, int N, int K);

extern void encoder_forward(void *handle, const void *ws,
                            uint16_t *hidden, int seq_len,
                            const void *cfg);
extern void encoder_forward_packed(void *handle, const void *ws,
                                   uint16_t *hidden, int total_rows,
                                   int num_seqs, const int *seq_lens,
                                   const int *seq_offsets,
                                   const void *cfg);

extern void encoder_scratch_profile_reset(void);
extern void encoder_scratch_profile_get(ScratchProfileStats *out_stats);
extern void encoder_attention_profile_reset(void);
extern void encoder_attention_profile_get(EncoderAttentionProfileStats *out_stats);
extern void encoder_attention_microbench(void *handle,
                                         int hidden_dim, int num_heads, int head_dim,
                                         const int *seq_lens, int n_seq_lens,
                                         int warmup_iters, int iters);
extern void encoder_cleanup(void);

extern void decoder_prefill(void *handle, const void *ws,
                            uint16_t *hidden, int seq_len,
                            void *kv, uint16_t *dequant_buf,
                            const void *cfg);

extern void decoder_cleanup(void);
extern void decoder_cleanup_all(void);

extern void decoder_scratch_profile_reset(void);
extern void decoder_scratch_profile_get(ScratchProfileStats *out_stats);
extern void decoder_decode_kernel_profile_reset(void);
extern void decoder_decode_kernel_profile_get(DecodeKernelProfileStats *out_stats);
extern void decoder_prefill_kernel_profile_reset(void);
extern void decoder_prefill_kernel_profile_get(PrefillKernelProfileStats *out_stats);

extern void decoder_decode_step(void *handle, const void *ws,
                                uint16_t *hidden,
                                void *kv, uint16_t *dequant_buf,
                                const void *cfg);

extern void decoder_prefill_paged(void *handle, const void *ws,
                                  uint16_t *hidden, int seq_len,
                                  KVPool *pool, BlockTable *bt,
                                  uint16_t *dequant_buf,
                                  const void *cfg);

extern void decoder_prefill_paged_batched(void *handle, const void *ws,
                                          uint16_t *hidden, int total_tokens,
                                          int num_seqs,
                                          const int *seq_lens,
                                          const int *seq_offsets,
                                          KVPool *pool, BlockTable *bts,
                                          uint16_t *dequant_buf,
                                          const void *cfg);

extern void decoder_decode_step_paged(void *handle, const void *ws,
                                      uint16_t *hidden,
                                      KVPool *pool, BlockTable *bt,
                                      int *d_seq_lens, int seq_idx,
                                      uint16_t *dequant_buf,
                                      const void *cfg);

extern void decoder_decode_step_batched(void *handle, const void *ws,
                                        uint16_t **query_ptrs,
                                        KVPool *pool, BlockTable *bts,
                                        int *d_seq_lens, int *h_seq_lens,
                                        int num_seqs, int *seq_indices,
                                        uint16_t *dequant_buf,
                                        const void *cfg);

extern void lm_head_argmax(const uint16_t *hidden,
                           const uint16_t *lm_weight,
                           int dim, int vocab_size, int *out_token);

extern void lm_head_argmax_batched(void *handle,
                                   const uint16_t *hidden,
                                   const uint16_t *lm_weight,
                                   int rows, int dim, int vocab_size,
                                   int *out_tokens);

extern void lm_head_cleanup(void);

extern void kv_cache_alloc(void *kv, const void *cfg, int max_seq);
extern void kv_cache_free(void *kv);

extern void reshape_and_cache(uint16_t *key_cache, uint16_t *value_cache,
                              const uint16_t *key, const uint16_t *value,
                              const int64_t *slot_mapping,
                              int num_tokens, int num_kv_heads,
                              int head_dim, int block_size);

extern void paged_decode_attention(uint16_t *out,
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
