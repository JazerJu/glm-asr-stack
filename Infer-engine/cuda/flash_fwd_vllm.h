#ifndef FLASH_FWD_VLLM_H
#define FLASH_FWD_VLLM_H

#ifdef __cplusplus
extern "C" {
#endif

int flash_fwd_vllm_available(void);
int flash_fwd_vllm_launch(
    void *q, void *k, void *v, void *out,
    int seq_len, int hidden_dim,
    int q_stride, int k_stride, int v_stride,
    int num_heads, int head_dim);

int flash_fwd_bridge_available(void);
int flash_fwd_bridge_launch(
    void *q, void *k, void *v, void *out,
    int seq_len, int num_heads, int head_dim,
    float softmax_scale);

int flash_fwd_vllm_gqa_available(void);
int flash_fwd_vllm_gqa_launch(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int q_heads, int kv_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_row_stride,
    int num_seqs,
    int *d_cu_seqlens,
    int is_causal);

int flash_fwd_vllm_launch_varlen(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int num_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_stride,
    int num_seqs,
    int *d_cu_seqlens);

int flash_fwd_vllm_dense_tma_available(void);
int flash_fwd_vllm_launch_dense_tma(
    void *q, void *k, void *v, void *out,
    int num_seqs, int seq_len,
    int num_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_row_stride);

#ifdef __cplusplus
}
#endif
#endif
