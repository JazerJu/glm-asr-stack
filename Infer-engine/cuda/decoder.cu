/*
 * GLM-ASR Decoder Forward Pass CUDA Implementation
 * 28-layer decoder with FP4 weights and GQA attention
 *
 * Key layouts:
 *   Q after transpose : [q_heads, seq, head_dim]  in d_dec_q_buf
 *   K/V proj output   : [seq, kv_hidden]           (kv_hidden = kv_heads*head_dim)
 *   KV cache          : [seq, kv_hidden]           (same layout, easy append)
 *   attn scores       : [q_heads, query_len, key_len]
 *   attn output buf   : [q_heads, query_len, head_dim]
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kernels.cuh"
#include "flash_fwd_vllm.h"
#include "../include/kv_pool.h"
#include "../include/types.h"

/* ---- scratch buffer state ---- */
static __nv_bfloat16 *d_dec_q_buf     = NULL; /* [q_heads, attn_max_seq, head_dim] */
static __nv_bfloat16 *d_dec_k_buf     = NULL; /* [token_capacity, kv_hidden]       */
static __nv_bfloat16 *d_dec_v_buf     = NULL; /* [token_capacity, kv_hidden]       */
static __nv_bfloat16 *d_dec_qk_buf    = NULL; /* [q_heads, attn_max_seq, attn_max_seq] */
static __nv_bfloat16 *d_dec_kv_buf    = NULL; /* [attn_max_seq, head_dim] one-head temp */
static __nv_bfloat16 *d_dec_attn_buf  = NULL; /* [q_heads, attn_max_seq, head_dim] */
static __nv_bfloat16 *d_dec_attn_out  = NULL; /* [token_capacity, hidden_dim]      */
static __nv_bfloat16 *d_dec_hidden_buf= NULL; /* [token_capacity, ffn_dim] also temp for Q proj */
static __nv_bfloat16 *d_dec_residual  = NULL; /* [token_capacity, hidden_dim]      */
static __nv_bfloat16 *d_dec_gate_buf  = NULL; /* [token_capacity, ffn_dim]         */
static __nv_bfloat16 *d_dec_up_buf    = NULL; /* [token_capacity, ffn_dim]         */
static __nv_bfloat16 *d_dec_gate_up_packed = NULL;

static int *d_dec_cu_seqlens = NULL;
static int  d_dec_cu_seqlens_cap = 0; /* [token_capacity, 2 * ffn_dim]   */
static bool dec_buffers_allocated = false;
static ScratchProfileStats decoder_scratch_stats = {};
static DecodeKernelProfileStats decoder_decode_kernel_stats = {};
static PrefillKernelProfileStats decoder_prefill_kernel_stats = {};
static int decoder_token_capacity = 0;
static int decoder_attn_capacity = 0;
static size_t decoder_q_bytes = 0;
static size_t decoder_kv_bytes = 0;
static size_t decoder_qk_bytes = 0;
static size_t decoder_kv_head_bytes = 0;
static size_t decoder_attn_out_bytes = 0;
static size_t decoder_hidden_bytes = 0;
static size_t decoder_residual_bytes = 0;
static size_t decoder_gate_bytes = 0;

/* Persistent batched-decode scratch buffers (grow-only pool) */
static __nv_bfloat16 *bd_hidden_batch   = NULL;
static __nv_bfloat16 *bd_residual_batch  = NULL;
static __nv_bfloat16 *bd_batched_query   = NULL;
static __nv_bfloat16 *bd_batched_k       = NULL;
static __nv_bfloat16 *bd_batched_v       = NULL;
static __nv_bfloat16 *bd_batched_attn    = NULL;
static __nv_bfloat16 *bd_gate_batch      = NULL;
static __nv_bfloat16 *bd_up_batch        = NULL;
static __nv_bfloat16 *bd_gate_up_packed  = NULL;
static int64_t        *bd_slot_mapping   = NULL;
static int            *bd_total_seq_lens = NULL;
static int            *bd_block_tables   = NULL;
static int bd_capacity = 0;            /* max num_seqs batched buffers can hold */
static int bd_max_blocks_per_seq = 0;  /* max_blocks_per_seq when bd_block_tables was allocated */

#define DECODER_SCALAR_CACHE_CAPACITY 1024
typedef struct {
    const void *device_ptr;
    float value;
} DecoderScalarCacheEntry;

static DecoderScalarCacheEntry decoder_scalar_cache[DECODER_SCALAR_CACHE_CAPACITY];
static int decoder_scalar_cache_len = 0;

typedef struct {
    __nv_bfloat16 *weight0;
    __nv_bfloat16 *aux;
    size_t weight_capacity_elems;
    size_t aux_capacity_elems;
} DecoderLinearScratchPlan;

static void free_decoder_buffers();

static void decoder_note_alloc(size_t bytes)
{
    decoder_scratch_stats.alloc_calls++;
    decoder_scratch_stats.alloc_bytes += (uint64_t)bytes;
    decoder_scratch_stats.current_bytes += (uint64_t)bytes;
    if (decoder_scratch_stats.current_bytes > decoder_scratch_stats.peak_bytes) {
        decoder_scratch_stats.peak_bytes = decoder_scratch_stats.current_bytes;
    }
}

static void decoder_note_free(size_t bytes)
{
    decoder_scratch_stats.free_calls++;
    decoder_scratch_stats.free_bytes += (uint64_t)bytes;
    if (decoder_scratch_stats.current_bytes >= (uint64_t)bytes) {
        decoder_scratch_stats.current_bytes -= (uint64_t)bytes;
    } else {
        decoder_scratch_stats.current_bytes = 0;
    }
}

static int decoder_profile_enabled()
{
    static int state = -1;
    if (state == -1) {
        const char *env = getenv("GLMASR_PROFILE");
        state = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return state;
}

static void decoder_profile_record_duration(uint64_t elapsed_ns,
                                            uint64_t *total_ns,
                                            uint64_t *max_ns)
{
    if (total_ns != NULL) {
        *total_ns += elapsed_ns;
    }
    if (max_ns != NULL && elapsed_ns > *max_ns) {
        *max_ns = elapsed_ns;
    }
}

static uint64_t decoder_cuda_event_elapsed_ns(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return (uint64_t) llround((double) elapsed_ms * 1000000.0);
}

static bool prefer_nvfp4_cutlass()
{
    static int state = -1;
    if (state == -1) {
        const char *env = getenv("GLMASR_USE_CUTLASS_NVFP4");
        state = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return state == 1;
}

static void decoder_scalar_cache_reset(void)
{
    memset(decoder_scalar_cache, 0, sizeof(decoder_scalar_cache));
    decoder_scalar_cache_len = 0;
}

static float decoder_scalar_read_cached(const Tensor *tensor, float fallback)
{
    float value;
    const void *key;

    if (tensor == NULL || tensor->data == NULL) {
        return fallback;
    }

    key = tensor->data;
    for (int i = 0; i < decoder_scalar_cache_len; i++) {
        if (decoder_scalar_cache[i].device_ptr == key) {
            return decoder_scalar_cache[i].value;
        }
    }

    CUDA_CHECK(cudaMemcpy(&value, tensor->data, sizeof(value), cudaMemcpyDeviceToHost));
    if (decoder_scalar_cache_len < DECODER_SCALAR_CACHE_CAPACITY) {
        decoder_scalar_cache[decoder_scalar_cache_len].device_ptr = key;
        decoder_scalar_cache[decoder_scalar_cache_len].value = value;
        decoder_scalar_cache_len++;
    }
    return value;
}

static DecoderLinearScratchPlan decoder_make_linear_scratch_plan(__nv_bfloat16 *base,
                                                                 const ModelConfig *cfg)
{
    DecoderLinearScratchPlan plan;
    size_t weight_elems = (size_t)cfg->dec_ffn * (size_t)cfg->dec_hidden;

    plan.weight0 = base;
    plan.aux = (base != NULL) ? (base + weight_elems) : NULL;
    plan.weight_capacity_elems = weight_elems;
    plan.aux_capacity_elems = weight_elems;
    return plan;
}

static int should_compare_nvfp4_once()
{
    static int state = -1;
    if (state == -1) {
        const char *env = getenv("GLMASR_NVFP4_COMPARE_ONCE");
        state = (env && env[0] == '1') ? 1 : 0;
    }
    return state;
}

static void compare_nvfp4_once(cublasHandle_t handle,
                               __nv_bfloat16 *out,
                               __nv_bfloat16 *weight_scratch,
                               __nv_bfloat16 *prequant_scratch,
                               const __nv_bfloat16 *x,
                               const uint8_t *w,
                               const uint8_t *ws,
                               float input_scale,
                               float global_scale,
                               const __nv_bfloat16 *pre_quant_scale,
                               int M, int N, int K,
                               const char *tag)
{
    static int done = 0;
    if (done || !should_compare_nvfp4_once()) return;
    done = 1;

    __nv_bfloat16 *native_out = NULL;
    __nv_bfloat16 *fallback_out = NULL;
    __nv_bfloat16 *host_native = NULL;
    __nv_bfloat16 *host_fallback = NULL;
    size_t count = (size_t)M * N;
    size_t bytes = count * sizeof(__nv_bfloat16);
    double max_abs = 0.0;
    double mean_abs = 0.0;
    size_t max_idx = 0;

    if (cudaMalloc(&native_out, bytes) != cudaSuccess) goto cleanup;
    if (cudaMalloc(&fallback_out, bytes) != cudaSuccess) goto cleanup;

    if (nvfp4_linear(handle, native_out, weight_scratch, x, w, ws, input_scale, global_scale,
                     pre_quant_scale, NULL, M, N, K) != 0) {
        fprintf(stderr, "NVFP4_COMPARE %s native_status=fail\n", tag);
        goto cleanup;
    }

    fp4_linear(handle, fallback_out, weight_scratch, prequant_scratch, x, w, ws, global_scale,
               pre_quant_scale, NULL, M, N, K);

    host_native = (__nv_bfloat16 *)malloc(bytes);
    host_fallback = (__nv_bfloat16 *)malloc(bytes);
    if (!host_native || !host_fallback) goto cleanup;

    CUDA_CHECK(cudaMemcpy(host_native, native_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_fallback, fallback_out, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < count; i++) {
        double a = (double)__bfloat162float(host_native[i]);
        double b = (double)__bfloat162float(host_fallback[i]);
        double d = fabs(a - b);
        mean_abs += d;
        if (d > max_abs) {
            max_abs = d;
            max_idx = i;
        }
    }
    mean_abs /= (double)count;
    fprintf(stderr,
            "NVFP4_COMPARE %s M=%d N=%d K=%d max_abs=%.6f mean_abs=%.6f idx=%zu native=%.6f fallback=%.6f\n",
            tag, M, N, K, max_abs, mean_abs, max_idx,
            (double)__bfloat162float(host_native[max_idx]),
            (double)__bfloat162float(host_fallback[max_idx]));
    CUDA_CHECK(cudaMemcpy(out, native_out, bytes, cudaMemcpyDeviceToDevice));

cleanup:
    if (host_native) free(host_native);
    if (host_fallback) free(host_fallback);
    if (native_out) cudaFree(native_out);
    if (fallback_out) cudaFree(fallback_out);
}

static void warn_nvfp4_cutlass_fallback()
{
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, "GLMASR_USE_CUTLASS_NVFP4 requested but native path is not ready, falling back to bf16 dequant path\n");
        warned = 1;
    }
}

static void allocate_decoder_buffers(int token_capacity, int attn_max_seq,
                                     int hidden_dim, int ffn_dim,
                                     int q_heads, int kv_heads, int head_dim)
{
    size_t q_bytes;
    size_t kv_bytes;
    size_t qk_bytes;
    size_t kv_head_bytes;
    size_t attn_bytes;
    size_t hidden_bytes;
    size_t residual_bytes;
    size_t gate_bytes;

    int kv_hidden = kv_heads * head_dim;

    q_bytes = (size_t)q_heads * attn_max_seq * head_dim * sizeof(__nv_bfloat16);
    kv_bytes = (size_t)token_capacity * kv_hidden * sizeof(__nv_bfloat16);
    qk_bytes = (size_t)q_heads * attn_max_seq * attn_max_seq * sizeof(__nv_bfloat16);
    kv_head_bytes = (size_t)attn_max_seq * head_dim * sizeof(__nv_bfloat16);
    attn_bytes = (size_t)token_capacity * hidden_dim * sizeof(__nv_bfloat16);
    hidden_bytes = (size_t)token_capacity * ffn_dim * sizeof(__nv_bfloat16);
    residual_bytes = (size_t)token_capacity * hidden_dim * sizeof(__nv_bfloat16);
    gate_bytes = (size_t)token_capacity * ffn_dim * sizeof(__nv_bfloat16);

    if (dec_buffers_allocated &&
        decoder_token_capacity >= token_capacity &&
        decoder_attn_capacity >= attn_max_seq &&
        decoder_q_bytes >= q_bytes &&
        decoder_kv_bytes >= kv_bytes &&
        decoder_qk_bytes >= qk_bytes &&
        decoder_kv_head_bytes >= kv_head_bytes &&
        decoder_attn_out_bytes >= attn_bytes &&
        decoder_hidden_bytes >= hidden_bytes &&
        decoder_residual_bytes >= residual_bytes &&
        decoder_gate_bytes >= gate_bytes) {
        return;
    }

    if (dec_buffers_allocated) {
        free_decoder_buffers();
    }

    decoder_q_bytes = q_bytes;
    decoder_kv_bytes = kv_bytes;
    decoder_qk_bytes = qk_bytes;
    decoder_kv_head_bytes = kv_head_bytes;
    decoder_attn_out_bytes = attn_bytes;
    decoder_hidden_bytes = hidden_bytes;
    decoder_residual_bytes = residual_bytes;
    decoder_gate_bytes = gate_bytes;
    decoder_token_capacity = token_capacity;
    decoder_attn_capacity = attn_max_seq;

    CUDA_CHECK(cudaMalloc(&d_dec_q_buf, q_bytes));
    decoder_note_alloc(q_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_k_buf, kv_bytes));
    decoder_note_alloc(kv_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_v_buf, kv_bytes));
    decoder_note_alloc(kv_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_qk_buf, qk_bytes));
    decoder_note_alloc(qk_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_kv_buf, kv_head_bytes));
    decoder_note_alloc(kv_head_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_attn_buf, q_bytes));
    decoder_note_alloc(q_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_attn_out, attn_bytes));
    decoder_note_alloc(attn_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_hidden_buf, hidden_bytes));
    decoder_note_alloc(hidden_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_residual, residual_bytes));
    decoder_note_alloc(residual_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_gate_buf, gate_bytes));
    decoder_note_alloc(gate_bytes);
    CUDA_CHECK(cudaMalloc(&d_dec_gate_up_packed, gate_bytes * 2));
    decoder_note_alloc(gate_bytes * 2);

    dec_buffers_allocated = true;
}

static void free_decoder_buffers()
{
    size_t q_bytes;
    size_t kv_bytes;
    size_t qk_bytes;
    size_t kv_head_bytes;
    size_t attn_bytes;
    size_t hidden_bytes;
    size_t residual_bytes;
    size_t gate_bytes;

    if (!dec_buffers_allocated) return;

    q_bytes = decoder_q_bytes;
    kv_bytes = decoder_kv_bytes;
    qk_bytes = decoder_qk_bytes;
    kv_head_bytes = decoder_kv_head_bytes;
    attn_bytes = decoder_attn_out_bytes;
    hidden_bytes = decoder_hidden_bytes;
    residual_bytes = decoder_residual_bytes;
    gate_bytes = decoder_gate_bytes;

    cudaFree(d_dec_q_buf);      d_dec_q_buf = NULL;
    decoder_note_free(q_bytes);
    cudaFree(d_dec_k_buf);      d_dec_k_buf = NULL;
    decoder_note_free(kv_bytes);
    cudaFree(d_dec_v_buf);      d_dec_v_buf = NULL;
    decoder_note_free(kv_bytes);
    cudaFree(d_dec_qk_buf);     d_dec_qk_buf = NULL;
    decoder_note_free(qk_bytes);
    cudaFree(d_dec_kv_buf);     d_dec_kv_buf = NULL;
    decoder_note_free(kv_head_bytes);
    cudaFree(d_dec_attn_buf);   d_dec_attn_buf = NULL;
    decoder_note_free(q_bytes);
    cudaFree(d_dec_attn_out);   d_dec_attn_out = NULL;
    decoder_note_free(attn_bytes);
    cudaFree(d_dec_hidden_buf); d_dec_hidden_buf = NULL;
    decoder_note_free(hidden_bytes);
    cudaFree(d_dec_residual);   d_dec_residual = NULL;
    decoder_note_free(residual_bytes);
    cudaFree(d_dec_gate_buf);   d_dec_gate_buf = NULL;
    decoder_note_free(gate_bytes);
    if (d_dec_up_buf) {
        cudaFree(d_dec_up_buf); d_dec_up_buf = NULL;
        decoder_note_free(gate_bytes);
    }
    cudaFree(d_dec_gate_up_packed); d_dec_gate_up_packed = NULL;
    decoder_note_free(gate_bytes * 2);

    if (d_dec_cu_seqlens) { cudaFree(d_dec_cu_seqlens); d_dec_cu_seqlens = NULL; d_dec_cu_seqlens_cap = 0; }

    decoder_q_bytes = 0;
    decoder_kv_bytes = 0;
    decoder_qk_bytes = 0;
    decoder_kv_head_bytes = 0;
    decoder_attn_out_bytes = 0;
    decoder_hidden_bytes = 0;
    decoder_residual_bytes = 0;
    decoder_gate_bytes = 0;
    decoder_token_capacity = 0;
    decoder_attn_capacity = 0;

    dec_buffers_allocated = false;
}

static void ensure_decoder_up_buf()
{
    if (d_dec_up_buf != NULL) {
        return;
    }
    CHECK(decoder_gate_bytes > 0, "decoder up buffer requested before decoder buffers were allocated");
    CUDA_CHECK(cudaMalloc(&d_dec_up_buf, decoder_gate_bytes));
    decoder_note_alloc(decoder_gate_bytes);
}

/* ---- persistent batched-decode buffer management ---- */

static void ensure_batched_buffers(int num_seqs, int max_blocks_per_seq,
                                    int hidden_dim, int ffn_dim, int kv_hidden) {
    if (bd_capacity >= num_seqs && bd_max_blocks_per_seq >= max_blocks_per_seq) {
        return;
    }

    if (bd_hidden_batch)    { CUDA_CHECK(cudaFree(bd_hidden_batch));    bd_hidden_batch = NULL; }
    if (bd_residual_batch)  { CUDA_CHECK(cudaFree(bd_residual_batch));  bd_residual_batch = NULL; }
    if (bd_batched_query)   { CUDA_CHECK(cudaFree(bd_batched_query));   bd_batched_query = NULL; }
    if (bd_batched_k)       { CUDA_CHECK(cudaFree(bd_batched_k));       bd_batched_k = NULL; }
    if (bd_batched_v)       { CUDA_CHECK(cudaFree(bd_batched_v));       bd_batched_v = NULL; }
    if (bd_batched_attn)    { CUDA_CHECK(cudaFree(bd_batched_attn));    bd_batched_attn = NULL; }
    if (bd_gate_batch)      { CUDA_CHECK(cudaFree(bd_gate_batch));      bd_gate_batch = NULL; }
    if (bd_up_batch)        { CUDA_CHECK(cudaFree(bd_up_batch));        bd_up_batch = NULL; }
    if (bd_gate_up_packed)  { CUDA_CHECK(cudaFree(bd_gate_up_packed));  bd_gate_up_packed = NULL; }
    if (bd_slot_mapping)    { CUDA_CHECK(cudaFree(bd_slot_mapping));    bd_slot_mapping = NULL; }
    if (bd_total_seq_lens)  { CUDA_CHECK(cudaFree(bd_total_seq_lens));  bd_total_seq_lens = NULL; }
    if (bd_block_tables)    { CUDA_CHECK(cudaFree(bd_block_tables));    bd_block_tables = NULL; }

    bd_capacity = num_seqs;
    bd_max_blocks_per_seq = max_blocks_per_seq;

    size_t h_bytes = (size_t)num_seqs * hidden_dim * sizeof(__nv_bfloat16);
    size_t f_bytes = (size_t)num_seqs * ffn_dim * sizeof(__nv_bfloat16);
    size_t k_bytes = (size_t)num_seqs * kv_hidden * sizeof(__nv_bfloat16);
    size_t bt_bytes = (size_t)num_seqs * max_blocks_per_seq * sizeof(int);

    CUDA_CHECK(cudaMalloc(&bd_hidden_batch,   h_bytes));
    CUDA_CHECK(cudaMalloc(&bd_residual_batch, h_bytes));
    CUDA_CHECK(cudaMalloc(&bd_batched_query,  h_bytes));
    CUDA_CHECK(cudaMalloc(&bd_batched_k,      k_bytes));
    CUDA_CHECK(cudaMalloc(&bd_batched_v,      k_bytes));
    CUDA_CHECK(cudaMalloc(&bd_batched_attn,   h_bytes));
    CUDA_CHECK(cudaMalloc(&bd_gate_batch,     f_bytes));
    CUDA_CHECK(cudaMalloc(&bd_up_batch,       f_bytes));
    CUDA_CHECK(cudaMalloc(&bd_gate_up_packed, f_bytes * 2));
    CUDA_CHECK(cudaMalloc(&bd_slot_mapping,   (size_t)num_seqs * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&bd_total_seq_lens, (size_t)num_seqs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bd_block_tables,   bt_bytes));

    decoder_note_alloc(h_bytes * 3 + f_bytes * 4 + k_bytes * 2 +
                       (size_t)num_seqs * (sizeof(int64_t) + sizeof(int)) + bt_bytes);
}

static void free_batched_buffers(void) {
    if (bd_hidden_batch)    { CUDA_CHECK(cudaFree(bd_hidden_batch));    bd_hidden_batch = NULL; }
    if (bd_residual_batch)  { CUDA_CHECK(cudaFree(bd_residual_batch));  bd_residual_batch = NULL; }
    if (bd_batched_query)   { CUDA_CHECK(cudaFree(bd_batched_query));   bd_batched_query = NULL; }
    if (bd_batched_k)       { CUDA_CHECK(cudaFree(bd_batched_k));       bd_batched_k = NULL; }
    if (bd_batched_v)       { CUDA_CHECK(cudaFree(bd_batched_v));       bd_batched_v = NULL; }
    if (bd_batched_attn)    { CUDA_CHECK(cudaFree(bd_batched_attn));    bd_batched_attn = NULL; }
    if (bd_gate_batch)      { CUDA_CHECK(cudaFree(bd_gate_batch));      bd_gate_batch = NULL; }
    if (bd_up_batch)        { CUDA_CHECK(cudaFree(bd_up_batch));        bd_up_batch = NULL; }
    if (bd_gate_up_packed)  { CUDA_CHECK(cudaFree(bd_gate_up_packed));  bd_gate_up_packed = NULL; }
    if (bd_slot_mapping)    { CUDA_CHECK(cudaFree(bd_slot_mapping));    bd_slot_mapping = NULL; }
    if (bd_total_seq_lens)  { CUDA_CHECK(cudaFree(bd_total_seq_lens));  bd_total_seq_lens = NULL; }
    if (bd_block_tables)    { CUDA_CHECK(cudaFree(bd_block_tables));    bd_block_tables = NULL; }
    bd_capacity = 0;
    bd_max_blocks_per_seq = 0;
}

extern "C" void decoder_cleanup(void)
{
    /*
     * Keep prefill/decode scratch buffers warm across scheduler steps.
     * allocate_decoder_buffers() already grows only when capacity is
     * insufficient; freeing here forces cudaFree/cudaMalloc syncs for every
     * prefill batch.
     */
}

extern "C" void decoder_cleanup_all(void)
{
    free_decoder_buffers();
    free_batched_buffers();
    decoder_scalar_cache_reset();
}

extern "C" void decoder_scratch_profile_reset(void)
{
    memset(&decoder_scratch_stats, 0, sizeof(decoder_scratch_stats));
}

extern "C" void decoder_scratch_profile_get(ScratchProfileStats *out_stats)
{
    if (out_stats == NULL) {
        return;
    }
    *out_stats = decoder_scratch_stats;
}

extern "C" void decoder_decode_kernel_profile_reset(void)
{
    memset(&decoder_decode_kernel_stats, 0, sizeof(decoder_decode_kernel_stats));
}

extern "C" void decoder_decode_kernel_profile_get(DecodeKernelProfileStats *out_stats)
{
    if (out_stats == NULL) {
        return;
    }
    *out_stats = decoder_decode_kernel_stats;
}

extern "C" void decoder_prefill_kernel_profile_reset(void)
{
    memset(&decoder_prefill_kernel_stats, 0, sizeof(decoder_prefill_kernel_stats));
}

extern "C" void decoder_prefill_kernel_profile_get(PrefillKernelProfileStats *out_stats)
{
    if (out_stats == NULL) {
        return;
    }
    *out_stats = decoder_prefill_kernel_stats;
}

/* ---- helper kernels ---- */

/* Transpose [seq, n_heads, head_dim] → [n_heads, seq, head_dim] */
__global__ void transpose_seq_heads_kernel(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int seq_len, int n_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_heads * head_dim) return;
    int s = idx / (n_heads * head_dim);
    int h = (idx / head_dim) % n_heads;
    int d = idx % head_dim;
    dst[(size_t)h * seq_len * head_dim + s * head_dim + d] =
        src[(size_t)s * n_heads * head_dim + h * head_dim + d];
}

/* Apply RoPE to Q in [seq, heads, head_dim] and write directly to
 * heads-first layout [heads, seq, head_dim]. */
__global__ void rope_transpose_q_kernel(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int pos_offset, int seq_len, int n_heads, int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * half_dim;
    if (idx >= total) return;

    int s = idx / (n_heads * half_dim);
    int h = (idx / half_dim) % n_heads;
    int d = idx % half_dim;
    float inv_freq = 1.0f / powf(10000.0f, (2.0f * d) / head_dim);
    float theta = (s + pos_offset) * inv_freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    const __nv_bfloat16 *src_head = src + (size_t)s * n_heads * head_dim + h * head_dim;
    __nv_bfloat16 *dst_head = dst + (size_t)h * seq_len * head_dim + s * head_dim;
    float v0 = __bfloat162float(src_head[d]);
    float v1 = __bfloat162float(src_head[d + half_dim]);

    dst_head[d] = __float2bfloat16(v0 * cos_t - v1 * sin_t);
    dst_head[d + half_dim] = __float2bfloat16(v0 * sin_t + v1 * cos_t);
}

/* Apply RoPE in-place to a batched decode tensor laid out as
 * [num_seqs, heads, head_dim], one token per sequence, using per-seq positions. */
__global__ void rope_batched_single_token_kernel(
    __nv_bfloat16 *tensor,
    const int *positions,
    int num_seqs,
    int n_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_seqs * n_heads * half_dim;
    if (idx >= total) return;

    int seq = idx / (n_heads * half_dim);
    int head = (idx / half_dim) % n_heads;
    int d = idx % half_dim;
    int pos = positions[seq];
    float inv_freq = 1.0f / powf(10000.0f, (2.0f * d) / head_dim);
    float theta = pos * inv_freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    __nv_bfloat16 *base = tensor + (size_t)seq * n_heads * head_dim + head * head_dim;
    float v0 = __bfloat162float(base[d]);
    float v1 = __bfloat162float(base[d + half_dim]);

    base[d] = __float2bfloat16(v0 * cos_t - v1 * sin_t);
    base[d + half_dim] = __float2bfloat16(v0 * sin_t + v1 * cos_t);
}

__global__ void split_packed_halves_kernel(
    __nv_bfloat16 *dst0,
    __nv_bfloat16 *dst1,
    const __nv_bfloat16 *src,
    int rows,
    int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx / cols;
    int col = idx % cols;
    const __nv_bfloat16 *row_ptr = src + (size_t)row * (2 * cols);
    dst0[(size_t)row * cols + col] = row_ptr[col];
    dst1[(size_t)row * cols + col] = row_ptr[cols + col];
}

__global__ void split_packed_qkv_kernel(
    __nv_bfloat16 *q_dst,
    __nv_bfloat16 *k_dst,
    __nv_bfloat16 *v_dst,
    const __nv_bfloat16 *src,
    int rows,
    int hidden_dim,
    int kv_hidden)
{
    int packed_cols = hidden_dim + 2 * kv_hidden;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * packed_cols;
    if (idx >= total) return;
    int row = idx / packed_cols;
    int col = idx - row * packed_cols;
    const __nv_bfloat16 *row_ptr = src + (size_t)row * packed_cols;
    if (col < hidden_dim) {
        q_dst[(size_t)row * hidden_dim + col] = row_ptr[col];
    } else if (col < hidden_dim + kv_hidden) {
        k_dst[(size_t)row * kv_hidden + (col - hidden_dim)] = row_ptr[col];
    } else {
        v_dst[(size_t)row * kv_hidden + (col - hidden_dim - kv_hidden)] = row_ptr[col];
    }
}

__global__ void split_packed_kv_rope_single_token_kernel(
    __nv_bfloat16 *k_dst,
    __nv_bfloat16 *v_dst,
    const __nv_bfloat16 *src,
    const int *positions,
    int rows,
    int num_kv_heads,
    int head_dim)
{
    int half_dim = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * num_kv_heads * half_dim;
    if (idx >= total) return;

    int row = idx / (num_kv_heads * half_dim);
    int head = (idx / half_dim) % num_kv_heads;
    int d = idx % half_dim;
    int pos = positions[row];
    int kv_hidden = num_kv_heads * head_dim;
    float inv_freq = 1.0f / powf(10000.0f, (2.0f * d) / head_dim);
    float theta = pos * inv_freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    const __nv_bfloat16 *row_ptr = src + (size_t)row * (2 * kv_hidden);
    const __nv_bfloat16 *k_src = row_ptr + head * head_dim;
    const __nv_bfloat16 *v_src = row_ptr + kv_hidden + head * head_dim;
    __nv_bfloat16 *k_out = k_dst + (size_t)row * kv_hidden + head * head_dim;
    __nv_bfloat16 *v_out = v_dst + (size_t)row * kv_hidden + head * head_dim;

    float k0 = __bfloat162float(k_src[d]);
    float k1 = __bfloat162float(k_src[d + half_dim]);
    k_out[d] = __float2bfloat16(k0 * cos_t - k1 * sin_t);
    k_out[d + half_dim] = __float2bfloat16(k0 * sin_t + k1 * cos_t);

    v_out[d] = v_src[d];
    v_out[d + half_dim] = v_src[d + half_dim];
}

/* Merge [n_heads, seq, head_dim] → [seq, n_heads*head_dim] */
__global__ void merge_attn_heads_kernel(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int seq_len, int n_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_heads * head_dim) return;
    int s = idx / (n_heads * head_dim);
    int h = (idx / head_dim) % n_heads;
    int d = idx % head_dim;
    dst[(size_t)s * n_heads * head_dim + h * head_dim + d] =
        src[(size_t)h * seq_len * head_dim + s * head_dim + d];
}

/* Extract KV head: [seq, kv_heads, head_dim] → [seq, head_dim] */
__global__ void extract_kv_head_kernel(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int seq_len, int kv_heads, int head_dim, int kv_head_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int s = idx / head_dim;
    int d = idx % head_dim;
    dst[s * head_dim + d] =
        src[(size_t)s * kv_heads * head_dim + kv_head_idx * head_dim + d];
}

/* Causal mask: zero-out future positions in score slice [query_len, key_len]
 * q_offset = position of first query token in the full sequence */
__global__ void causal_mask_kernel(
    __nv_bfloat16 *scores, int query_len, int key_len, int q_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= query_len * key_len) return;
    int q = idx / key_len;
    int k = idx % key_len;
    if (k > q_offset + q)
        scores[idx] = __float2bfloat16(-1e9f);
}

/* ---- GQA attention ---- */
/*
 * q_buf   : [q_heads, query_len, head_dim]   (heads-first)
 * k_cache : [key_len, kv_hidden]             (seq-first, kv_hidden = kv_heads*head_dim)
 * v_cache : [key_len, kv_hidden]
 * attn_out: [query_len, hidden_dim]          output
 * q_offset: abs position of first query token (for causal mask)
 */
static void gqa_attention(
    cublasHandle_t handle,
    __nv_bfloat16 *q_buf,
    const __nv_bfloat16 *k_cache,
    const __nv_bfloat16 *v_cache,
    __nv_bfloat16 *attn_out,
    int query_len, int key_len,
    int q_heads, int kv_heads, int head_dim,
    bool causal, int q_offset)
{
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float alpha = scale, beta = 0.0f;
    const float a1 = 1.0f,    b1 = 0.0f;
    int q_ratio = q_heads / kv_heads;
    long long stride_q = (long long)query_len * head_dim;
    long long stride_score = (long long)query_len * key_len;
    long long stride_out = head_dim;
    int kv_hidden = kv_heads * head_dim;
    int hidden_dim = q_heads * head_dim;

    /* ---------- Q @ K^T ---------- */
    for (int kv_h = 0; kv_h < kv_heads; kv_h++) {
        int h_base = kv_h * q_ratio;
        const __nv_bfloat16 *K_h = k_cache + kv_h * head_dim;
        __nv_bfloat16 *Q_base = q_buf + (size_t)h_base * query_len * head_dim;
        __nv_bfloat16 *score_base = d_dec_qk_buf + (size_t)h_base * query_len * key_len;

        /*
         * score_h[query_len, key_len] = Q_h[query_len, head_dim] @ K_h^T[head_dim, key_len]
         * Process all query heads that share the same KV head in one strided batched GEMM.
         */
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            key_len, query_len, head_dim,
            &alpha,
            K_h,         CUDA_R_16BF, kv_hidden, 0,
            Q_base,      CUDA_R_16BF, head_dim, stride_q,
            &beta,
            score_base,  CUDA_R_16BF, key_len, stride_score,
            q_ratio, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS Q@K^T failed kv_head=%d: %d\n", kv_h, st);
            exit(1);
        }

        for (int q_sub = 0; q_sub < q_ratio; q_sub++) {
            __nv_bfloat16 *score_h = score_base + (size_t)q_sub * query_len * key_len;
            if (causal) {
                int n_score = query_len * key_len;
                causal_mask_kernel<<<(n_score+255)/256, 256>>>(
                    score_h, query_len, key_len, q_offset);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        softmax_inplace(score_base, q_ratio * query_len, key_len);
    }

    /* ---------- score @ V ---------- */
    for (int kv_h = 0; kv_h < kv_heads; kv_h++) {
        int h_base = kv_h * q_ratio;
        const __nv_bfloat16 *V_h = v_cache + kv_h * head_dim;
        __nv_bfloat16 *score_base = d_dec_qk_buf + (size_t)h_base * query_len * key_len;
        __nv_bfloat16 *out_base = attn_out + (size_t)h_base * head_dim;

        /*
         * out_h[query_len, head_dim] = score_h[query_len, key_len] @ V_h[key_len, head_dim]
         * Reuse the same V head across all query heads mapped to this KV head and write
         * directly into the seq-first [query_len, hidden_dim] output buffer.
         */
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, query_len, key_len,
            &a1,
            V_h,         CUDA_R_16BF, kv_hidden, 0,
            score_base,  CUDA_R_16BF, key_len, stride_score,
            &b1,
            out_base,    CUDA_R_16BF, hidden_dim, stride_out,
            q_ratio, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS score@V failed kv_head=%d: %d\n", kv_h, st);
            exit(1);
        }
    }
}

/* ---- weight lookup helpers ---- */
static inline const Tensor *wget(const WeightStore *ws, const char *name)
{
    const Tensor *t = ws_get_const(ws, name);
    if (!t) { fprintf(stderr, "Weight not found: %s\n", name); exit(1); }
    return t;
}
static inline const Tensor *wget_opt(const WeightStore *ws, const char *name)
{
    return ws_get_const(ws, name);
}

/* ---- linear wrapper with named weight lookup ---- */
static void linear_proj(
    cublasHandle_t handle,
    __nv_bfloat16 *out,
    __nv_bfloat16 *weight_scratch,
    __nv_bfloat16 *prequant_scratch,
    const __nv_bfloat16 *x,
    const WeightStore *ws,
    const char *w_name, const char *ws_name, const char *ws2_name,
    const char *in_name, const char *pre_name,
    int M, int N, int K,
    bool use_native_nvfp4)
{
    const Tensor *tw   = wget(ws, w_name);
    const Tensor *tws  = wget_opt(ws, ws_name);
    const Tensor *tws2 = wget_opt(ws, ws2_name);
    const Tensor *tin  = wget_opt(ws, in_name);
    const Tensor *tpre = wget_opt(ws, pre_name);

    float input_scale = decoder_scalar_read_cached(tin, 1.0f);
    float global_scale = decoder_scalar_read_cached(tws2, 1.0f);

    if (tw->dtype == DTYPE_BF16) {
        bf16_linear(handle, out, x, (__nv_bfloat16 *)tw->data, NULL, M, N, K);
        return;
    }

    CHECK(tw->dtype == DTYPE_UINT8, "Unsupported decoder weight dtype %d for %s", tw->dtype, w_name);

    if (use_native_nvfp4 && prefer_nvfp4_cutlass()) {
        compare_nvfp4_once(handle, out, weight_scratch, prequant_scratch, x,
                           (uint8_t *)tw->data,
                           tws ? (uint8_t *)tws->data : NULL,
                           input_scale,
                           global_scale,
                           tpre ? (__nv_bfloat16 *)tpre->data : NULL,
                           M, N, K, w_name);
        if (should_compare_nvfp4_once()) {
            return;
        }
        int status = nvfp4_linear(handle, out, weight_scratch, x,
                                  (uint8_t *)tw->data,
                                  tws ? (uint8_t *)tws->data : NULL,
                                  input_scale,
                                  global_scale,
                                  tpre ? (__nv_bfloat16 *)tpre->data : NULL,
                                  NULL,
                                  M, N, K);
        if (status == 0) {
            return;
        }
        warn_nvfp4_cutlass_fallback();
    }

    fp4_linear(handle, out, weight_scratch, prequant_scratch, x,
               (uint8_t *)tw->data,
               tws ? (uint8_t *)tws->data : NULL,
               global_scale,
               tpre ? (__nv_bfloat16 *)tpre->data : NULL,
               NULL,
               M, N, K);
}

static int linear_proj_pair_same_input(
    cublasHandle_t handle,
    __nv_bfloat16 *out_packed,
    __nv_bfloat16 *weight_scratch,
    const __nv_bfloat16 *x,
    const WeightStore *ws,
    const char *w0_name, const char *ws0_name, const char *ws20_name, const char *pre0_name,
    const char *w1_name, const char *ws1_name, const char *ws21_name, const char *pre1_name,
    int M, int N, int K)
{
    const Tensor *tw0   = wget(ws, w0_name);
    const Tensor *tws0  = wget_opt(ws, ws0_name);
    const Tensor *tws20 = wget_opt(ws, ws20_name);
    const Tensor *tpre0 = wget_opt(ws, pre0_name);
    const Tensor *tw1   = wget(ws, w1_name);
    const Tensor *tws1  = wget_opt(ws, ws1_name);
    const Tensor *tws21 = wget_opt(ws, ws21_name);
    const Tensor *tpre1 = wget_opt(ws, pre1_name);
    __nv_bfloat16 *weight0_scratch = weight_scratch;
    __nv_bfloat16 *weight1_scratch = weight_scratch + (size_t)N * K;

    /* BF16 fusion: copy both weights into scratch for one wide GEMM */
    if (tw0->dtype == DTYPE_BF16 && tw1->dtype == DTYPE_BF16) {
        if (weight_scratch == NULL) return 0;
        CUDA_CHECK(cudaMemcpy(weight0_scratch, tw0->data,
                              (size_t)N * K * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(weight1_scratch, tw1->data,
                              (size_t)N * K * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        bf16_linear(handle, out_packed, x, weight0_scratch, NULL, M, 2 * N, K);
        return 1;
    }

    if (tw0->dtype != DTYPE_UINT8 || tw1->dtype != DTYPE_UINT8) {
        return 0;
    }

    if (tpre0 != NULL || tpre1 != NULL) {
        return 0;
    }

    fp4_dequant_to_bf16(weight0_scratch,
                        (const uint8_t *)tw0->data,
                        tws0 ? (const uint8_t *)tws0->data : NULL,
                        decoder_scalar_read_cached(tws20, 1.0f),
                        N, K);
    fp4_dequant_to_bf16(weight1_scratch,
                        (const uint8_t *)tw1->data,
                        tws1 ? (const uint8_t *)tws1->data : NULL,
                        decoder_scalar_read_cached(tws21, 1.0f),
                        N, K);

    bf16_linear(handle, out_packed, x, weight0_scratch, NULL, M, 2 * N, K);
    return 1;
}

static int linear_proj_triple_same_input_bf16(
    cublasHandle_t handle,
    __nv_bfloat16 *out_packed,
    __nv_bfloat16 *weight_scratch,
    const __nv_bfloat16 *x,
    const WeightStore *ws,
    const char *w0_name,
    const char *w1_name,
    const char *w2_name,
    int M, int N0, int N1, int K)
{
    const Tensor *tw0 = wget(ws, w0_name);
    const Tensor *tw1 = wget(ws, w1_name);
    const Tensor *tw2 = wget(ws, w2_name);
    if (tw0->dtype != DTYPE_BF16 || tw1->dtype != DTYPE_BF16 || tw2->dtype != DTYPE_BF16) {
        return 0;
    }
    if (weight_scratch == NULL) return 0;

    __nv_bfloat16 *weight0_scratch = weight_scratch;
    __nv_bfloat16 *weight1_scratch = weight_scratch + (size_t)N0 * K;
    __nv_bfloat16 *weight2_scratch = weight1_scratch + (size_t)N1 * K;
    CUDA_CHECK(cudaMemcpy(weight0_scratch, tw0->data,
                          (size_t)N0 * K * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(weight1_scratch, tw1->data,
                          (size_t)N1 * K * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(weight2_scratch, tw2->data,
                          (size_t)N1 * K * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToDevice));
    bf16_linear(handle, out_packed, x, weight0_scratch, NULL, M, N0 + 2 * N1, K);
    return 1;
}

static void get_paged_layer_cache(
    KVPool *pool,
    int layer,
    __nv_bfloat16 **k_base,
    __nv_bfloat16 **v_base)
{
    *k_base = (__nv_bfloat16 *)pool->k_cache + (size_t)layer * pool->layer_stride;
    *v_base = (__nv_bfloat16 *)pool->v_cache + (size_t)layer * pool->layer_stride;
}

extern "C" {

/* ---- decoder_prefill ---- */
void decoder_prefill(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int seq_len,
    KVCache *kv,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    int hidden_dim  = cfg->dec_hidden;    /* 2048 */
    int num_layers  = cfg->dec_layers;    /* 28   */
    int num_heads   = cfg->dec_heads;     /* 16   */
    int num_kv_heads= cfg->dec_kv_heads;  /* 4    */
    int head_dim    = cfg->dec_head_dim;  /* 128  */
    int ffn_dim     = cfg->dec_ffn;       /* 6144 */
    int kv_hidden   = num_kv_heads * head_dim; /* 512 */

    allocate_decoder_buffers(seq_len, seq_len, hidden_dim, ffn_dim,
                             num_heads, num_kv_heads, head_dim);

    kv->seq_len = seq_len;

    __nv_bfloat16 *layer_out = hidden;
    DecoderLinearScratchPlan scratch = decoder_make_linear_scratch_plan(dequant_buf, cfg);
    char w[256], ws_n[256], ws2[256], in_s[256], pre[256];

    for (int layer = 0; layer < num_layers; layer++) {
        int n = seq_len * hidden_dim;

        /* ===== Self-Attention ===== */

        /* Save residual */
        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        /* 1. Input RMSNorm */
        snprintf(w, sizeof(w), "language_model.model.layers.%d.input_layernorm.weight", layer);
        const Tensor *tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        /* 2. Q projection → d_dec_hidden_buf [seq, hidden_dim], layout [seq, q_heads, head_dim] */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.q_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.q_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.q_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.q_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.q_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_hidden_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);

        /* 3-4. Fuse Q RoPE + transpose into heads-first layout */
        int total_q_pairs = seq_len * num_heads * (head_dim / 2);
        rope_transpose_q_kernel<<<(total_q_pairs+255)/256, 256>>>(
            d_dec_q_buf, d_dec_hidden_buf, 0, seq_len, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());

        /* 5. K projection → d_dec_k_buf [seq, kv_hidden] */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.k_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.k_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.k_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.k_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.k_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_k_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);

        /* 6. Apply RoPE to K in [seq, kv_heads, head_dim] */
        rope_decoder(NULL, d_dec_k_buf, 0, seq_len, 0, num_kv_heads, head_dim);

        /* 7. V projection → d_dec_v_buf [seq, kv_hidden] */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.v_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.v_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.v_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.v_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.v_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_v_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);

        /* 8. Store K, V in cache (layout: [seq, kv_hidden]) */
        LayerKV *lkv = &kv->layers[layer];
        CUDA_CHECK(cudaMemcpy(lkv->k.data, d_dec_k_buf,
                              (size_t)seq_len * kv_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(lkv->v.data, d_dec_v_buf,
                              (size_t)seq_len * kv_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));

        /* 9. GQA attention (causal, q_offset=0) */
        gqa_attention(handle, d_dec_q_buf,
                      (__nv_bfloat16 *)lkv->k.data,
                      (__nv_bfloat16 *)lkv->v.data,
                      d_dec_attn_out,
                      seq_len, seq_len,
                      num_heads, num_kv_heads, head_dim,
                      true, 0);

        /* 10. O projection */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.o_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.o_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.o_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.o_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.o_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_attn_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);

        /* 11. Residual add */
        residual_add(layer_out, layer_out, d_dec_residual, n);

        /* ===== MLP ===== */

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        /* 12. Post-attention RMSNorm */
        snprintf(w, sizeof(w), "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        /* 13. Gate projection */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.gate_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.gate_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.gate_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.gate_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.gate_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_gate_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        /* 14. Up projection */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.up_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.up_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.up_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.up_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.up_proj.pre_quant_scale", layer);
        ensure_decoder_up_buf();
        linear_proj(handle, d_dec_up_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        /* 15. SiLU(gate) * up */
        silu_elementwise_mul(d_dec_gate_buf, d_dec_up_buf, seq_len * ffn_dim);

        /* 16. Down projection */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.down_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.down_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.down_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.down_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.down_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_gate_buf, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, ffn_dim, true);

        /* 17. Residual add */
        residual_add(layer_out, layer_out, d_dec_residual, n);
    }

    /* Final RMSNorm */
    const Tensor *t_norm = wget(ws, "language_model.model.norm.weight");
    rmsnorm(hidden, hidden, (__nv_bfloat16 *)t_norm->data, 1e-5f, seq_len, hidden_dim);
}

void decoder_prefill_paged(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int seq_len,
    KVPool *pool,
    BlockTable *bt,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    KVCache temp_kv;
    LayerKV *temp_layers = (LayerKV *)calloc(cfg->dec_layers, sizeof(LayerKV));
    int first_block = bt->h_table[0];

    CHECK(temp_layers != NULL, "Failed to allocate temporary paged KV view");

    temp_kv.layers = temp_layers;
    temp_kv.seq_len = 0;
    temp_kv.max_seq = bt->num_blocks * pool->block_size;

    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        __nv_bfloat16 *k_base = (__nv_bfloat16 *)pool->k_cache +
                                (size_t)layer * pool->layer_stride +
                                (size_t)first_block * pool->block_stride;
        __nv_bfloat16 *v_base = (__nv_bfloat16 *)pool->v_cache +
                                (size_t)layer * pool->layer_stride +
                                (size_t)first_block * pool->block_stride;
        temp_layers[layer].k.data = k_base;
        temp_layers[layer].v.data = v_base;
    }

    decoder_prefill(handle, ws, hidden, seq_len, &temp_kv, dequant_buf, cfg);
    free(temp_layers);
}

void decoder_prefill_paged_batched(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    int total_tokens,
    int num_seqs,
    const int *seq_lens,
    const int *seq_offsets,
    KVPool *pool,
    BlockTable *bts,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    int hidden_dim = cfg->dec_hidden;
    int num_layers = cfg->dec_layers;
    int num_heads = cfg->dec_heads;
    int num_kv_heads = cfg->dec_kv_heads;
    int head_dim = cfg->dec_head_dim;
    int ffn_dim = cfg->dec_ffn;
    int kv_hidden = num_kv_heads * head_dim;
    int max_seq_len = 0;
    int64_t *h_slot_mapping = NULL;
    int64_t *d_slot_mapping = NULL;
    __nv_bfloat16 *layer_out = hidden;
    DecoderLinearScratchPlan scratch = decoder_make_linear_scratch_plan(dequant_buf, cfg);
    int profiling_enabled = decoder_profile_enabled();
    cudaEvent_t phase_start = NULL;
    cudaEvent_t phase_stop = NULL;
    uint64_t qkv_ns_total = 0;
    uint64_t cache_ns_total = 0;
    uint64_t attn_ns_total = 0;
    uint64_t o_proj_ns_total = 0;
    uint64_t mlp_ns_total = 0;
    char w[256], ws_n[256], ws2[256], in_s[256], pre[256];

    CHECK(total_tokens > 0, "decoder_prefill_paged_batched requires tokens");
    CHECK(num_seqs > 0, "decoder_prefill_paged_batched requires sequences");
    CHECK(seq_lens != NULL && seq_offsets != NULL && pool != NULL && bts != NULL,
          "decoder_prefill_paged_batched received null metadata");

    for (int seq = 0; seq < num_seqs; seq++) {
        if (seq_lens[seq] > max_seq_len) {
            max_seq_len = seq_lens[seq];
        }
    }
    CHECK(max_seq_len > 0, "decoder_prefill_paged_batched max_seq_len invalid");

    allocate_decoder_buffers(total_tokens, max_seq_len, hidden_dim, ffn_dim,
                             num_heads, num_kv_heads, head_dim);

    if (profiling_enabled) {
        CUDA_CHECK(cudaEventCreate(&phase_start));
        CUDA_CHECK(cudaEventCreate(&phase_stop));
    }

    h_slot_mapping = (int64_t *) malloc((size_t) total_tokens * sizeof(*h_slot_mapping));
    CHECK(h_slot_mapping != NULL, "Failed to allocate batched prefill slot mapping");
    CUDA_CHECK(cudaMalloc(&d_slot_mapping, (size_t) total_tokens * sizeof(*d_slot_mapping)));
    decoder_note_alloc((size_t) total_tokens * sizeof(*d_slot_mapping));

    for (int seq = 0; seq < num_seqs; seq++) {
        int seq_len = seq_lens[seq];
        int seq_offset = seq_offsets[seq];
        const int *block_ids = bts[seq].h_table;
        int num_blocks = bts[seq].num_blocks;

        for (int tok = 0; tok < seq_len; tok++) {
            int block_idx = tok / pool->block_size;
            int block_offset = tok % pool->block_size;
            CHECK(block_idx >= 0 && block_idx < num_blocks,
                  "Prefill block table too small for seq=%d tok=%d", seq, tok);
            h_slot_mapping[seq_offset + tok] =
                (int64_t) block_ids[block_idx] * (int64_t) pool->block_size + (int64_t) block_offset;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_slot_mapping, h_slot_mapping,
                          (size_t) total_tokens * sizeof(*d_slot_mapping),
                          cudaMemcpyHostToDevice));

    int use_dec_fa2 = flash_fwd_vllm_gqa_available();
    if (getenv("GLMASR_DECODER_FA_DISABLE")) use_dec_fa2 = 0;
    if (use_dec_fa2) {
        int cu_need = num_seqs + 1;
        if (cu_need > d_dec_cu_seqlens_cap) {
            if (d_dec_cu_seqlens) { cudaFree(d_dec_cu_seqlens); d_dec_cu_seqlens = NULL; }
            CUDA_CHECK(cudaMalloc(&d_dec_cu_seqlens, (size_t) cu_need * sizeof(int)));
            d_dec_cu_seqlens_cap = cu_need;
        }
        int *h_cu = (int *) malloc((size_t) cu_need * sizeof(int));
        CHECK(h_cu != NULL, "Failed to allocate cu_seqlens host buffer");
        h_cu[0] = 0;
        for (int seq = 0; seq < num_seqs; seq++) h_cu[seq + 1] = h_cu[seq] + seq_lens[seq];
        CUDA_CHECK(cudaMemcpy(d_dec_cu_seqlens, h_cu, (size_t) cu_need * sizeof(int),
                               cudaMemcpyHostToDevice));
        free(h_cu);
        if (decoder_profile_enabled()) {
            fprintf(stderr, "decoder FA2: enabled (seqs=%d tokens=%d max_seq=%d)\n",
                    num_seqs, total_tokens, max_seq_len);
        }
    }

    for (int layer = 0; layer < num_layers; layer++) {
        int n_hidden = total_tokens * hidden_dim;

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              (size_t) n_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.input_layernorm.weight", layer);
        const Tensor *tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, total_tokens, hidden_dim);

        int qkv_rope_done = 0;
        {
            char q_w[256], q_ws[256], q_ws2[256], q_in[256], q_pre[256];
            char k_w[256], k_ws[256], k_ws2[256], k_in[256], k_pre[256];
            char v_w[256], v_ws[256], v_ws2[256], v_in[256], v_pre[256];
            int fused_qkv = 0;
            int fused_kv = 0;

            snprintf(q_w,   sizeof(q_w),   "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
            snprintf(q_ws,  sizeof(q_ws),  "language_model.model.layers.%d.self_attn.q_proj.weight_scale", layer);
            snprintf(q_ws2, sizeof(q_ws2), "language_model.model.layers.%d.self_attn.q_proj.weight_scale_2", layer);
            snprintf(q_in,  sizeof(q_in),  "language_model.model.layers.%d.self_attn.q_proj.input_scale", layer);
            snprintf(q_pre, sizeof(q_pre), "language_model.model.layers.%d.self_attn.q_proj.pre_quant_scale", layer);

            snprintf(k_w,   sizeof(k_w),   "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
            snprintf(k_ws,  sizeof(k_ws),  "language_model.model.layers.%d.self_attn.k_proj.weight_scale", layer);
            snprintf(k_ws2, sizeof(k_ws2), "language_model.model.layers.%d.self_attn.k_proj.weight_scale_2", layer);
            snprintf(k_in,  sizeof(k_in),  "language_model.model.layers.%d.self_attn.k_proj.input_scale", layer);
            snprintf(k_pre, sizeof(k_pre), "language_model.model.layers.%d.self_attn.k_proj.pre_quant_scale", layer);

            snprintf(v_w,   sizeof(v_w),   "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
            snprintf(v_ws,  sizeof(v_ws),  "language_model.model.layers.%d.self_attn.v_proj.weight_scale", layer);
            snprintf(v_ws2, sizeof(v_ws2), "language_model.model.layers.%d.self_attn.v_proj.weight_scale_2", layer);
            snprintf(v_in,  sizeof(v_in),  "language_model.model.layers.%d.self_attn.v_proj.input_scale", layer);
            snprintf(v_pre, sizeof(v_pre), "language_model.model.layers.%d.self_attn.v_proj.pre_quant_scale", layer);

            fused_qkv = linear_proj_triple_same_input_bf16(
                handle, d_dec_gate_up_packed, scratch.weight0, layer_out, ws,
                q_w, k_w, v_w, total_tokens, hidden_dim, kv_hidden, hidden_dim);
            if (fused_qkv) {
                if (use_dec_fa2) {
                    split_packed_qkv_rope_decoder_batched(
                        d_dec_hidden_buf, d_dec_k_buf, d_dec_v_buf,
                        d_dec_gate_up_packed, d_dec_cu_seqlens,
                        num_seqs, max_seq_len, num_heads, num_kv_heads, head_dim);
                    qkv_rope_done = 1;
                } else {
                    int total_qkv = total_tokens * (hidden_dim + 2 * kv_hidden);
                    split_packed_qkv_kernel<<<(total_qkv + 255) / 256, 256>>>(
                        d_dec_hidden_buf, d_dec_k_buf, d_dec_v_buf,
                        d_dec_gate_up_packed, total_tokens, hidden_dim, kv_hidden);
                    CUDA_CHECK(cudaGetLastError());
                }
            } else {
                linear_proj(handle, d_dec_hidden_buf, scratch.weight0, scratch.aux, layer_out, ws,
                         q_w, q_ws, q_ws2, q_in, q_pre, total_tokens, hidden_dim, hidden_dim, true);
                fused_kv = linear_proj_pair_same_input(handle, d_dec_gate_up_packed, scratch.weight0, layer_out, ws,
                                                       k_w, k_ws, k_ws2, k_pre,
                                                       v_w, v_ws, v_ws2, v_pre,
                                                       total_tokens, kv_hidden, hidden_dim);
                if (fused_kv) {
                    int total_kv = total_tokens * kv_hidden;
                    split_packed_halves_kernel<<<(total_kv + 255) / 256, 256>>>(
                        d_dec_k_buf, d_dec_v_buf, d_dec_gate_up_packed, total_tokens, kv_hidden);
                    CUDA_CHECK(cudaGetLastError());
                } else {
                linear_proj(handle, d_dec_k_buf, scratch.weight0, scratch.aux, layer_out, ws,
                         k_w, k_ws, k_ws2, k_in, k_pre, total_tokens, kv_hidden, hidden_dim, true);
                linear_proj(handle, d_dec_v_buf, scratch.weight0, scratch.aux, layer_out, ws,
                         v_w, v_ws, v_ws2, v_in, v_pre, total_tokens, kv_hidden, hidden_dim, true);
                }
            }
        }
        if (profiling_enabled) {
            CUDA_CHECK(cudaEventRecord(phase_start));
            CUDA_CHECK(cudaEventRecord(phase_stop));
            qkv_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        if (use_dec_fa2) {
            /* Batched RoPE: one launch for Q, one for K (vs 2*num_seqs per layer) */
            if (!qkv_rope_done) {
                rope_decoder_batched(d_dec_hidden_buf, d_dec_cu_seqlens,
                                     num_seqs, max_seq_len, num_heads, head_dim);
                rope_decoder_batched(d_dec_k_buf, d_dec_cu_seqlens,
                                     num_seqs, max_seq_len, num_kv_heads, head_dim);
            }
            int fa2_ok = flash_fwd_vllm_gqa_launch(
                (void *) d_dec_hidden_buf, (void *) d_dec_k_buf,
                (void *) d_dec_v_buf, (void *) d_dec_attn_out,
                total_tokens, max_seq_len,
                num_heads, num_kv_heads, head_dim,
                hidden_dim, kv_hidden, kv_hidden,
                num_seqs,
                d_dec_cu_seqlens, 1);
            if (!fa2_ok) {
                fprintf(stderr, "decoder FA2 launch failed, falling back\n");
                use_dec_fa2 = 0;
            }
        }
        if (!use_dec_fa2) {
            for (int seq = 0; seq < num_seqs; seq++) {
                int seq_len = seq_lens[seq];
                int seq_offset = seq_offsets[seq];
                __nv_bfloat16 *q_seq = d_dec_hidden_buf + (size_t) seq_offset * hidden_dim;
                __nv_bfloat16 *k_seq = d_dec_k_buf + (size_t) seq_offset * kv_hidden;
                __nv_bfloat16 *v_seq = d_dec_v_buf + (size_t) seq_offset * kv_hidden;
                __nv_bfloat16 *attn_seq_out = d_dec_attn_out + (size_t) seq_offset * hidden_dim;
                int total_q = seq_len * num_heads * head_dim;

                rope_transpose_q_kernel<<<(total_q + 255) / 256, 256>>>(
                    d_dec_q_buf, q_seq, 0, seq_len, num_heads, head_dim);
                CUDA_CHECK(cudaGetLastError());
                rope_decoder(NULL, k_seq, 0, seq_len, 0, num_kv_heads, head_dim);

                gqa_attention(handle, d_dec_q_buf, k_seq, v_seq, attn_seq_out,
                              seq_len, seq_len, num_heads, num_kv_heads, head_dim,
                              true, 0);
            }
        }
        {
            __nv_bfloat16 *k_base = NULL;
            __nv_bfloat16 *v_base = NULL;
            get_paged_layer_cache(pool, layer, &k_base, &v_base);
            reshape_and_cache((uint16_t *) k_base, (uint16_t *) v_base,
                              (const uint16_t *) d_dec_k_buf, (const uint16_t *) d_dec_v_buf,
                              d_slot_mapping,
                              total_tokens, num_kv_heads, head_dim, pool->block_size);
        }
        if (profiling_enabled) {
            cache_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.o_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.o_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.o_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.o_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.o_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_attn_out, ws,
                 w, ws_n, ws2, in_s, pre, total_tokens, hidden_dim, hidden_dim, true);
        if (profiling_enabled) {
            attn_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        residual_add(layer_out, layer_out, d_dec_residual, n_hidden);

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              (size_t) n_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, total_tokens, hidden_dim);

        {
            char gate_w[256], gate_ws[256], gate_ws2[256], gate_in[256], gate_pre[256];
            char up_w[256], up_ws[256], up_ws2[256], up_in[256], up_pre[256];
            int fused_gate_up = 0;

            snprintf(gate_w,   sizeof(gate_w),   "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
            snprintf(gate_ws,  sizeof(gate_ws),  "language_model.model.layers.%d.mlp.gate_proj.weight_scale", layer);
            snprintf(gate_ws2, sizeof(gate_ws2), "language_model.model.layers.%d.mlp.gate_proj.weight_scale_2", layer);
            snprintf(gate_in,  sizeof(gate_in),  "language_model.model.layers.%d.mlp.gate_proj.input_scale", layer);
            snprintf(gate_pre, sizeof(gate_pre), "language_model.model.layers.%d.mlp.gate_proj.pre_quant_scale", layer);

            snprintf(up_w,   sizeof(up_w),   "language_model.model.layers.%d.mlp.up_proj.weight", layer);
            snprintf(up_ws,  sizeof(up_ws),  "language_model.model.layers.%d.mlp.up_proj.weight_scale", layer);
            snprintf(up_ws2, sizeof(up_ws2), "language_model.model.layers.%d.mlp.up_proj.weight_scale_2", layer);
            snprintf(up_in,  sizeof(up_in),  "language_model.model.layers.%d.mlp.up_proj.input_scale", layer);
            snprintf(up_pre, sizeof(up_pre), "language_model.model.layers.%d.mlp.up_proj.pre_quant_scale", layer);

            fused_gate_up = linear_proj_pair_same_input(handle, d_dec_gate_up_packed, scratch.weight0, layer_out, ws,
                                                     gate_w, gate_ws, gate_ws2, gate_pre,
                                                     up_w, up_ws, up_ws2, up_pre,
                                                     total_tokens, ffn_dim, hidden_dim);
            if (fused_gate_up) {
                silu_mul_split_packed(d_dec_gate_buf, d_dec_gate_up_packed, total_tokens, ffn_dim);
            } else {
                ensure_decoder_up_buf();
                linear_proj(handle, d_dec_gate_buf, scratch.weight0, scratch.aux, layer_out, ws,
                         gate_w, gate_ws, gate_ws2, gate_in, gate_pre, total_tokens, ffn_dim, hidden_dim, true);
                linear_proj(handle, d_dec_up_buf, scratch.weight0, scratch.aux, layer_out, ws,
                         up_w, up_ws, up_ws2, up_in, up_pre, total_tokens, ffn_dim, hidden_dim, true);
                silu_elementwise_mul(d_dec_gate_buf, d_dec_up_buf, total_tokens * ffn_dim);
            }
        }

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.down_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.down_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.down_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.down_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.down_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_gate_buf, ws,
                 w, ws_n, ws2, in_s, pre, total_tokens, hidden_dim, ffn_dim, true);
        if (profiling_enabled) {
            o_proj_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        residual_add(layer_out, layer_out, d_dec_residual, n_hidden);
        if (profiling_enabled) {
            CUDA_CHECK(cudaEventRecord(phase_stop));
            mlp_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
        }
    }

    {
        const Tensor *t_norm = wget(ws, "language_model.model.norm.weight");
        rmsnorm(hidden, hidden, (__nv_bfloat16 *)t_norm->data, 1e-5f, total_tokens, hidden_dim);
    }

    CUDA_CHECK(cudaFree(d_slot_mapping));
    decoder_note_free((size_t) total_tokens * sizeof(*d_slot_mapping));
    free(h_slot_mapping);
    if (profiling_enabled) {
        decoder_profile_record_duration(qkv_ns_total,
                                        &decoder_prefill_kernel_stats.qkv_ns_total,
                                        &decoder_prefill_kernel_stats.qkv_ns_max);
        decoder_profile_record_duration(cache_ns_total,
                                        &decoder_prefill_kernel_stats.cache_ns_total,
                                        &decoder_prefill_kernel_stats.cache_ns_max);
        decoder_profile_record_duration(attn_ns_total,
                                        &decoder_prefill_kernel_stats.attn_ns_total,
                                        &decoder_prefill_kernel_stats.attn_ns_max);
        decoder_profile_record_duration(o_proj_ns_total,
                                        &decoder_prefill_kernel_stats.o_proj_ns_total,
                                        &decoder_prefill_kernel_stats.o_proj_ns_max);
        decoder_profile_record_duration(mlp_ns_total,
                                        &decoder_prefill_kernel_stats.mlp_ns_total,
                                        &decoder_prefill_kernel_stats.mlp_ns_max);
        CUDA_CHECK(cudaEventDestroy(phase_start));
        CUDA_CHECK(cudaEventDestroy(phase_stop));
    }
}

void decoder_decode_step(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,       /* single token [1, hidden_dim] */
    KVCache *kv,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    int hidden_dim   = cfg->dec_hidden;
    int num_layers   = cfg->dec_layers;
    int num_heads    = cfg->dec_heads;
    int num_kv_heads = cfg->dec_kv_heads;
    int head_dim     = cfg->dec_head_dim;
    int ffn_dim      = cfg->dec_ffn;
    int kv_hidden    = num_kv_heads * head_dim;
    int seq_len      = 1;
    int cached_len   = kv->seq_len;   /* tokens already in cache */

    __nv_bfloat16 *layer_out = hidden;
    DecoderLinearScratchPlan scratch = decoder_make_linear_scratch_plan(dequant_buf, cfg);
    char w[256], ws_n[256], ws2[256], in_s[256], pre[256];

    for (int layer = 0; layer < num_layers; layer++) {
        int n = seq_len * hidden_dim;

        /* ===== Self-Attention ===== */

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.input_layernorm.weight", layer);
        const Tensor *tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        /* Q proj → d_dec_hidden_buf, rope, transpose → d_dec_q_buf */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.q_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.q_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.q_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.q_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.q_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_hidden_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);
        int total_q = seq_len * num_heads * (head_dim / 2);
        rope_transpose_q_kernel<<<(total_q+255)/256, 256>>>(
            d_dec_q_buf, d_dec_hidden_buf, cached_len, seq_len, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());

        /* K proj → d_dec_k_buf, rope */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.k_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.k_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.k_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.k_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.k_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_k_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);
        rope_decoder(NULL, d_dec_k_buf, cached_len, seq_len, 0, num_kv_heads, head_dim);

        /* V proj → d_dec_v_buf */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.v_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.v_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.v_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.v_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.v_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_v_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);

        /* Append K, V to cache at position cached_len */
        LayerKV *lkv = &kv->layers[layer];
        __nv_bfloat16 *k_cache_ptr = (__nv_bfloat16 *)lkv->k.data + (size_t)cached_len * kv_hidden;
        __nv_bfloat16 *v_cache_ptr = (__nv_bfloat16 *)lkv->v.data + (size_t)cached_len * kv_hidden;
        CUDA_CHECK(cudaMemcpy(k_cache_ptr, d_dec_k_buf,
                              (size_t)seq_len * kv_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(v_cache_ptr, d_dec_v_buf,
                              (size_t)seq_len * kv_hidden * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));

        /* GQA attention (no causal mask for single new token) */
        int total_key = cached_len + seq_len;
        gqa_attention(handle, d_dec_q_buf,
                      (__nv_bfloat16 *)lkv->k.data,
                      (__nv_bfloat16 *)lkv->v.data,
                      d_dec_attn_out,
                      seq_len, total_key,
                      num_heads, num_kv_heads, head_dim,
                      false, cached_len);

        /* O projection */
        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.o_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.o_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.o_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.o_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.o_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_attn_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);

        residual_add(layer_out, layer_out, d_dec_residual, n);

        /* ===== MLP ===== */

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.gate_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.gate_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.gate_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.gate_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.gate_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_gate_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.up_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.up_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.up_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.up_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.up_proj.pre_quant_scale", layer);
        ensure_decoder_up_buf();
        linear_proj(handle, d_dec_up_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        silu_elementwise_mul(d_dec_gate_buf, d_dec_up_buf, seq_len * ffn_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.down_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.down_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.down_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.down_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.down_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_gate_buf, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, ffn_dim, true);

        residual_add(layer_out, layer_out, d_dec_residual, n);
    }

    const Tensor *t_norm = wget(ws, "language_model.model.norm.weight");
    rmsnorm(hidden, hidden, (__nv_bfloat16 *)t_norm->data, 1e-5f, seq_len, hidden_dim);

    kv->seq_len += 1;
    CUDA_CHECK(cudaDeviceSynchronize());
}

void decoder_decode_step_paged(
    cublasHandle_t handle,
    const WeightStore *ws,
    __nv_bfloat16 *hidden,
    KVPool *pool,
    BlockTable *bt,
    int *d_seq_lens,
    int seq_idx,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    int hidden_dim   = cfg->dec_hidden;
    int num_layers   = cfg->dec_layers;
    int num_heads    = cfg->dec_heads;
    int num_kv_heads = cfg->dec_kv_heads;
    int head_dim     = cfg->dec_head_dim;
    int ffn_dim      = cfg->dec_ffn;
    int kv_hidden    = num_kv_heads * head_dim;
    int seq_len      = 1;
    int cached_len   = 0;
    int64_t h_slot_mapping[1];
    int64_t *d_slot_mapping = NULL;
    __nv_bfloat16 *layer_out = hidden;
    DecoderLinearScratchPlan scratch = decoder_make_linear_scratch_plan(dequant_buf, cfg);
    char w[256], ws_n[256], ws2[256], in_s[256], pre[256];

    CUDA_CHECK(cudaMemcpy(&cached_len, d_seq_lens + seq_idx, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&d_slot_mapping, sizeof(int64_t)));
    decoder_note_alloc(sizeof(int64_t));
    build_slot_mapping(bt->h_table, bt->num_blocks, cached_len, 1,
                       pool->block_size, h_slot_mapping, d_slot_mapping);

    for (int layer = 0; layer < num_layers; layer++) {
        __nv_bfloat16 *k_base = NULL;
        __nv_bfloat16 *v_base = NULL;
        __nv_bfloat16 *k_attn_base = NULL;
        __nv_bfloat16 *v_attn_base = NULL;
        int n = seq_len * hidden_dim;

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.input_layernorm.weight", layer);
        const Tensor *tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.q_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.q_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.q_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.q_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.q_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_hidden_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);
        int total_q = seq_len * num_heads * (head_dim / 2);
        rope_transpose_q_kernel<<<(total_q+255)/256, 256>>>(
            d_dec_q_buf, d_dec_hidden_buf, cached_len, seq_len, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.k_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.k_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.k_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.k_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.k_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_k_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);
        rope_decoder(NULL, d_dec_k_buf, cached_len, seq_len, 0, num_kv_heads, head_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.v_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.v_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.v_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.v_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.v_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_v_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, kv_hidden, hidden_dim, true);

        get_paged_layer_cache(pool, layer, &k_base, &v_base);
        k_attn_base = k_base + (size_t)bt->h_table[0] * pool->block_stride;
        v_attn_base = v_base + (size_t)bt->h_table[0] * pool->block_stride;
        reshape_and_cache((uint16_t *)k_base, (uint16_t *)v_base,
                          (const uint16_t *)d_dec_k_buf, (const uint16_t *)d_dec_v_buf,
                          d_slot_mapping, seq_len, num_kv_heads, head_dim, pool->block_size);

        gqa_attention(handle, d_dec_q_buf,
                      k_attn_base,
                      v_attn_base,
                      d_dec_attn_out,
                      seq_len, cached_len + seq_len,
                      num_heads, num_kv_heads, head_dim,
                      false, cached_len);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.o_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.o_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.o_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.o_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.o_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_attn_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, hidden_dim, true);

        residual_add(layer_out, layer_out, d_dec_residual, n);

        CUDA_CHECK(cudaMemcpy(d_dec_residual, layer_out,
                              n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(layer_out, layer_out, (__nv_bfloat16 *)tw->data, 1e-5f, seq_len, hidden_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.gate_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.gate_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.gate_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.gate_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.gate_proj.pre_quant_scale", layer);
        linear_proj(handle, d_dec_gate_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.up_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.up_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.up_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.up_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.up_proj.pre_quant_scale", layer);
        ensure_decoder_up_buf();
        linear_proj(handle, d_dec_up_buf, scratch.weight0, scratch.aux, layer_out, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, ffn_dim, hidden_dim, true);

        silu_elementwise_mul(d_dec_gate_buf, d_dec_up_buf, seq_len * ffn_dim);

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.down_proj.weight",     layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.down_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.down_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.down_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.down_proj.pre_quant_scale", layer);
        linear_proj(handle, layer_out, scratch.weight0, scratch.aux, d_dec_gate_buf, ws,
                 w, ws_n, ws2, in_s, pre, seq_len, hidden_dim, ffn_dim, true);

        residual_add(layer_out, layer_out, d_dec_residual, n);
    }

    const Tensor *t_norm = wget(ws, "language_model.model.norm.weight");
    rmsnorm(hidden, hidden, (__nv_bfloat16 *)t_norm->data, 1e-5f, seq_len, hidden_dim);

    cached_len += 1;
    CUDA_CHECK(cudaMemcpy(d_seq_lens + seq_idx, &cached_len, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(d_slot_mapping));
    decoder_note_free(sizeof(int64_t));
}

void decoder_decode_step_batched(
    cublasHandle_t handle,
    const WeightStore *ws,
    bf16_t **query_ptrs,
    KVPool *pool,
    BlockTable *bts,
    int *d_seq_lens,
    int *h_seq_lens,
    int num_seqs,
    int *seq_indices,
    __nv_bfloat16 *dequant_buf,
    const ModelConfig *cfg)
{
    int hidden_dim   = cfg->dec_hidden;
    int num_layers   = cfg->dec_layers;
    int num_heads    = cfg->dec_heads;
    int num_kv_heads = cfg->dec_kv_heads;
    int head_dim     = cfg->dec_head_dim;
    int ffn_dim      = cfg->dec_ffn;
    int kv_hidden    = num_kv_heads * head_dim;
    size_t hidden_bytes;
    int64_t *h_slot_mapping = NULL;
    int *h_total_seq_lens = NULL;
    __nv_bfloat16 *query_batch_base = NULL;
    int query_batch_contiguous = 1;
    DecoderLinearScratchPlan scratch = decoder_make_linear_scratch_plan(dequant_buf, cfg);
    int profiling_enabled = decoder_profile_enabled();
    cudaEvent_t phase_start = NULL;
    cudaEvent_t phase_stop = NULL;
    uint64_t qkv_ns_total = 0;
    uint64_t cache_ns_total = 0;
    uint64_t attn_ns_total = 0;
    uint64_t o_proj_ns_total = 0;
    uint64_t mlp_ns_total = 0;
    char w[256], ws_n[256], ws2[256], in_s[256], pre[256];

    if (num_seqs <= 0) {
        return;
    }

    CHECK(query_ptrs != NULL, "decoder_decode_step_batched: query_ptrs is NULL");
    CHECK(pool != NULL, "decoder_decode_step_batched: pool is NULL");
    CHECK(bts != NULL, "decoder_decode_step_batched: bts is NULL");
    CHECK(d_seq_lens != NULL, "decoder_decode_step_batched: d_seq_lens is NULL");
    CHECK(h_seq_lens != NULL, "decoder_decode_step_batched: h_seq_lens is NULL");

    hidden_bytes = (size_t)num_seqs * hidden_dim * sizeof(__nv_bfloat16);

    h_slot_mapping = (int64_t *)malloc((size_t)num_seqs * sizeof(int64_t));
    h_total_seq_lens = (int *)malloc((size_t)num_seqs * sizeof(int));
    CHECK(h_slot_mapping != NULL, "decoder_decode_step_batched: failed to allocate host slot mapping");
    CHECK(h_total_seq_lens != NULL, "decoder_decode_step_batched: failed to allocate host seq lens");

    /* Ensure persistent batched decode buffers are large enough */
    ensure_batched_buffers(num_seqs, pool->max_blocks_per_seq,
                           hidden_dim, ffn_dim, kv_hidden);

    if (profiling_enabled) {
        CUDA_CHECK(cudaEventCreate(&phase_start));
        CUDA_CHECK(cudaEventCreate(&phase_stop));
    }

    query_batch_base = (__nv_bfloat16 *)query_ptrs[0];

    for (int seq = 0; seq < num_seqs; seq++) {
        int cached_len = h_seq_lens[seq];
        int logical_block;
        int table_row;

        CHECK(query_ptrs[seq] != NULL, "decoder_decode_step_batched: query_ptrs[%d] is NULL", seq);
        CHECK(bts[seq].h_table != NULL, "decoder_decode_step_batched: bts[%d].h_table is NULL", seq);
        CHECK(cached_len >= 0, "decoder_decode_step_batched: negative seq len for seq %d", seq);

        logical_block = cached_len / pool->block_size;
        CHECK(logical_block < bts[seq].num_blocks,
              "decoder_decode_step_batched: seq %d length %d exceeds %d allocated blocks",
              seq, cached_len, bts[seq].num_blocks);

        h_slot_mapping[seq] = (int64_t)bts[seq].h_table[logical_block] * (int64_t)pool->block_size +
                              (int64_t)(cached_len % pool->block_size);
        h_total_seq_lens[seq] = cached_len + 1;
        if (query_batch_contiguous &&
            ((__nv_bfloat16 *)query_ptrs[seq] != query_batch_base + (size_t)seq * hidden_dim)) {
            query_batch_contiguous = 0;
        }

        table_row = seq_indices ? seq_indices[seq] : bts[seq].seq_idx;
        CHECK(table_row >= 0 && table_row < pool->max_seqs,
              "decoder_decode_step_batched: invalid seq index %d for batch row %d",
              table_row, seq);
    }

    if (query_batch_contiguous) {
        CUDA_CHECK(cudaMemcpy(bd_hidden_batch, query_batch_base,
                              hidden_bytes, cudaMemcpyDeviceToDevice));
    } else {
        for (int seq = 0; seq < num_seqs; seq++) {
            CUDA_CHECK(cudaMemcpy(bd_hidden_batch + (size_t)seq * hidden_dim,
                                  query_ptrs[seq],
                                  (size_t)hidden_dim * sizeof(__nv_bfloat16),
                                  cudaMemcpyDeviceToDevice));
        }
    }

    /* Copy block tables from pool to batched buffer */
    {
        size_t per_seq_bt = (size_t)pool->max_blocks_per_seq * sizeof(int);
        for (int seq = 0; seq < num_seqs; seq++) {
            int table_row = seq_indices ? seq_indices[seq] : bts[seq].seq_idx;
            const int *src_block_table = pool->d_block_tables + (size_t)table_row * pool->max_blocks_per_seq;
            int *dst_block_table = bd_block_tables + (size_t)seq * pool->max_blocks_per_seq;
            CUDA_CHECK(cudaMemcpy(dst_block_table, src_block_table, per_seq_bt,
                                  cudaMemcpyDeviceToDevice));
        }
    }

    CUDA_CHECK(cudaMemcpy(bd_slot_mapping, h_slot_mapping,
                          (size_t)num_seqs * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(bd_total_seq_lens, h_total_seq_lens,
                          (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    for (int layer = 0; layer < num_layers; layer++) {
        __nv_bfloat16 *k_base = NULL;
        __nv_bfloat16 *v_base = NULL;
        const Tensor *tw;

        CUDA_CHECK(cudaMemcpy(bd_residual_batch, bd_hidden_batch,
                              hidden_bytes, cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.input_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(bd_hidden_batch, bd_hidden_batch, (__nv_bfloat16 *)tw->data, 1e-5f, num_seqs, hidden_dim);

        if (profiling_enabled) {
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.q_proj.weight", layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.q_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.q_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.q_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.q_proj.pre_quant_scale", layer);
        linear_proj(handle, bd_batched_query, scratch.weight0, scratch.aux, bd_hidden_batch, ws,
                 w, ws_n, ws2, in_s, pre, num_seqs, hidden_dim, hidden_dim, true);
        {
            int total_q_pairs = num_seqs * num_heads * (head_dim / 2);
            rope_batched_single_token_kernel<<<(total_q_pairs + 255) / 256, 256>>>(
                bd_batched_query, d_seq_lens, num_seqs, num_heads, head_dim);
            CUDA_CHECK(cudaGetLastError());
        }

        {
            char k_w[256], k_ws[256], k_ws2[256], k_in[256], k_pre[256];
            char v_w[256], v_ws[256], v_ws2[256], v_in[256], v_pre[256];
            int fused_kv = 0;

            snprintf(k_w,   sizeof(k_w),   "language_model.model.layers.%d.self_attn.k_proj.weight", layer);
            snprintf(k_ws,  sizeof(k_ws),  "language_model.model.layers.%d.self_attn.k_proj.weight_scale", layer);
            snprintf(k_ws2, sizeof(k_ws2), "language_model.model.layers.%d.self_attn.k_proj.weight_scale_2", layer);
            snprintf(k_in,  sizeof(k_in),  "language_model.model.layers.%d.self_attn.k_proj.input_scale", layer);
            snprintf(k_pre, sizeof(k_pre), "language_model.model.layers.%d.self_attn.k_proj.pre_quant_scale", layer);

            snprintf(v_w,   sizeof(v_w),   "language_model.model.layers.%d.self_attn.v_proj.weight", layer);
            snprintf(v_ws,  sizeof(v_ws),  "language_model.model.layers.%d.self_attn.v_proj.weight_scale", layer);
            snprintf(v_ws2, sizeof(v_ws2), "language_model.model.layers.%d.self_attn.v_proj.weight_scale_2", layer);
            snprintf(v_in,  sizeof(v_in),  "language_model.model.layers.%d.self_attn.v_proj.input_scale", layer);
            snprintf(v_pre, sizeof(v_pre), "language_model.model.layers.%d.self_attn.v_proj.pre_quant_scale", layer);

            fused_kv = linear_proj_pair_same_input(handle, bd_gate_up_packed, scratch.weight0, bd_hidden_batch, ws,
                                                   k_w, k_ws, k_ws2, k_pre,
                                                   v_w, v_ws, v_ws2, v_pre,
                                                   num_seqs, kv_hidden, hidden_dim);
            if (fused_kv) {
                int total_k_pairs = num_seqs * num_kv_heads * (head_dim / 2);
                split_packed_kv_rope_single_token_kernel<<<(total_k_pairs + 255) / 256, 256>>>(
                    bd_batched_k, bd_batched_v, bd_gate_up_packed, d_seq_lens, num_seqs, num_kv_heads, head_dim);
                CUDA_CHECK(cudaGetLastError());
            } else {
                linear_proj(handle, bd_batched_k, scratch.weight0, scratch.aux, bd_hidden_batch, ws,
                         k_w, k_ws, k_ws2, k_in, k_pre, num_seqs, kv_hidden, hidden_dim, true);
                linear_proj(handle, bd_batched_v, scratch.weight0, scratch.aux, bd_hidden_batch, ws,
                         v_w, v_ws, v_ws2, v_in, v_pre, num_seqs, kv_hidden, hidden_dim, true);
                {
                    int total_k_pairs = num_seqs * num_kv_heads * (head_dim / 2);
                    rope_batched_single_token_kernel<<<(total_k_pairs + 255) / 256, 256>>>(
                        bd_batched_k, d_seq_lens, num_seqs, num_kv_heads, head_dim);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        }
        if (profiling_enabled) {
            qkv_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
        }

        if (profiling_enabled) {
            CUDA_CHECK(cudaEventRecord(phase_start));
        }
        get_paged_layer_cache(pool, layer, &k_base, &v_base);
        reshape_and_cache((uint16_t *)k_base, (uint16_t *)v_base,
                          (const uint16_t *)bd_batched_k, (const uint16_t *)bd_batched_v,
                          bd_slot_mapping, num_seqs, num_kv_heads, head_dim, pool->block_size);
        if (profiling_enabled) {
            cache_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        paged_decode_attention((uint16_t *)bd_batched_attn,
                               (const uint16_t *)bd_batched_query,
                               (const uint16_t *)k_base,
                               (const uint16_t *)v_base,
                               bd_block_tables,
                               bd_total_seq_lens,
                               num_seqs, num_heads,
                               num_kv_heads, head_dim,
                               pool->block_size, pool->max_blocks_per_seq);
        if (profiling_enabled) {
            attn_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.self_attn.o_proj.weight", layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.self_attn.o_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.self_attn.o_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.self_attn.o_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.self_attn.o_proj.pre_quant_scale", layer);
        linear_proj(handle, bd_hidden_batch, scratch.weight0, scratch.aux, bd_batched_attn, ws,
                 w, ws_n, ws2, in_s, pre, num_seqs, hidden_dim, hidden_dim, true);
        if (profiling_enabled) {
            o_proj_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
            CUDA_CHECK(cudaEventRecord(phase_start));
        }

        residual_add(bd_hidden_batch, bd_hidden_batch, bd_residual_batch, num_seqs * hidden_dim);

        CUDA_CHECK(cudaMemcpy(bd_residual_batch, bd_hidden_batch,
                              hidden_bytes, cudaMemcpyDeviceToDevice));

        snprintf(w, sizeof(w), "language_model.model.layers.%d.post_attention_layernorm.weight", layer);
        tw = wget(ws, w);
        rmsnorm(bd_hidden_batch, bd_hidden_batch, (__nv_bfloat16 *)tw->data, 1e-5f, num_seqs, hidden_dim);

        {
            char gate_w[256], gate_ws[256], gate_ws2[256], gate_in[256], gate_pre[256];
            char up_w[256], up_ws[256], up_ws2[256], up_in[256], up_pre[256];
            int fused_gate_up = 0;

            snprintf(gate_w,   sizeof(gate_w),   "language_model.model.layers.%d.mlp.gate_proj.weight", layer);
            snprintf(gate_ws,  sizeof(gate_ws),  "language_model.model.layers.%d.mlp.gate_proj.weight_scale", layer);
            snprintf(gate_ws2, sizeof(gate_ws2), "language_model.model.layers.%d.mlp.gate_proj.weight_scale_2", layer);
            snprintf(gate_in,  sizeof(gate_in),  "language_model.model.layers.%d.mlp.gate_proj.input_scale", layer);
            snprintf(gate_pre, sizeof(gate_pre), "language_model.model.layers.%d.mlp.gate_proj.pre_quant_scale", layer);

            snprintf(up_w,   sizeof(up_w),   "language_model.model.layers.%d.mlp.up_proj.weight", layer);
            snprintf(up_ws,  sizeof(up_ws),  "language_model.model.layers.%d.mlp.up_proj.weight_scale", layer);
            snprintf(up_ws2, sizeof(up_ws2), "language_model.model.layers.%d.mlp.up_proj.weight_scale_2", layer);
            snprintf(up_in,  sizeof(up_in),  "language_model.model.layers.%d.mlp.up_proj.input_scale", layer);
            snprintf(up_pre, sizeof(up_pre), "language_model.model.layers.%d.mlp.up_proj.pre_quant_scale", layer);

            fused_gate_up = linear_proj_pair_same_input(handle, bd_gate_up_packed, scratch.weight0, bd_hidden_batch, ws,
                                                     gate_w, gate_ws, gate_ws2, gate_pre,
                                                     up_w, up_ws, up_ws2, up_pre,
                                                     num_seqs, ffn_dim, hidden_dim);
            if (fused_gate_up) {
                silu_mul_split_packed(bd_gate_batch, bd_gate_up_packed, num_seqs, ffn_dim);
            } else {
                linear_proj(handle, bd_gate_batch, scratch.weight0, scratch.aux, bd_hidden_batch, ws,
                         gate_w, gate_ws, gate_ws2, gate_in, gate_pre, num_seqs, ffn_dim, hidden_dim, true);
                linear_proj(handle, bd_up_batch, scratch.weight0, scratch.aux, bd_hidden_batch, ws,
                         up_w, up_ws, up_ws2, up_in, up_pre, num_seqs, ffn_dim, hidden_dim, true);
                silu_elementwise_mul(bd_gate_batch, bd_up_batch, num_seqs * ffn_dim);
            }
        }

        snprintf(w,   sizeof(w),   "language_model.model.layers.%d.mlp.down_proj.weight", layer);
        snprintf(ws_n,sizeof(ws_n),"language_model.model.layers.%d.mlp.down_proj.weight_scale", layer);
        snprintf(ws2, sizeof(ws2), "language_model.model.layers.%d.mlp.down_proj.weight_scale_2", layer);
        snprintf(in_s,sizeof(in_s),"language_model.model.layers.%d.mlp.down_proj.input_scale", layer);
        snprintf(pre, sizeof(pre), "language_model.model.layers.%d.mlp.down_proj.pre_quant_scale", layer);
        linear_proj(handle, bd_hidden_batch, scratch.weight0, scratch.aux, bd_gate_batch, ws,
                 w, ws_n, ws2, in_s, pre, num_seqs, hidden_dim, ffn_dim, true);

        residual_add(bd_hidden_batch, bd_hidden_batch, bd_residual_batch, num_seqs * hidden_dim);
        if (profiling_enabled) {
            mlp_ns_total += decoder_cuda_event_elapsed_ns(phase_start, phase_stop);
        }
    }

    {
        const Tensor *t_norm = wget(ws, "language_model.model.norm.weight");
        rmsnorm(bd_hidden_batch, bd_hidden_batch, (__nv_bfloat16 *)t_norm->data, 1e-5f, num_seqs, hidden_dim);
    }

    for (int seq = 0; seq < num_seqs; seq++) {
        CUDA_CHECK(cudaMemcpy(query_ptrs[seq],
                              bd_hidden_batch + (size_t)seq * hidden_dim,
                              (size_t)hidden_dim * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToDevice));
        h_seq_lens[seq] = h_total_seq_lens[seq];
    }
    CUDA_CHECK(cudaMemcpy(d_seq_lens, h_total_seq_lens,
                          (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice));

    /* No GPU free — buffers are persistent, reused across decode steps */
    free(h_slot_mapping);
    free(h_total_seq_lens);
    if (profiling_enabled) {
        decoder_profile_record_duration(qkv_ns_total,
                                        &decoder_decode_kernel_stats.qkv_ns_total,
                                        &decoder_decode_kernel_stats.qkv_ns_max);
        decoder_profile_record_duration(cache_ns_total,
                                        &decoder_decode_kernel_stats.cache_ns_total,
                                        &decoder_decode_kernel_stats.cache_ns_max);
        decoder_profile_record_duration(attn_ns_total,
                                        &decoder_decode_kernel_stats.attn_ns_total,
                                        &decoder_decode_kernel_stats.attn_ns_max);
        decoder_profile_record_duration(o_proj_ns_total,
                                        &decoder_decode_kernel_stats.o_proj_ns_total,
                                        &decoder_decode_kernel_stats.o_proj_ns_max);
        decoder_profile_record_duration(mlp_ns_total,
                                        &decoder_decode_kernel_stats.mlp_ns_total,
                                        &decoder_decode_kernel_stats.mlp_ns_max);
        CUDA_CHECK(cudaEventDestroy(phase_start));
        CUDA_CHECK(cudaEventDestroy(phase_stop));
    }
}

} /* extern "C" */
