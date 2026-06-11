#ifndef GLMASR_ENGINE_H
#define GLMASR_ENGINE_H

#include <stdint.h>

#include "model.h"
#include "scheduler.h"
#include "tokenizer.h"

#define GLMASR_PROFILE_BATCH_HIST_BINS 257

typedef struct {
    int   seq_id;
    char *text;
    int   n_generated;
} EngineResult;

typedef struct {
    int      enabled;
    int      waiting_peak;
    int      running_peak;
    int      free_blocks_low_water;
    uint64_t admission_calls;
    uint64_t admission_ns_total;
    uint64_t admission_ns_max;
    uint64_t step_calls;
    uint64_t step_ns_total;
    uint64_t step_ns_max;
    uint64_t prefill_batches;
    uint64_t prefill_seqs;
    uint64_t prefill_ns_total;
    uint64_t prefill_ns_max;
    uint64_t prefill_audio_encode_ns_total;
    uint64_t prefill_audio_encode_ns_max;
    uint64_t prefill_prompt_build_ns_total;
    uint64_t prefill_prompt_build_ns_max;
    uint64_t prefill_prepare_ns_total;
    uint64_t prefill_prepare_ns_max;
    uint64_t prefill_block_alloc_ns_total;
    uint64_t prefill_block_alloc_ns_max;
    uint64_t prefill_embed_ns_total;
    uint64_t prefill_embed_ns_max;
    uint64_t prefill_audio_copy_ns_total;
    uint64_t prefill_audio_copy_ns_max;
    uint64_t prefill_kernel_ns_total;
    uint64_t prefill_kernel_ns_max;
    uint64_t prefill_sample_ns_total;
    uint64_t prefill_sample_ns_max;
    uint64_t decode_batches;
    uint64_t decode_tokens;
    uint64_t decode_ns_total;
    uint64_t decode_ns_max;
    uint64_t decode_prepare_ns_total;
    uint64_t decode_prepare_ns_max;
    uint64_t decode_h2d_ns_total;
    uint64_t decode_h2d_ns_max;
    uint64_t decode_kernel_ns_total;
    uint64_t decode_kernel_ns_max;
    uint64_t decode_sample_ns_total;
    uint64_t decode_sample_ns_max;
    uint64_t prefill_batch_hist[GLMASR_PROFILE_BATCH_HIST_BINS];
    uint64_t decode_batch_hist[GLMASR_PROFILE_BATCH_HIST_BINS];
    uint64_t audio_embed_alloc_calls;
    uint64_t audio_embed_free_calls;
    uint64_t audio_embed_alloc_bytes;
    uint64_t audio_embed_free_bytes;
    uint64_t audio_embed_live_bytes;
    uint64_t audio_embed_peak_live_bytes;
} EngineProfile;

typedef struct {
    Model      *model;
    Tokenizer  *tok;
    Tensor      mel_filters;   /* [201, 128] bf16 on GPU, transposed for coalesced access */
    Scheduler   scheduler;
    Sequence   *sequences;
    EngineResult *pending_results;
    int         max_seqs;
    int         next_request_id;
    int        *free_slots;
    int         free_slot_count;
    int         pending_result_count;
    int         sequences_ready;
    int         sequences_done;
    int         runtime_kv_prepared;

    float  *ws_d_pcm;
    bf16_t *ws_d_mel;
    bf16_t *ws_d_conv1;
    bf16_t *ws_d_conv2;
    bf16_t *ws_d_enc_out;
    bf16_t *ws_d_concat;
    bf16_t *ws_d_proj1;
    int    *ws_d_pf_ids;
    bf16_t *ws_d_pf_input;
    uint64_t ws_d_pf_ids_capacity;
    uint64_t ws_d_pf_input_capacity;
    int    *ws_d_prev_ids;
    int    *ws_d_seq_lens;
    bf16_t *ws_d_batch_q;
    bf16_t *ws_d_embed;
    int    *ws_d_single_id;
    int    *ws_d_sr_seq_lens;
    bf16_t *ws_d_batch_enc;
    bf16_t *ws_d_batch_concat;
    bf16_t *ws_d_batch_proj1;
    bf16_t *ws_d_batch_audio;
    uint64_t ws_d_batch_enc_capacity;
    uint64_t ws_d_batch_concat_capacity;
    uint64_t ws_d_batch_proj1_capacity;
    uint64_t ws_d_batch_audio_capacity;
    int    *h_decode_prev_ids;
    int    *h_decode_seq_lens;
    int    *h_decode_seq_indices;
    bf16_t **h_decode_query_ptrs;
    BlockTable *h_decode_batch_bts;
    EngineProfile profile;
} Engine;

int   engine_init(Engine *eng, const char *model_dir);
int   engine_prepare_runtime_kv_pool(Engine *eng);
void  engine_free(Engine *eng);
void  engine_reset_batch(Engine *eng);
int   engine_submit(Engine *eng, const float *pcm, int num_samples);
int   engine_submit_with_prompt_ids(Engine *eng, const float *pcm, int num_samples,
                                    const int *prompt_token_ids, int num_prompt_tokens);
int   engine_step(Engine *eng, EngineResult *results, int *n_results, int max_results);
void  engine_profile_report_run(Engine *eng, const char *scope);
void  engine_profile_reset_run(Engine *eng);

/*
 * Full pipeline: audio PCM float32 → transcribed text.
 * Caller must free() the returned string.
 */
char *engine_transcribe(Engine *eng, const float *pcm, int num_samples);

#endif
