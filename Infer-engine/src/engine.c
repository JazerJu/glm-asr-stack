#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>

#include "/usr/local/cuda-12.8/include/cuda_runtime.h"
#include "/usr/local/cuda-12.8/include/cublas_v2.h"

#include "engine.h"
#include "model.h"
#include "kv_pool.h"
#include "tokenizer.h"
#include "types.h"
#include "cuda_kernels.h"

#define ENGINE_MAX_SCHED_SEQS 256
#define ENGINE_MAX_BATCHED_TOKENS 10000

static int prompt_fixed_token_count(void);
static int prompt_total_token_count(int user_prompt_tokens);
static int feature_frames_from_samples(int num_samples);
static int audio_tokens_from_feature_frames(int frames);

static int engine_profile_enabled_from_env(void) {
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("GLMASR_PROFILE");
        cached = (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }

    return cached;
}

static int engine_int_from_env_bounded(const char *name, int default_value, int min_value, int max_value) {
    const char *env;
    long parsed;
    char *endptr;

    env = getenv(name);
    if (env == NULL || env[0] == '\0') {
        return default_value;
    }

    parsed = strtol(env, &endptr, 10);
    if (endptr == env || *endptr != '\0') {
        return default_value;
    }
    if (parsed < min_value) {
        return min_value;
    }
    if (parsed > max_value) {
        return max_value;
    }
    return (int)parsed;
}

static int engine_max_sched_seqs(void) {
    return engine_int_from_env_bounded("GLMASR_MAX_SEQS",
                                       ENGINE_MAX_SCHED_SEQS,
                                       1,
                                       ENGINE_MAX_SCHED_SEQS);
}

static int engine_max_batched_tokens(void) {
    return engine_int_from_env_bounded("GLMASR_MAX_BATCHED_TOKENS",
                                       ENGINE_MAX_BATCHED_TOKENS,
                                       1,
                                       65536);
}

static uint64_t engine_now_ns(void) {
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

static double engine_ns_to_ms(uint64_t ns) {
    return (double) ns / 1000000.0;
}

static void engine_profile_record_duration(uint64_t elapsed_ns,
                                           uint64_t *total_ns,
                                           uint64_t *max_ns) {
    if (total_ns != NULL) {
        *total_ns += elapsed_ns;
    }
    if (max_ns != NULL && elapsed_ns > *max_ns) {
        *max_ns = elapsed_ns;
    }
}

static void engine_profile_note_hist(uint64_t *hist, int batch_size) {
    int idx;

    if (hist == NULL || batch_size <= 0) {
        return;
    }

    idx = batch_size;
    if (idx >= GLMASR_PROFILE_BATCH_HIST_BINS) {
        idx = GLMASR_PROFILE_BATCH_HIST_BINS - 1;
    }
    hist[idx]++;
}

static uint64_t engine_audio_embed_bytes(const ModelConfig *cfg, int num_audio_tokens) {
    if (cfg == NULL || num_audio_tokens <= 0) {
        return 0;
    }

    return (uint64_t) num_audio_tokens * (uint64_t) cfg->dec_hidden * (uint64_t) sizeof(bf16_t);
}

static void engine_reserve_bf16_buffer(bf16_t **ptr,
                                       uint64_t *capacity,
                                       uint64_t needed,
                                       const char *name) {
    cudaError_t err;

    if (needed == 0 || *capacity >= needed) {
        return;
    }
    if (*ptr != NULL) {
        cudaFree(*ptr);
        *ptr = NULL;
    }
    err = cudaMalloc((void **)ptr, (size_t)needed * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "cudaMalloc %s failed: %s", name, cudaGetErrorString(err));
    *capacity = needed;
}

static void engine_reserve_int_buffer(int **ptr,
                                      uint64_t *capacity,
                                      uint64_t needed,
                                      const char *name) {
    cudaError_t err;

    if (needed == 0 || *capacity >= needed) {
        return;
    }
    if (*ptr != NULL) {
        cudaFree(*ptr);
        *ptr = NULL;
    }
    err = cudaMalloc((void **)ptr, (size_t)needed * sizeof(int));
    CHECK(err == cudaSuccess, "cudaMalloc %s failed: %s", name, cudaGetErrorString(err));
    *capacity = needed;
}

static void engine_reserve_prefill_workspace(Engine *eng, uint64_t tokens) {
    CHECK(eng != NULL, "engine_reserve_prefill_workspace called with NULL engine");
    engine_reserve_int_buffer(&eng->ws_d_pf_ids, &eng->ws_d_pf_ids_capacity,
                              tokens, "ws_d_pf_ids");
    engine_reserve_bf16_buffer(&eng->ws_d_pf_input, &eng->ws_d_pf_input_capacity,
                               tokens * (uint64_t)eng->model->cfg.dec_hidden,
                               "ws_d_pf_input");
}

static void engine_profile_note_audio_alloc(Engine *eng, uint64_t bytes) {
    if (eng == NULL || !eng->profile.enabled || bytes == 0) {
        return;
    }

    eng->profile.audio_embed_alloc_calls++;
    eng->profile.audio_embed_alloc_bytes += bytes;
    eng->profile.audio_embed_live_bytes += bytes;
    if (eng->profile.audio_embed_live_bytes > eng->profile.audio_embed_peak_live_bytes) {
        eng->profile.audio_embed_peak_live_bytes = eng->profile.audio_embed_live_bytes;
    }
}

static void engine_profile_note_audio_free(Engine *eng, uint64_t bytes) {
    if (eng == NULL || !eng->profile.enabled || bytes == 0) {
        return;
    }

    eng->profile.audio_embed_free_calls++;
    eng->profile.audio_embed_free_bytes += bytes;
    if (eng->profile.audio_embed_live_bytes >= bytes) {
        eng->profile.audio_embed_live_bytes -= bytes;
    } else {
        eng->profile.audio_embed_live_bytes = 0;
    }
}

static void engine_profile_note_scheduler(Engine *eng) {
    if (eng == NULL || !eng->profile.enabled) {
        return;
    }

    if (eng->scheduler.waiting_len > eng->profile.waiting_peak) {
        eng->profile.waiting_peak = eng->scheduler.waiting_len;
    }
    if (eng->scheduler.running_len > eng->profile.running_peak) {
        eng->profile.running_peak = eng->scheduler.running_len;
    }
}

static void engine_profile_note_pool(Engine *eng) {
    int free_blocks;

    if (eng == NULL || !eng->profile.enabled || eng->model == NULL || eng->model->kv_pool == NULL) {
        return;
    }

    free_blocks = eng->model->kv_pool->free_top;
    if (eng->profile.free_blocks_low_water < 0 || free_blocks < eng->profile.free_blocks_low_water) {
        eng->profile.free_blocks_low_water = free_blocks;
    }
}

static int engine_profile_has_data(const Engine *eng) {
    if (eng == NULL || !eng->profile.enabled) {
        return 0;
    }

    return eng->profile.admission_calls > 0 ||
           eng->profile.step_calls > 0 ||
           eng->profile.prefill_batches > 0 ||
           eng->profile.decode_batches > 0 ||
           eng->profile.audio_embed_alloc_calls > 0;
}

static void engine_profile_print_hist(const char *label, const uint64_t *hist) {
    int printed = 0;

    fprintf(stderr, "  %s=", label);
    for (int i = 1; i < GLMASR_PROFILE_BATCH_HIST_BINS; i++) {
        if (hist[i] == 0) {
            continue;
        }
        fprintf(stderr, "%s%d:%" PRIu64, printed ? "," : "", i, hist[i]);
        printed = 1;
    }
    if (!printed) {
        fprintf(stderr, "none");
    }
    fputc('\n', stderr);
}

static void engine_profile_reset(Engine *eng) {
    int enabled;
    int free_blocks_low_water = -1;

    if (eng == NULL) {
        return;
    }

    enabled = eng->profile.enabled;
    if (enabled && eng->model != NULL && eng->model->kv_pool != NULL) {
        free_blocks_low_water = eng->model->kv_pool->free_top;
    }

    memset(&eng->profile, 0, sizeof(eng->profile));
    eng->profile.enabled = enabled;
    eng->profile.free_blocks_low_water = free_blocks_low_water;

    if (enabled) {
        encoder_scratch_profile_reset();
        encoder_attention_profile_reset();
        decoder_scratch_profile_reset();
        decoder_decode_kernel_profile_reset();
        decoder_prefill_kernel_profile_reset();
    }
}

static void engine_profile_report(Engine *eng, const char *scope) {
    ScratchProfileStats encoder_stats;
    EncoderAttentionProfileStats encoder_attn_stats;
    ScratchProfileStats decoder_stats;
    DecodeKernelProfileStats decode_kernel_stats;
    PrefillKernelProfileStats prefill_kernel_stats;
    int total_blocks = 0;

    if (!engine_profile_has_data(eng)) {
        return;
    }

    memset(&encoder_stats, 0, sizeof(encoder_stats));
    memset(&encoder_attn_stats, 0, sizeof(encoder_attn_stats));
    memset(&decoder_stats, 0, sizeof(decoder_stats));
    memset(&decode_kernel_stats, 0, sizeof(decode_kernel_stats));
    memset(&prefill_kernel_stats, 0, sizeof(prefill_kernel_stats));
    encoder_scratch_profile_get(&encoder_stats);
    encoder_attention_profile_get(&encoder_attn_stats);
    decoder_scratch_profile_get(&decoder_stats);
    decoder_decode_kernel_profile_get(&decode_kernel_stats);
    decoder_prefill_kernel_profile_get(&prefill_kernel_stats);

    if (eng->model != NULL && eng->model->kv_pool != NULL) {
        total_blocks = eng->model->kv_pool->num_blocks;
    }

    fprintf(stderr, "GLMASR_PROFILE scope=%s\n", scope != NULL ? scope : "run");
    fprintf(stderr,
            "  admission calls=%" PRIu64 " total_ms=%.3f max_ms=%.3f\n",
            eng->profile.admission_calls,
            engine_ns_to_ms(eng->profile.admission_ns_total),
            engine_ns_to_ms(eng->profile.admission_ns_max));
    fprintf(stderr,
            "  engine_step calls=%" PRIu64 " total_ms=%.3f max_ms=%.3f\n",
            eng->profile.step_calls,
            engine_ns_to_ms(eng->profile.step_ns_total),
            engine_ns_to_ms(eng->profile.step_ns_max));
    fprintf(stderr,
            "  prefill batches=%" PRIu64 " seqs=%" PRIu64 " total_ms=%.3f max_ms=%.3f\n",
            eng->profile.prefill_batches,
            eng->profile.prefill_seqs,
            engine_ns_to_ms(eng->profile.prefill_ns_total),
            engine_ns_to_ms(eng->profile.prefill_ns_max));
    fprintf(stderr,
            "  prefill_audio_encode total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_audio_encode_ns_total),
            engine_ns_to_ms(eng->profile.prefill_audio_encode_ns_max));
    fprintf(stderr,
            "  prefill_prompt_build total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_prompt_build_ns_total),
            engine_ns_to_ms(eng->profile.prefill_prompt_build_ns_max));
    fprintf(stderr,
            "  prefill_prepare total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_prepare_ns_total),
            engine_ns_to_ms(eng->profile.prefill_prepare_ns_max));
    fprintf(stderr,
            "  prefill_block_alloc total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_block_alloc_ns_total),
            engine_ns_to_ms(eng->profile.prefill_block_alloc_ns_max));
    fprintf(stderr,
            "  prefill_embed total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_embed_ns_total),
            engine_ns_to_ms(eng->profile.prefill_embed_ns_max));
    fprintf(stderr,
            "  prefill_audio_copy total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_audio_copy_ns_total),
            engine_ns_to_ms(eng->profile.prefill_audio_copy_ns_max));
    fprintf(stderr,
            "  prefill_kernel total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_kernel_ns_total),
            engine_ns_to_ms(eng->profile.prefill_kernel_ns_max));
    fprintf(stderr,
            "  prefill_kernel_qkv total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(prefill_kernel_stats.qkv_ns_total),
            engine_ns_to_ms(prefill_kernel_stats.qkv_ns_max));
    fprintf(stderr,
            "  prefill_kernel_cache total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(prefill_kernel_stats.cache_ns_total),
            engine_ns_to_ms(prefill_kernel_stats.cache_ns_max));
    fprintf(stderr,
            "  prefill_kernel_attn total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(prefill_kernel_stats.attn_ns_total),
            engine_ns_to_ms(prefill_kernel_stats.attn_ns_max));
    fprintf(stderr,
            "  prefill_kernel_o_proj total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(prefill_kernel_stats.o_proj_ns_total),
            engine_ns_to_ms(prefill_kernel_stats.o_proj_ns_max));
    fprintf(stderr,
            "  prefill_kernel_mlp total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(prefill_kernel_stats.mlp_ns_total),
            engine_ns_to_ms(prefill_kernel_stats.mlp_ns_max));
    fprintf(stderr,
            "  prefill_sample total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.prefill_sample_ns_total),
            engine_ns_to_ms(eng->profile.prefill_sample_ns_max));
    fprintf(stderr,
            "  decode batches=%" PRIu64 " tokens=%" PRIu64 " total_ms=%.3f max_ms=%.3f\n",
            eng->profile.decode_batches,
            eng->profile.decode_tokens,
            engine_ns_to_ms(eng->profile.decode_ns_total),
            engine_ns_to_ms(eng->profile.decode_ns_max));
    fprintf(stderr,
            "  decode_prepare total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.decode_prepare_ns_total),
            engine_ns_to_ms(eng->profile.decode_prepare_ns_max));
    fprintf(stderr,
            "  decode_h2d total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.decode_h2d_ns_total),
            engine_ns_to_ms(eng->profile.decode_h2d_ns_max));
    fprintf(stderr,
            "  decode_kernel total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.decode_kernel_ns_total),
            engine_ns_to_ms(eng->profile.decode_kernel_ns_max));
    fprintf(stderr,
            "  decode_kernel_qkv total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(decode_kernel_stats.qkv_ns_total),
            engine_ns_to_ms(decode_kernel_stats.qkv_ns_max));
    fprintf(stderr,
            "  decode_kernel_cache total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(decode_kernel_stats.cache_ns_total),
            engine_ns_to_ms(decode_kernel_stats.cache_ns_max));
    fprintf(stderr,
            "  decode_kernel_attn total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(decode_kernel_stats.attn_ns_total),
            engine_ns_to_ms(decode_kernel_stats.attn_ns_max));
    fprintf(stderr,
            "  decode_kernel_o_proj total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(decode_kernel_stats.o_proj_ns_total),
            engine_ns_to_ms(decode_kernel_stats.o_proj_ns_max));
    fprintf(stderr,
            "  decode_kernel_mlp total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(decode_kernel_stats.mlp_ns_total),
            engine_ns_to_ms(decode_kernel_stats.mlp_ns_max));
    fprintf(stderr,
            "  decode_sample total_ms=%.3f max_ms=%.3f\n",
            engine_ns_to_ms(eng->profile.decode_sample_ns_total),
            engine_ns_to_ms(eng->profile.decode_sample_ns_max));
    fprintf(stderr,
            "  scheduler peaks waiting=%d running=%d free_blocks_low_water=%d/%d\n",
            eng->profile.waiting_peak,
            eng->profile.running_peak,
            eng->profile.free_blocks_low_water,
            total_blocks);
    fprintf(stderr,
            "  scheduler kv preemptions=%" PRIu64
            " self_preemptions=%" PRIu64
            " appended_blocks=%" PRIu64 "\n",
            eng->scheduler.preemptions,
            eng->scheduler.self_preemptions,
            eng->scheduler.appended_blocks);
    engine_profile_print_hist("prefill_batch_hist", eng->profile.prefill_batch_hist);
    engine_profile_print_hist("decode_batch_hist", eng->profile.decode_batch_hist);
    fprintf(stderr,
            "  audio_embed_scratch alloc_calls=%" PRIu64 " free_calls=%" PRIu64
            " alloc_mb=%.3f free_mb=%.3f live_mb=%.3f peak_live_mb=%.3f\n",
            eng->profile.audio_embed_alloc_calls,
            eng->profile.audio_embed_free_calls,
            (double) eng->profile.audio_embed_alloc_bytes / (1024.0 * 1024.0),
            (double) eng->profile.audio_embed_free_bytes / (1024.0 * 1024.0),
            (double) eng->profile.audio_embed_live_bytes / (1024.0 * 1024.0),
            (double) eng->profile.audio_embed_peak_live_bytes / (1024.0 * 1024.0));
    fprintf(stderr,
            "  encoder_scratch alloc_calls=%" PRIu64 " free_calls=%" PRIu64
            " alloc_mb=%.3f free_mb=%.3f live_mb=%.3f peak_mb=%.3f\n",
            encoder_stats.alloc_calls,
            encoder_stats.free_calls,
            (double) encoder_stats.alloc_bytes / (1024.0 * 1024.0),
            (double) encoder_stats.free_bytes / (1024.0 * 1024.0),
            (double) encoder_stats.current_bytes / (1024.0 * 1024.0),
            (double) encoder_stats.peak_bytes / (1024.0 * 1024.0));
    fprintf(stderr,
            "  encoder_attn_paths fa_calls=%" PRIu64
            " simple_calls=%" PRIu64
            " pipeline_calls=%" PRIu64
            " fa_rows=%" PRIu64
            " fa_score_melems=%.3f cublas_calls=%" PRIu64
            " cublas_rows=%" PRIu64 " cublas_score_melems=%.3f"
            " compare_calls=%" PRIu64 "\n",
            encoder_attn_stats.fa_calls,
            encoder_attn_stats.fa_simple_calls,
            encoder_attn_stats.fa_pipeline_calls,
            encoder_attn_stats.fa_rows,
            (double) encoder_attn_stats.fa_score_elems / 1000000.0,
            encoder_attn_stats.cublas_calls,
            encoder_attn_stats.cublas_rows,
            (double) encoder_attn_stats.cublas_score_elems / 1000000.0,
            encoder_attn_stats.compare_calls);
    fprintf(stderr,
            "  decoder_scratch alloc_calls=%" PRIu64 " free_calls=%" PRIu64
            " alloc_mb=%.3f free_mb=%.3f live_mb=%.3f peak_mb=%.3f\n",
            decoder_stats.alloc_calls,
            decoder_stats.free_calls,
            (double) decoder_stats.alloc_bytes / (1024.0 * 1024.0),
            (double) decoder_stats.free_bytes / (1024.0 * 1024.0),
            (double) decoder_stats.current_bytes / (1024.0 * 1024.0),
            (double) decoder_stats.peak_bytes / (1024.0 * 1024.0));
}

static void engine_alloc_workspace(Engine *eng) {
    ModelConfig *cfg = &eng->model->cfg;
    int max_frames = cfg->max_audio_frames;
    int max_samples = max_frames * 160 + 400;
    int max_conv2 = (max_frames + 2 - 3) / 2 + 1;
    int max_concat = max_conv2 / 4;
    int max_prefill_tokens = prompt_fixed_token_count() + audio_tokens_from_feature_frames(max_frames);
    cudaError_t err;

    err = cudaMalloc((void**)&eng->ws_d_pcm, (size_t)max_samples * sizeof(float));
    CHECK(err == cudaSuccess, "ws_d_pcm alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_mel, (size_t)cfg->num_mel_bins * max_frames * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_mel alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_conv1, (size_t)cfg->enc_hidden * max_frames * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_conv1 alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_conv2, (size_t)cfg->enc_hidden * max_conv2 * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_conv2 alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_enc_out, (size_t)max_conv2 * cfg->enc_hidden * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_enc_out alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_concat, (size_t)max_concat * cfg->enc_hidden * 4 * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_concat alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_proj1, (size_t)max_concat * cfg->proj_mid * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_proj1 alloc failed: %s", cudaGetErrorString(err));

    engine_reserve_prefill_workspace(eng, (uint64_t)max_prefill_tokens);

    err = cudaMalloc((void**)&eng->ws_d_prev_ids, (size_t)eng->max_seqs * sizeof(int));
    CHECK(err == cudaSuccess, "ws_d_prev_ids alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_seq_lens, (size_t)eng->max_seqs * sizeof(int));
    CHECK(err == cudaSuccess, "ws_d_seq_lens alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_batch_q, (size_t)eng->max_seqs * cfg->dec_hidden * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_batch_q alloc failed: %s", cudaGetErrorString(err));

    err = cudaMalloc((void**)&eng->ws_d_embed, (size_t)cfg->dec_hidden * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "ws_d_embed alloc failed: %s", cudaGetErrorString(err));
    err = cudaMalloc((void**)&eng->ws_d_single_id, sizeof(int));
    CHECK(err == cudaSuccess, "ws_d_single_id alloc failed: %s", cudaGetErrorString(err));

    err = cudaMalloc((void**)&eng->ws_d_sr_seq_lens, sizeof(int));
    CHECK(err == cudaSuccess, "ws_d_sr_seq_lens alloc failed: %s", cudaGetErrorString(err));

    eng->h_decode_prev_ids = malloc((size_t)eng->max_seqs * sizeof(int));
    CHECK(eng->h_decode_prev_ids != NULL, "h_decode_prev_ids alloc failed");
    eng->h_decode_seq_lens = malloc((size_t)eng->max_seqs * sizeof(int));
    CHECK(eng->h_decode_seq_lens != NULL, "h_decode_seq_lens alloc failed");
    eng->h_decode_seq_indices = malloc((size_t)eng->max_seqs * sizeof(int));
    CHECK(eng->h_decode_seq_indices != NULL, "h_decode_seq_indices alloc failed");
    eng->h_decode_query_ptrs = malloc((size_t)eng->max_seqs * sizeof(bf16_t *));
    CHECK(eng->h_decode_query_ptrs != NULL, "h_decode_query_ptrs alloc failed");
    eng->h_decode_batch_bts = malloc((size_t)eng->max_seqs * sizeof(BlockTable));
    CHECK(eng->h_decode_batch_bts != NULL, "h_decode_batch_bts alloc failed");
}

static void engine_free_workspace(Engine *eng) {
    if (eng->ws_d_pcm) { cudaFree(eng->ws_d_pcm); eng->ws_d_pcm = NULL; }
    if (eng->ws_d_mel) { cudaFree(eng->ws_d_mel); eng->ws_d_mel = NULL; }
    if (eng->ws_d_conv1) { cudaFree(eng->ws_d_conv1); eng->ws_d_conv1 = NULL; }
    if (eng->ws_d_conv2) { cudaFree(eng->ws_d_conv2); eng->ws_d_conv2 = NULL; }
    if (eng->ws_d_enc_out) { cudaFree(eng->ws_d_enc_out); eng->ws_d_enc_out = NULL; }
    if (eng->ws_d_concat) { cudaFree(eng->ws_d_concat); eng->ws_d_concat = NULL; }
    if (eng->ws_d_proj1) { cudaFree(eng->ws_d_proj1); eng->ws_d_proj1 = NULL; }
    if (eng->ws_d_pf_ids) { cudaFree(eng->ws_d_pf_ids); eng->ws_d_pf_ids = NULL; }
    if (eng->ws_d_pf_input) { cudaFree(eng->ws_d_pf_input); eng->ws_d_pf_input = NULL; }
    eng->ws_d_pf_ids_capacity = 0;
    eng->ws_d_pf_input_capacity = 0;
    if (eng->ws_d_prev_ids) { cudaFree(eng->ws_d_prev_ids); eng->ws_d_prev_ids = NULL; }
    if (eng->ws_d_seq_lens) { cudaFree(eng->ws_d_seq_lens); eng->ws_d_seq_lens = NULL; }
    if (eng->ws_d_batch_q) { cudaFree(eng->ws_d_batch_q); eng->ws_d_batch_q = NULL; }
    if (eng->ws_d_embed) { cudaFree(eng->ws_d_embed); eng->ws_d_embed = NULL; }
    if (eng->ws_d_single_id) { cudaFree(eng->ws_d_single_id); eng->ws_d_single_id = NULL; }
    if (eng->ws_d_sr_seq_lens) { cudaFree(eng->ws_d_sr_seq_lens); eng->ws_d_sr_seq_lens = NULL; }
    if (eng->ws_d_batch_enc) { cudaFree(eng->ws_d_batch_enc); eng->ws_d_batch_enc = NULL; }
    if (eng->ws_d_batch_concat) { cudaFree(eng->ws_d_batch_concat); eng->ws_d_batch_concat = NULL; }
    if (eng->ws_d_batch_proj1) { cudaFree(eng->ws_d_batch_proj1); eng->ws_d_batch_proj1 = NULL; }
    if (eng->ws_d_batch_audio) { cudaFree(eng->ws_d_batch_audio); eng->ws_d_batch_audio = NULL; }
    eng->ws_d_batch_enc_capacity = 0;
    eng->ws_d_batch_concat_capacity = 0;
    eng->ws_d_batch_proj1_capacity = 0;
    eng->ws_d_batch_audio_capacity = 0;
    free(eng->h_decode_prev_ids); eng->h_decode_prev_ids = NULL;
    free(eng->h_decode_seq_lens); eng->h_decode_seq_lens = NULL;
    free(eng->h_decode_seq_indices); eng->h_decode_seq_indices = NULL;
    free(eng->h_decode_query_ptrs); eng->h_decode_query_ptrs = NULL;
    free(eng->h_decode_batch_bts); eng->h_decode_batch_bts = NULL;
    lm_head_cleanup();
    mel_cleanup();
    encoder_cleanup();
    decoder_cleanup_all();
}

static const int k_transcribe_prompt_tokens[] = {
    8705, 455, 1700, 8091, 283, 16926, 1636, 261, 8237, 7099, 63
};

static int prompt_fixed_token_count(void) {
    int prompt_len = (int)(sizeof(k_transcribe_prompt_tokens) / sizeof(k_transcribe_prompt_tokens[0]));
    return 8 + prompt_len;
}

static int prompt_total_token_count(int user_prompt_tokens) {
    if (user_prompt_tokens < 0) {
        user_prompt_tokens = 0;
    }
    return prompt_fixed_token_count() + user_prompt_tokens;
}

static int feature_frames_from_samples(int num_samples) {
    if (num_samples <= 0) {
        return 1;
    }
    return (num_samples + 159) / 160;
}

static int conv2_rows_from_feature_frames(int frames) {
    if (frames <= 0) {
        return 1;
    }
    return (frames + 1) / 2;
}

static int audio_tokens_from_feature_frames(int frames) {
    return conv2_rows_from_feature_frames(frames) / 4;
}

static int padded_feature_frames_for_merge(int frames, int max_frames) {
    int padded = frames;

    if (padded < 1) {
        padded = 1;
    }
    if (padded > max_frames) {
        padded = max_frames;
    }
    while (padded < max_frames && (conv2_rows_from_feature_frames(padded) % 4) != 0) {
        padded++;
    }
    return padded;
}

static int is_eos(int token, const ModelConfig *cfg) {
    for (int i = 0; i < 4; i++) {
        if (cfg->eos_ids[i] < 0) break;
        if (token == cfg->eos_ids[i]) return 1;
    }
    return 0;
}

static int build_prompt_input_ids(const ModelConfig *cfg,
                                  int actual_audio_tokens,
                                  const int *user_prompt_ids,
                                  int num_user_prompt_ids,
                                  int **out_input_ids,
                                  int *out_total_seq) {
    int prompt_len = (int)(sizeof(k_transcribe_prompt_tokens) / sizeof(k_transcribe_prompt_tokens[0]));
    int user_len = num_user_prompt_ids > 0 ? num_user_prompt_ids : 0;
    int total_seq = prompt_total_token_count(user_len) + actual_audio_tokens;
    int *h_input_ids = malloc((size_t)total_seq * sizeof(int));
    int pos = 0;

    if (!h_input_ids) {
        return -1;
    }

    h_input_ids[pos++] = 59253;
    h_input_ids[pos++] = 10;
    h_input_ids[pos++] = 59261;
    for (int i = 0; i < actual_audio_tokens; i++) {
        h_input_ids[pos++] = cfg->pad_token_id;
    }
    h_input_ids[pos++] = 59262;
    h_input_ids[pos++] = 59253;
    h_input_ids[pos++] = 10;
    for (int i = 0; i < prompt_len; i++) {
        h_input_ids[pos++] = k_transcribe_prompt_tokens[i];
    }
    for (int i = 0; i < user_len; i++) {
        h_input_ids[pos++] = user_prompt_ids[i];
    }
    h_input_ids[pos++] = 59254;
    h_input_ids[pos++] = 10;

    *out_input_ids = h_input_ids;
    *out_total_seq = total_seq;
    return 0;
}

static int estimate_audio_tokens_from_samples(int num_samples) {
    int frames;
    int conv2;

    if (num_samples <= 0) {
        return 1;
    }

    frames = feature_frames_from_samples(num_samples);
    conv2 = conv2_rows_from_feature_frames(frames);
    return conv2 / 4;
}

static int engine_acquire_slot(Engine *eng) {
    if (eng == NULL || eng->free_slot_count <= 0 || eng->free_slots == NULL) {
        return -1;
    }

    return eng->free_slots[--eng->free_slot_count];
}

static void engine_release_slot(Engine *eng, int slot_idx) {
    if (eng == NULL || eng->free_slots == NULL || slot_idx < 0 || slot_idx >= eng->max_seqs ||
        eng->free_slot_count >= eng->max_seqs) {
        return;
    }

    eng->free_slots[eng->free_slot_count++] = slot_idx;
}

static void engine_queue_result(Engine *eng, int seq_id, char *text, int n_generated) {
    if (eng == NULL || eng->pending_results == NULL || text == NULL ||
        eng->pending_result_count >= eng->max_seqs) {
        free(text);
        return;
    }

    eng->pending_results[eng->pending_result_count].seq_id = seq_id;
    eng->pending_results[eng->pending_result_count].text = text;
    eng->pending_results[eng->pending_result_count].n_generated = n_generated;
    eng->pending_result_count++;
}

static void engine_drain_results(Engine *eng, EngineResult *results, int *n_results, int max_results) {
    int copied;

    if (eng == NULL || results == NULL || n_results == NULL || max_results <= 0) {
        return;
    }

    copied = 0;
    while (copied < max_results && eng->pending_result_count > 0) {
        results[copied] = eng->pending_results[0];
        eng->pending_results[0].seq_id = 0;
        eng->pending_results[0].text = NULL;
        eng->pending_results[0].n_generated = 0;
        copied++;
        eng->pending_result_count--;
        if (eng->pending_result_count > 0) {
            memmove(eng->pending_results, eng->pending_results + 1,
                    (size_t) eng->pending_result_count * sizeof(*eng->pending_results));
        }
        eng->pending_results[eng->pending_result_count].seq_id = 0;
        eng->pending_results[eng->pending_result_count].text = NULL;
        eng->pending_results[eng->pending_result_count].n_generated = 0;
    }

    *n_results = copied;
}

static char *decode_generated_text(Engine *eng, const int *generated_tokens, int n_generated) {
    ModelConfig *cfg = &eng->model->cfg;
    int start_idx = 0;
    int end_idx = n_generated;
    char *text = NULL;

    if (end_idx > 0 && (generated_tokens[0] == 10 || generated_tokens[0] == cfg->pad_token_id)) {
        start_idx = 1;
    }
    while (end_idx > start_idx && is_eos(generated_tokens[end_idx - 1], cfg)) {
        end_idx--;
    }

    if (eng->tok) {
        text = tokenizer_decode(eng->tok, generated_tokens + start_idx, end_idx - start_idx);
    } else {
        text = malloc(32);
        if (text) {
            sprintf(text, "[tokens: %d]", n_generated);
        }
    }

    return text;
}

static void prepare_single_request_kv_pool(Model *m) {
    int block_size;
    int max_audio_tokens;
    int max_total_tokens;
    int needed_blocks;

    if (!m->kv_pool) {
        return;
    }

    block_size = m->kv_pool->block_size > 0 ? m->kv_pool->block_size : KV_DEFAULT_BLOCK_SIZE;
    max_audio_tokens = audio_tokens_from_feature_frames(m->cfg.max_audio_frames);
    max_total_tokens = prompt_fixed_token_count() + max_audio_tokens + m->cfg.max_new_tokens;
    needed_blocks = (max_total_tokens + block_size - 1) / block_size;

    if (needed_blocks < 4) {
        needed_blocks = 4;
    }
    if (m->kv_pool->num_blocks <= needed_blocks) {
        return;
    }

    kv_pool_free(m->kv_pool);
    m->kv_pool->max_seqs = 1;
    m->kv_pool->max_blocks_per_seq = needed_blocks;
    if (kv_pool_init(m->kv_pool, needed_blocks, block_size,
                     m->cfg.dec_layers, m->cfg.dec_kv_heads,
                     m->cfg.dec_head_dim) != 0) {
        fprintf(stderr, "WARNING: Failed to resize kv_pool, falling back to legacy KV path\n");
        free(m->kv_pool);
        m->kv_pool = NULL;
    }
}

static int prepare_scheduler_kv_pool(Model *m, int max_seqs) {
    int block_size;
    int max_audio_tokens;
    int max_total_tokens;
    int needed_blocks_per_seq;
    int total_blocks;

    if (!m->kv_pool) {
        return -1;
    }

    block_size = m->kv_pool->block_size > 0 ? m->kv_pool->block_size : KV_DEFAULT_BLOCK_SIZE;
    max_audio_tokens = audio_tokens_from_feature_frames(m->cfg.max_audio_frames);
    max_total_tokens = prompt_fixed_token_count() + max_audio_tokens + m->cfg.max_new_tokens;
    needed_blocks_per_seq = (max_total_tokens + block_size - 1) / block_size;
    if (needed_blocks_per_seq < 1) {
        needed_blocks_per_seq = 1;
    }

    kv_pool_free(m->kv_pool);
    m->kv_pool->max_seqs = max_seqs;
    m->kv_pool->max_blocks_per_seq = needed_blocks_per_seq;

    total_blocks = kv_pool_warmup(block_size,
                                  m->cfg.dec_layers,
                                  m->cfg.dec_kv_heads,
                                  m->cfg.dec_head_dim);
    if (total_blocks < needed_blocks_per_seq) {
        total_blocks = needed_blocks_per_seq;
    }

    m->kv_pool->max_seqs = max_seqs;
    m->kv_pool->max_blocks_per_seq = needed_blocks_per_seq;
    if (kv_pool_init(m->kv_pool, total_blocks, block_size,
                     m->cfg.dec_layers, m->cfg.dec_kv_heads,
                     m->cfg.dec_head_dim) != 0) {
        fprintf(stderr, "WARNING: Failed to initialize scheduler KV pool\n");
        free(m->kv_pool);
        m->kv_pool = NULL;
        return -1;
    }

    return 0;
}

static int mixed_batch(Engine *eng,
                       ScheduleResult *sr,
                       const Tensor *embed_tokens,
                       const Tensor *lm_head_w);
static int decode_batch(Engine *eng,
                        ScheduleResult *sr,
                        const Tensor *embed_tokens,
                        const Tensor *lm_head_w);

static int init_temp_warmup_kv_pool(Model *m, int max_seqs, int max_batched_tokens,
                                    int *out_warmup_seqs) {
    int block_size;
    int max_audio_tokens;
    int max_prompt_tokens;
    int blocks_per_seq;
    int warmup_seqs;
    int total_blocks;

    if (m == NULL || m->kv_pool == NULL || out_warmup_seqs == NULL) {
        return -1;
    }

    block_size = m->kv_pool->block_size > 0 ? m->kv_pool->block_size : KV_DEFAULT_BLOCK_SIZE;
    max_audio_tokens = audio_tokens_from_feature_frames(m->cfg.max_audio_frames);
    max_prompt_tokens = prompt_fixed_token_count() + max_audio_tokens;
    blocks_per_seq = (max_prompt_tokens + block_size - 1) / block_size;
    if (blocks_per_seq < 1) {
        blocks_per_seq = 1;
    }

    warmup_seqs = max_seqs;
    if (max_batched_tokens > 0) {
        int token_capped_seqs = max_batched_tokens / max_prompt_tokens;
        if (token_capped_seqs < 1) {
            token_capped_seqs = 1;
        }
        if (token_capped_seqs < warmup_seqs) {
            warmup_seqs = token_capped_seqs;
        }
    }

    total_blocks = warmup_seqs * blocks_per_seq;
    if (total_blocks < blocks_per_seq) {
        total_blocks = blocks_per_seq;
    }

    kv_pool_free(m->kv_pool);
    m->kv_pool->max_seqs = max_seqs;
    m->kv_pool->max_blocks_per_seq = blocks_per_seq;
    while (total_blocks >= blocks_per_seq) {
        if (kv_pool_init(m->kv_pool, total_blocks, block_size,
                         m->cfg.dec_layers, m->cfg.dec_kv_heads,
                         m->cfg.dec_head_dim) == 0) {
            *out_warmup_seqs = total_blocks / blocks_per_seq;
            fprintf(stderr,
                    "Runtime warmup KV: temp_blocks=%d warmup_seqs=%d blocks_per_seq=%d\n",
                    total_blocks, *out_warmup_seqs, blocks_per_seq);
            return 0;
        }
        total_blocks /= 2;
    }

    return -1;
}

static void reserve_runtime_warmup_workspace(Engine *eng, int warmup_seqs) {
    ModelConfig *cfg;
    int max_audio_tokens;
    int enc_tokens_per_seq;
    int audio_compute_tokens_per_seq;
    uint64_t total_encoder_tokens;
    uint64_t total_audio_tokens;
    uint64_t total_prefill_tokens;

    CHECK(eng != NULL && eng->model != NULL, "reserve_runtime_warmup_workspace called before engine init");
    cfg = &eng->model->cfg;
    max_audio_tokens = audio_tokens_from_feature_frames(cfg->max_audio_frames);
    enc_tokens_per_seq = max_audio_tokens * 4 + 4;
    audio_compute_tokens_per_seq = enc_tokens_per_seq / 4;
    if (warmup_seqs < 1) {
        warmup_seqs = 1;
    }
    total_encoder_tokens = (uint64_t)warmup_seqs * (uint64_t)enc_tokens_per_seq;
    total_audio_tokens = (uint64_t)warmup_seqs * (uint64_t)audio_compute_tokens_per_seq;
    total_prefill_tokens = (uint64_t)warmup_seqs *
                           (uint64_t)(prompt_fixed_token_count() + max_audio_tokens);

    engine_reserve_bf16_buffer(&eng->ws_d_batch_enc, &eng->ws_d_batch_enc_capacity,
                               total_encoder_tokens * (uint64_t)cfg->enc_hidden,
                               "ws_d_batch_enc");
    engine_reserve_bf16_buffer(&eng->ws_d_batch_concat, &eng->ws_d_batch_concat_capacity,
                               total_audio_tokens * (uint64_t)cfg->enc_hidden * 4ULL,
                               "ws_d_batch_concat");
    engine_reserve_bf16_buffer(&eng->ws_d_batch_proj1, &eng->ws_d_batch_proj1_capacity,
                               total_audio_tokens * (uint64_t)cfg->proj_mid,
                               "ws_d_batch_proj1");
    engine_reserve_bf16_buffer(&eng->ws_d_batch_audio, &eng->ws_d_batch_audio_capacity,
                               total_audio_tokens * (uint64_t)cfg->dec_hidden,
                               "ws_d_batch_audio");
    engine_reserve_prefill_workspace(eng, total_prefill_tokens);
}

static int run_runtime_memory_warmup(Engine *eng, int max_batched_tokens) {
    ModelConfig *cfg;
    float *dummy_pcm = NULL;
    Tensor *embed_tokens;
    Tensor *lm_head_w;
    int warmup_seqs = 0;
    int max_samples;
    int submitted = 0;
    int steps = 0;
    int decode_warmup = 0;

    if (eng == NULL || eng->model == NULL || eng->model->kv_pool == NULL ||
        eng->mel_filters.data == NULL) {
        return -1;
    }

    cfg = &eng->model->cfg;
    embed_tokens = ws_get(&eng->model->ws, "language_model.model.embed_tokens.weight");
    lm_head_w = ws_get(&eng->model->ws, "language_model.lm_head.weight");
    CHECK(embed_tokens != NULL, "embed_tokens weight not found during warmup");
    CHECK(lm_head_w != NULL, "lm_head weight not found during warmup");

    if (init_temp_warmup_kv_pool(eng->model, eng->max_seqs, max_batched_tokens,
                                 &warmup_seqs) != 0) {
        return -1;
    }
    reserve_runtime_warmup_workspace(eng, warmup_seqs);

    scheduler_free(&eng->scheduler);
    scheduler_init(&eng->scheduler, eng->model->kv_pool,
                   eng->max_seqs, max_batched_tokens);

    max_samples = cfg->max_audio_frames * 160;
    dummy_pcm = (float *) calloc((size_t) max_samples, sizeof(*dummy_pcm));
    if (dummy_pcm == NULL) {
        return -1;
    }
    for (int i = 0; i < max_samples; ++i) {
        dummy_pcm[i] = ((i / 80) & 1) ? 0.05f : -0.05f;
    }

    for (int i = 0; i < warmup_seqs; ++i) {
        if (engine_submit(eng, dummy_pcm, max_samples) < 0) {
            break;
        }
        submitted++;
    }
    free(dummy_pcm);
    dummy_pcm = NULL;

    if (submitted <= 0) {
        return -1;
    }

    while (eng->scheduler.waiting_len > 0 && steps < submitted + 8) {
        ScheduleResult *sr = scheduler_schedule(&eng->scheduler);
        if (sr == NULL || sr->num_seqs <= 0) {
            break;
        }
        mixed_batch(eng, sr, embed_tokens, lm_head_w);
        scheduler_postprocess(&eng->scheduler, sr);
        steps++;
    }

    if (eng->scheduler.running_len > 0) {
        ScheduleResult decode_sr;
        int decode_count = eng->scheduler.running_len;
        if (decode_count > eng->max_seqs) {
            decode_count = eng->max_seqs;
        }
        decode_sr.seqs = (Sequence **) malloc((size_t) decode_count * sizeof(*decode_sr.seqs));
        decode_sr.seq_input_lens = (int *) malloc((size_t) decode_count * sizeof(*decode_sr.seq_input_lens));
        decode_sr.seq_cached_lens = (int *) malloc((size_t) decode_count * sizeof(*decode_sr.seq_cached_lens));
        CHECK(decode_sr.seqs != NULL && decode_sr.seq_input_lens != NULL && decode_sr.seq_cached_lens != NULL,
              "Failed to allocate runtime decode warmup schedule");
        decode_sr.num_seqs = decode_count;
        decode_sr.type = STEP_DECODE;
        for (int i = 0; i < decode_count; ++i) {
            Sequence *seq = eng->scheduler.running[i];
            decode_sr.seqs[i] = seq;
            decode_sr.seq_input_lens[i] = 1;
            decode_sr.seq_cached_lens[i] = seq != NULL ? seq->total_seq_len : 0;
        }
        decode_warmup = 1;
        decode_batch(eng, &decode_sr, embed_tokens, lm_head_w);
        free(decode_sr.seq_cached_lens);
        free(decode_sr.seq_input_lens);
        free(decode_sr.seqs);
    }

    cudaDeviceSynchronize();
    fprintf(stderr,
            "Runtime warmup complete: submitted=%d prefill_steps=%d decode_warmup=%d running=%d waiting=%d\n",
            submitted, steps, decode_warmup, eng->scheduler.running_len, eng->scheduler.waiting_len);
    return 0;
}

int engine_prepare_runtime_kv_pool(Engine *eng) {
    int max_batched_tokens;

    if (eng == NULL || eng->model == NULL || eng->model->kv_pool == NULL) {
        return -1;
    }
    if (eng->runtime_kv_prepared) {
        return 0;
    }
    if (eng->mel_filters.data == NULL) {
        fprintf(stderr, "ERROR: runtime KV warmup requires mel filters to be loaded first\n");
        return -1;
    }

    max_batched_tokens = engine_max_batched_tokens();

    /* Allow the temporary warmup requests to use the normal engine path. */
    eng->runtime_kv_prepared = 1;
    if (run_runtime_memory_warmup(eng, max_batched_tokens) != 0) {
        eng->runtime_kv_prepared = 0;
        return -1;
    }

    engine_reset_batch(eng);
    scheduler_free(&eng->scheduler);
    if (eng->model->kv_pool) {
        kv_pool_free(eng->model->kv_pool);
    }

    if (prepare_scheduler_kv_pool(eng->model, eng->max_seqs) != 0) {
        eng->runtime_kv_prepared = 0;
        return -1;
    }
    scheduler_init(&eng->scheduler, eng->model->kv_pool,
                   eng->max_seqs, max_batched_tokens);
    eng->next_request_id = 0;
    eng->runtime_kv_prepared = 1;
    fprintf(stderr,
            "Runtime KV pool prepared after real prefill+decode warmup\n");
    return 0;
}

static bf16_t* encode_audio(Engine *eng, const float *pcm, int num_samples, int *out_n_audio_tokens) {
    Model *m = eng->model;
    cublasHandle_t handle = (cublasHandle_t)m->cublas;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    bf16_t *proj2 = NULL;
    int n_frames = 0;
    int conv1_out_len;
    int conv2_out_len;
    int enc_seq_len;
    int n_concat;
    int actual_audio_tokens;
    int pcm_samples;
    int feature_frames;
    int padded_feature_frames;
    int padded_pcm_samples;
    int max_pcm_samples;
    cudaError_t err;

    max_pcm_samples = cfg->max_audio_frames * 160;
    pcm_samples = num_samples;
    if (pcm_samples > max_pcm_samples) {
        pcm_samples = max_pcm_samples;
    }

    feature_frames = feature_frames_from_samples(pcm_samples);
    if (feature_frames > cfg->max_audio_frames) {
        feature_frames = cfg->max_audio_frames;
    }
    padded_feature_frames = padded_feature_frames_for_merge(feature_frames, cfg->max_audio_frames);
    padded_pcm_samples = padded_feature_frames * 160;

    err = cudaMemcpy(eng->ws_d_pcm, pcm, (size_t)pcm_samples * sizeof(float), cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess, "cudaMemcpy ws_d_pcm failed: %s", cudaGetErrorString(err));
    if (padded_pcm_samples > pcm_samples) {
        err = cudaMemset(eng->ws_d_pcm + pcm_samples, 0,
                         (size_t)(padded_pcm_samples - pcm_samples) * sizeof(float));
        CHECK(err == cudaSuccess, "cudaMemset ws_d_pcm tail failed: %s", cudaGetErrorString(err));
    }

    mel_spectrogram(eng->ws_d_pcm, padded_pcm_samples,
                    (const float *)eng->mel_filters.data,
                    (void*)eng->ws_d_mel, &n_frames);

    Tensor *conv1_w = ws_get(ws, "audio_tower.conv1.weight");
    Tensor *conv1_b = ws_get(ws, "audio_tower.conv1.bias");
    Tensor *conv2_w = ws_get(ws, "audio_tower.conv2.weight");
    Tensor *conv2_b = ws_get(ws, "audio_tower.conv2.bias");

    conv1_out_len = n_frames;
    conv1d_forward((void*)eng->ws_d_conv1, (void*)eng->ws_d_mel,
                   conv1_w->data, conv1_b->data,
                   cfg->num_mel_bins, cfg->enc_hidden, n_frames, 3, 1, 1);
    gelu_inplace((void*)eng->ws_d_conv1, cfg->enc_hidden * conv1_out_len);

    conv2_out_len = (conv1_out_len + 2 * 1 - 3) / 2 + 1;
    conv1d_forward((void*)eng->ws_d_conv2, (void*)eng->ws_d_conv1,
                   conv2_w->data, conv2_b->data,
                   cfg->enc_hidden, cfg->enc_hidden, conv1_out_len, 3, 2, 1);
    gelu_inplace((void*)eng->ws_d_conv2, cfg->enc_hidden * conv2_out_len);

    enc_seq_len = conv2_out_len;
    transpose_2d((void*)eng->ws_d_enc_out, (void*)eng->ws_d_conv2, cfg->enc_hidden, enc_seq_len);
    encoder_forward(handle, ws, (void*)eng->ws_d_enc_out, enc_seq_len, cfg);

    n_frames = enc_seq_len;
    n_concat = n_frames / 4;
    concat_4frames(eng->ws_d_concat, eng->ws_d_enc_out, n_frames, cfg->enc_hidden);

    Tensor *w1 = ws_get(ws, "multi_modal_projector.linear_1.weight");
    Tensor *b1 = ws_get(ws, "multi_modal_projector.linear_1.bias");
    bf16_linear_gelu(handle, eng->ws_d_proj1, eng->ws_d_concat,
                     (const bf16_t *)w1->data, (const bf16_t *)b1->data,
                     n_concat, cfg->proj_mid, cfg->enc_hidden * 4);

    err = cudaMalloc((void**)&proj2, (size_t)n_concat * cfg->dec_hidden * sizeof(bf16_t));
    CHECK(err == cudaSuccess, "cudaMalloc proj2 failed: %s", cudaGetErrorString(err));
    engine_profile_note_audio_alloc(eng, engine_audio_embed_bytes(cfg, n_concat));

    Tensor *w2 = ws_get(ws, "multi_modal_projector.linear_2.weight");
    Tensor *b2 = ws_get(ws, "multi_modal_projector.linear_2.bias");
    bf16_linear(handle, proj2, eng->ws_d_proj1,
                (const bf16_t *)w2->data, (const bf16_t *)b2->data,
                n_concat, cfg->dec_hidden, cfg->proj_mid);

    actual_audio_tokens = audio_tokens_from_feature_frames(feature_frames);
    if (actual_audio_tokens > n_concat) {
        actual_audio_tokens = n_concat;
    }

    *out_n_audio_tokens = actual_audio_tokens;
    return proj2;
}

static int audio_frontend_to_encoder_rows_into(Engine *eng,
                                               const float *pcm,
                                               int num_samples,
                                               bf16_t *dst_rows,
                                               int *out_enc_seq_len,
                                               int *out_audio_tokens) {
    Model *m = eng->model;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    int n_frames = 0;
    int conv1_out_len;
    int conv2_out_len;
    int enc_seq_len;
    int pcm_samples;
    int feature_frames;
    int padded_feature_frames;
    int padded_pcm_samples;
    int max_pcm_samples;
    cudaError_t err;

    max_pcm_samples = cfg->max_audio_frames * 160;
    pcm_samples = num_samples;
    if (pcm_samples > max_pcm_samples) {
        pcm_samples = max_pcm_samples;
    }

    feature_frames = feature_frames_from_samples(pcm_samples);
    if (feature_frames > cfg->max_audio_frames) {
        feature_frames = cfg->max_audio_frames;
    }
    padded_feature_frames = padded_feature_frames_for_merge(feature_frames, cfg->max_audio_frames);
    padded_pcm_samples = padded_feature_frames * 160;

    err = cudaMemcpy(eng->ws_d_pcm, pcm, (size_t)pcm_samples * sizeof(float), cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess, "cudaMemcpy ws_d_pcm failed: %s", cudaGetErrorString(err));
    if (padded_pcm_samples > pcm_samples) {
        err = cudaMemset(eng->ws_d_pcm + pcm_samples, 0,
                         (size_t)(padded_pcm_samples - pcm_samples) * sizeof(float));
        CHECK(err == cudaSuccess, "cudaMemset ws_d_pcm tail failed: %s", cudaGetErrorString(err));
    }

    mel_spectrogram(eng->ws_d_pcm, padded_pcm_samples,
                    (const float *)eng->mel_filters.data,
                    (void*)eng->ws_d_mel, &n_frames);

    Tensor *conv1_w = ws_get(ws, "audio_tower.conv1.weight");
    Tensor *conv1_b = ws_get(ws, "audio_tower.conv1.bias");
    Tensor *conv2_w = ws_get(ws, "audio_tower.conv2.weight");
    Tensor *conv2_b = ws_get(ws, "audio_tower.conv2.bias");

    conv1_out_len = n_frames;
    conv1d_forward((void*)eng->ws_d_conv1, (void*)eng->ws_d_mel,
                   conv1_w->data, conv1_b->data,
                   cfg->num_mel_bins, cfg->enc_hidden, n_frames, 3, 1, 1);
    gelu_inplace((void*)eng->ws_d_conv1, cfg->enc_hidden * conv1_out_len);

    conv2_out_len = (conv1_out_len + 2 * 1 - 3) / 2 + 1;
    conv1d_forward((void*)eng->ws_d_conv2, (void*)eng->ws_d_conv1,
                   conv2_w->data, conv2_b->data,
                   cfg->enc_hidden, cfg->enc_hidden, conv1_out_len, 3, 2, 1);
    gelu_inplace((void*)eng->ws_d_conv2, cfg->enc_hidden * conv2_out_len);

    enc_seq_len = conv2_out_len;
    transpose_2d((void*)eng->ws_d_enc_out, (void*)eng->ws_d_conv2, cfg->enc_hidden, enc_seq_len);
    if (enc_seq_len > 0 && dst_rows != NULL) {
        err = cudaMemcpy(dst_rows, eng->ws_d_enc_out,
                         (size_t)enc_seq_len * cfg->enc_hidden * sizeof(bf16_t),
                         cudaMemcpyDeviceToDevice);
        CHECK(err == cudaSuccess, "encoder row batch copy failed: %s", cudaGetErrorString(err));
    }

    *out_enc_seq_len = enc_seq_len;
    *out_audio_tokens = audio_tokens_from_feature_frames(feature_frames);
    if (*out_audio_tokens > enc_seq_len / 4) {
        *out_audio_tokens = enc_seq_len / 4;
    }
    return 0;
}

static int materialize_sequence_prompt(Engine *eng, Sequence *seq) {
    ModelConfig *cfg;
    bf16_t *audio_embeds;
    int actual_audio_tokens;
    int *prompt_ids;
    int total_seq;
    uint64_t start_ns;
    uint64_t elapsed_ns;

    if (eng == NULL || seq == NULL || seq->prompt_token_ids != NULL) {
        return 0;
    }

    cfg = &eng->model->cfg;
    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    audio_embeds = encode_audio(eng, seq->pcm, seq->num_samples, &actual_audio_tokens);
    if (audio_embeds == NULL) {
        return -1;
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_audio_encode_ns_total,
                                       &eng->profile.prefill_audio_encode_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    if (build_prompt_input_ids(cfg, actual_audio_tokens,
                               seq->user_prompt_token_ids,
                               seq->num_user_prompt_tokens,
                               &prompt_ids, &total_seq) != 0) {
        engine_profile_note_audio_free(eng, engine_audio_embed_bytes(cfg, actual_audio_tokens));
        cudaFree(audio_embeds);
        return -1;
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_prompt_build_ns_total,
                                       &eng->profile.prefill_prompt_build_ns_max);
    }

    seq_set_prompt(seq, prompt_ids, total_seq, actual_audio_tokens, audio_embeds);
    seq->estimated_audio_tokens = actual_audio_tokens;
    seq->estimated_prompt_tokens = total_seq;
    seq->estimated_total_tokens = total_seq + seq->max_new_tokens;
    seq->estimated_total_blocks = (total_seq + eng->model->kv_pool->block_size - 1) /
                                  eng->model->kv_pool->block_size;
    free(prompt_ids);
    return 0;
}

static int engine_append_block(Engine *eng, Sequence *seq) {
    int block_id;

    if (eng == NULL || eng->model == NULL || eng->model->kv_pool == NULL ||
        seq == NULL || seq->bt.h_table == NULL || seq->bt.num_blocks >= seq->bt.max_blocks) {
        return -1;
    }

    block_id = kv_pool_alloc_block(eng->model->kv_pool);
    if (block_id < 0) {
        return -1;
    }

    seq->bt.h_table[seq->bt.num_blocks++] = block_id;
    block_table_upload(&seq->bt);
    return 0;
}

static int prefill_sequence(Engine *eng,
                            Sequence *seq,
                            const Tensor *embed_tokens,
                            const Tensor *lm_head_w) {
    Model *m = eng->model;
    cublasHandle_t handle = (cublasHandle_t)m->cublas;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    int total_seq;
    int needed_blocks;
    int first_token = 0;
    int prefix_tokens;
    int suffix_tokens;
    uint64_t start_ns;
    uint64_t elapsed_ns;
    uint64_t block_alloc_ns = 0;
    uint64_t embed_ns = 0;
    uint64_t audio_copy_ns = 0;

    decoder_cleanup();

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    if (materialize_sequence_prompt(eng, seq) != 0) {
        return -1;
    }

    total_seq = seq->num_prompt_tokens;
    needed_blocks = (total_seq + m->kv_pool->block_size - 1) / m->kv_pool->block_size;
    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    while (seq->bt.num_blocks < needed_blocks) {
        if (engine_append_block(eng, seq) != 0) {
            return -1;
        }
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        block_alloc_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_block_alloc_ns_total,
                                       &eng->profile.prefill_block_alloc_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    engine_reserve_prefill_workspace(eng, (uint64_t)total_seq);
    cudaMemcpy(eng->ws_d_pf_ids, seq->prompt_token_ids,
               (size_t)total_seq * sizeof(int), cudaMemcpyHostToDevice);
    prefix_tokens = 3;
    suffix_tokens = total_seq - prefix_tokens - seq->num_audio_tokens;
    if (prefix_tokens > 0) {
        embedding_lookup(eng->ws_d_pf_input,
                         (const bf16_t *)embed_tokens->data,
                         eng->ws_d_pf_ids,
                         prefix_tokens,
                         cfg->dec_hidden);
    }
    if (suffix_tokens > 0) {
        embedding_lookup(eng->ws_d_pf_input + (size_t)(prefix_tokens + seq->num_audio_tokens) * cfg->dec_hidden,
                         (const bf16_t *)embed_tokens->data,
                         eng->ws_d_pf_ids + prefix_tokens + seq->num_audio_tokens,
                         suffix_tokens,
                         cfg->dec_hidden);
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        embed_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_embed_ns_total,
                                       &eng->profile.prefill_embed_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    if (seq->num_audio_tokens > 0) {
        cudaError_t err;
        err = cudaMemcpy(eng->ws_d_pf_input + (size_t)3 * cfg->dec_hidden,
                         seq->audio_embeds,
                         (size_t)seq->num_audio_tokens * (size_t)cfg->dec_hidden * sizeof(bf16_t),
                         cudaMemcpyDeviceToDevice);
        CHECK(err == cudaSuccess, "prefill audio embed copy failed: %s", cudaGetErrorString(err));
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        audio_copy_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_audio_copy_ns_total,
                                       &eng->profile.prefill_audio_copy_ns_max);
    }
    if (eng->profile.enabled) {
        elapsed_ns = block_alloc_ns + embed_ns + audio_copy_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_prepare_ns_total,
                                       &eng->profile.prefill_prepare_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    decoder_prefill_paged(handle, ws, eng->ws_d_pf_input, total_seq,
                          m->kv_pool, &seq->bt,
                          (bf16_t *)m->dequant_buf.data, cfg);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_kernel_ns_total,
                                       &eng->profile.prefill_kernel_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    lm_head_argmax(eng->ws_d_pf_input + (size_t)(total_seq - 1) * cfg->dec_hidden,
                   (const bf16_t *)lm_head_w->data,
                   cfg->dec_hidden, cfg->vocab_size, &first_token);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_sample_ns_total,
                                       &eng->profile.prefill_sample_ns_max);
    }

    seq->total_seq_len = total_seq;
    seq->generated_ids[0] = first_token;
    seq->n_generated = 1;
    if (is_eos(first_token, cfg)) {
        seq->eos_reason = 1;
    } else if (seq->n_generated >= seq->max_new_tokens) {
        seq->eos_reason = 2;
    }

    if (seq->audio_embeds) {
        cudaFree(seq->audio_embeds);
        seq->audio_embeds = NULL;
    }
    free(seq->prompt_token_ids);
    seq->prompt_token_ids = NULL;

    return 0;
}

static int prefill_batch(Engine *eng,
                         ScheduleResult *sr,
                         const Tensor *embed_tokens,
                         const Tensor *lm_head_w) {
    Model *m = eng->model;
    cublasHandle_t handle = (cublasHandle_t)m->cublas;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    int *h_prefill_ids = NULL;
    int *seq_lens = eng->h_decode_seq_lens;
    int *seq_offsets = eng->h_decode_seq_indices;
    int *sampled_tokens = eng->h_decode_prev_ids;
    BlockTable *batch_bts = eng->h_decode_batch_bts;
    Sequence **prefill_seqs = NULL;
    int *audio_tokens = NULL;
    int *audio_compute_tokens = NULL;
    int *audio_offsets = NULL;
    int *replay_counts = NULL;
    int *enc_seq_lens = NULL;
    int *enc_seq_offsets = NULL;
    int num_prefill = 0;
    int total_tokens = 0;
    int max_total_audio_tokens = 0;
    int total_audio_tokens = 0;
    int max_total_encoder_tokens = 0;
    int total_encoder_tokens = 0;
    bf16_t *d_batch_enc = NULL;
    bf16_t *d_batch_concat = NULL;
    bf16_t *d_batch_proj1 = NULL;
    bf16_t *d_batch_audio = NULL;
    uint64_t start_ns;
    uint64_t elapsed_ns;
    uint64_t audio_encode_ns = 0;
    uint64_t prompt_build_ns = 0;
    uint64_t block_alloc_ns = 0;
    uint64_t embed_ns = 0;
    uint64_t audio_copy_ns = 0;
    cudaError_t err;

    CHECK(seq_lens && seq_offsets && sampled_tokens && batch_bts,
          "Prefill batch metadata buffers are not initialized");

    prefill_seqs = (Sequence **) malloc((size_t) sr->num_seqs * sizeof(*prefill_seqs));
    CHECK(prefill_seqs != NULL, "Failed to allocate prefill sequence scratch");
    audio_tokens = (int *) calloc((size_t) sr->num_seqs, sizeof(*audio_tokens));
    CHECK(audio_tokens != NULL, "Failed to allocate batched audio token scratch");
    audio_compute_tokens = (int *) calloc((size_t) sr->num_seqs, sizeof(*audio_compute_tokens));
    CHECK(audio_compute_tokens != NULL, "Failed to allocate batched audio compute token scratch");
    audio_offsets = (int *) calloc((size_t) sr->num_seqs, sizeof(*audio_offsets));
    CHECK(audio_offsets != NULL, "Failed to allocate batched audio offset scratch");
    replay_counts = (int *) calloc((size_t) sr->num_seqs, sizeof(*replay_counts));
    CHECK(replay_counts != NULL, "Failed to allocate batched replay count scratch");
    enc_seq_lens = (int *) calloc((size_t) sr->num_seqs, sizeof(*enc_seq_lens));
    CHECK(enc_seq_lens != NULL, "Failed to allocate batched encoder seq len scratch");
    enc_seq_offsets = (int *) calloc((size_t) sr->num_seqs, sizeof(*enc_seq_offsets));
    CHECK(enc_seq_offsets != NULL, "Failed to allocate batched encoder seq offset scratch");

    decoder_cleanup();

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < sr->num_seqs; i++) {
        Sequence *seq = sr->seqs[i];
        if (sr->seq_input_lens[i] <= 1) {
            continue;
        }

        prefill_seqs[num_prefill] = seq;
        max_total_audio_tokens += seq->estimated_audio_tokens + 1;
        max_total_encoder_tokens += seq->estimated_audio_tokens * 4 + 4;
        num_prefill++;
    }

    if (max_total_encoder_tokens > 0) {
        uint64_t need = (uint64_t)max_total_encoder_tokens * (uint64_t)cfg->enc_hidden;
        engine_reserve_bf16_buffer(&eng->ws_d_batch_enc, &eng->ws_d_batch_enc_capacity,
                                   need, "ws_d_batch_enc");
        d_batch_enc = eng->ws_d_batch_enc;
    }
    if (max_total_audio_tokens > 0) {
        uint64_t concat_need = (uint64_t)max_total_audio_tokens * (uint64_t)cfg->enc_hidden * 4u;
        uint64_t proj1_need = (uint64_t)max_total_audio_tokens * (uint64_t)cfg->proj_mid;
        uint64_t audio_need = (uint64_t)max_total_audio_tokens * (uint64_t)cfg->dec_hidden;
        engine_reserve_bf16_buffer(&eng->ws_d_batch_concat, &eng->ws_d_batch_concat_capacity,
                                   concat_need, "ws_d_batch_concat");
        engine_reserve_bf16_buffer(&eng->ws_d_batch_proj1, &eng->ws_d_batch_proj1_capacity,
                                   proj1_need, "ws_d_batch_proj1");
        engine_reserve_bf16_buffer(&eng->ws_d_batch_audio, &eng->ws_d_batch_audio_capacity,
                                   audio_need, "ws_d_batch_audio");
        d_batch_concat = eng->ws_d_batch_concat;
        d_batch_proj1 = eng->ws_d_batch_proj1;
        d_batch_audio = eng->ws_d_batch_audio;
    }

    for (int i = 0; i < num_prefill; i++) {
        Sequence *seq = prefill_seqs[i];
        int enc_seq_len = 0;
        enc_seq_offsets[i] = total_encoder_tokens;
        audio_offsets[i] = total_audio_tokens;
        CHECK(audio_frontend_to_encoder_rows_into(eng, seq->pcm, seq->num_samples,
                                                  d_batch_enc != NULL
                                                      ? d_batch_enc + (size_t) total_encoder_tokens * cfg->enc_hidden
                                                      : NULL,
                                                  &enc_seq_len,
                                                  &audio_tokens[i]) == 0,
              "audio_frontend_to_encoder_rows_into failed for seq %d", seq->seq_id);
        enc_seq_lens[i] = enc_seq_len;
        audio_compute_tokens[i] = enc_seq_len / 4;
        if (audio_tokens[i] > audio_compute_tokens[i]) {
            audio_tokens[i] = audio_compute_tokens[i];
        }
        total_encoder_tokens += enc_seq_len;
        total_audio_tokens += audio_compute_tokens[i];
    }
    if (total_encoder_tokens > 0) {
        encoder_forward_packed(handle, ws, d_batch_enc, total_encoder_tokens,
                               num_prefill, enc_seq_lens, enc_seq_offsets, cfg);
        for (int i = 0; i < num_prefill; i++) {
            if (audio_compute_tokens[i] <= 0) {
                continue;
            }
            concat_4frames(d_batch_concat + (size_t) audio_offsets[i] * cfg->enc_hidden * 4,
                           d_batch_enc + (size_t) enc_seq_offsets[i] * cfg->enc_hidden,
                           enc_seq_lens[i], cfg->enc_hidden);
        }
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        audio_encode_ns = elapsed_ns;
    }

    if (num_prefill <= 0) {
        free(enc_seq_offsets);
        free(enc_seq_lens);
        free(replay_counts);
        free(audio_offsets);
        free(audio_compute_tokens);
        free(audio_tokens);
        free(prefill_seqs);
        return 0;
    }

    if (total_audio_tokens > 0) {
        Tensor *w1 = ws_get(ws, "multi_modal_projector.linear_1.weight");
        Tensor *b1 = ws_get(ws, "multi_modal_projector.linear_1.bias");
        Tensor *w2 = ws_get(ws, "multi_modal_projector.linear_2.weight");
        Tensor *b2 = ws_get(ws, "multi_modal_projector.linear_2.bias");

        bf16_linear_gelu(handle, d_batch_proj1, d_batch_concat,
                         (const bf16_t *)w1->data, (const bf16_t *)b1->data,
                         total_audio_tokens, cfg->proj_mid, cfg->enc_hidden * 4);
        bf16_linear(handle, d_batch_audio, d_batch_proj1,
                    (const bf16_t *)w2->data, (const bf16_t *)b2->data,
                    total_audio_tokens, cfg->dec_hidden, cfg->proj_mid);
    }

    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        audio_encode_ns += elapsed_ns;
        engine_profile_record_duration(audio_encode_ns,
                                       &eng->profile.prefill_audio_encode_ns_total,
                                       &eng->profile.prefill_audio_encode_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < num_prefill; i++) {
        Sequence *seq = prefill_seqs[i];
        int *prompt_ids = NULL;
        int prompt_len = 0;
        int total_seq = 0;
        int replay = seq->total_seq_len > 0 ? seq->n_generated : 0;
        int *combined_ids = NULL;

        if (replay < 0) {
            replay = 0;
        }
        if (replay > seq->max_new_tokens) {
            replay = seq->max_new_tokens;
        }
        replay_counts[i] = replay;

        CHECK(build_prompt_input_ids(cfg, audio_tokens[i],
                                     seq->user_prompt_token_ids,
                                     seq->num_user_prompt_tokens,
                                     &prompt_ids, &prompt_len) == 0,
              "build_prompt_input_ids failed for seq %d", seq->seq_id);
        total_seq = prompt_len + replay;
        combined_ids = (int *) malloc((size_t) total_seq * sizeof(*combined_ids));
        CHECK(combined_ids != NULL, "Failed to allocate prefill replay ids");
        memcpy(combined_ids, prompt_ids, (size_t) prompt_len * sizeof(*combined_ids));
        if (replay > 0) {
            CHECK(seq->generated_ids != NULL, "preempted seq %d has no generated ids", seq->seq_id);
            memcpy(combined_ids + prompt_len,
                   seq->generated_ids,
                   (size_t) replay * sizeof(*combined_ids));
        }

        seq_set_prompt(seq, combined_ids, total_seq, audio_tokens[i], NULL);
        seq->estimated_audio_tokens = audio_tokens[i];
        seq->estimated_prompt_tokens = prompt_len;
        seq->estimated_total_tokens = total_seq + seq->max_new_tokens;
        seq->estimated_total_blocks = (total_seq + m->kv_pool->block_size - 1) /
                                      m->kv_pool->block_size;
        free(prompt_ids);
        free(combined_ids);
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        prompt_build_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_prompt_build_ns_total,
                                       &eng->profile.prefill_prompt_build_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < num_prefill; i++) {
        Sequence *seq = prefill_seqs[i];
        int total_seq = seq->num_prompt_tokens;
        int needed_blocks = (total_seq + m->kv_pool->block_size - 1) / m->kv_pool->block_size;

        while (seq->bt.num_blocks < needed_blocks) {
            CHECK(engine_append_block(eng, seq) == 0,
                  "engine_append_block failed for seq %d", seq->seq_id);
        }

        seq_offsets[i] = total_tokens;
        seq_lens[i] = total_seq;
        batch_bts[i] = seq->bt;
        total_tokens += total_seq;
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        block_alloc_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_block_alloc_ns_total,
                                       &eng->profile.prefill_block_alloc_ns_max);
    }

    h_prefill_ids = (int *) malloc((size_t) total_tokens * sizeof(*h_prefill_ids));
    CHECK(h_prefill_ids != NULL, "Failed to allocate host prefill ids");

    for (int i = 0; i < num_prefill; i++) {
        Sequence *seq = prefill_seqs[i];
        memcpy(h_prefill_ids + seq_offsets[i],
               seq->prompt_token_ids,
               (size_t) seq_lens[i] * sizeof(*h_prefill_ids));
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    engine_reserve_prefill_workspace(eng, (uint64_t)total_tokens);
    err = cudaMemcpy(eng->ws_d_pf_ids, h_prefill_ids,
                     (size_t) total_tokens * sizeof(*h_prefill_ids),
                     cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess, "prefill batch ids copy failed: %s", cudaGetErrorString(err));
    embedding_lookup(eng->ws_d_pf_input, (const bf16_t *) embed_tokens->data,
                     eng->ws_d_pf_ids, total_tokens, cfg->dec_hidden);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        embed_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_embed_ns_total,
                                       &eng->profile.prefill_embed_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < num_prefill; i++) {
        if (audio_tokens[i] <= 0) {
            continue;
        }
        err = cudaMemcpy(eng->ws_d_pf_input + (size_t) (seq_offsets[i] + 3) * cfg->dec_hidden,
                         d_batch_audio + (size_t) audio_offsets[i] * cfg->dec_hidden,
                         (size_t) audio_tokens[i] * (size_t) cfg->dec_hidden * sizeof(bf16_t),
                         cudaMemcpyDeviceToDevice);
        CHECK(err == cudaSuccess, "prefill batch audio copy failed: %s", cudaGetErrorString(err));
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        audio_copy_ns = elapsed_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_audio_copy_ns_total,
                                       &eng->profile.prefill_audio_copy_ns_max);
        engine_profile_record_duration(prompt_build_ns + block_alloc_ns + embed_ns + audio_copy_ns,
                                       &eng->profile.prefill_prepare_ns_total,
                                       &eng->profile.prefill_prepare_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    decoder_prefill_paged_batched(handle, ws, eng->ws_d_pf_input, total_tokens,
                                  num_prefill, seq_lens, seq_offsets,
                                  m->kv_pool, batch_bts,
                                  (bf16_t *) m->dequant_buf.data, cfg);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_kernel_ns_total,
                                       &eng->profile.prefill_kernel_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < num_prefill; i++) {
        err = cudaMemcpy(eng->ws_d_batch_q + (size_t) i * cfg->dec_hidden,
                         eng->ws_d_pf_input + (size_t) (seq_offsets[i] + seq_lens[i] - 1) * cfg->dec_hidden,
                         (size_t) cfg->dec_hidden * sizeof(bf16_t),
                         cudaMemcpyDeviceToDevice);
        CHECK(err == cudaSuccess, "prefill batch last-hidden gather failed: %s", cudaGetErrorString(err));
    }
    lm_head_argmax_batched(handle,
                           (const bf16_t *) eng->ws_d_batch_q,
                           (const bf16_t *) lm_head_w->data,
                           num_prefill, cfg->dec_hidden, cfg->vocab_size,
                           sampled_tokens);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_sample_ns_total,
                                       &eng->profile.prefill_sample_ns_max);
    }

    for (int i = 0; i < num_prefill; i++) {
        Sequence *seq = prefill_seqs[i];
        int first_token = sampled_tokens[i];
        int replay = replay_counts[i];

        seq->total_seq_len = seq_lens[i];
        if (replay < seq->max_new_tokens) {
            seq->generated_ids[replay] = first_token;
            seq->n_generated = replay + 1;
        } else {
            seq->n_generated = replay;
        }
        if (is_eos(first_token, cfg)) {
            seq->eos_reason = 1;
        } else if (seq->n_generated >= seq->max_new_tokens) {
            seq->eos_reason = 2;
        }

        free(seq->prompt_token_ids);
        seq->prompt_token_ids = NULL;
    }

    free(h_prefill_ids);
    free(enc_seq_offsets);
    free(enc_seq_lens);
    free(replay_counts);
    free(audio_offsets);
    free(audio_compute_tokens);
    free(audio_tokens);
    free(prefill_seqs);
    return num_prefill;
}

/* Forward declaration */
static int decode_batch(Engine *eng, ScheduleResult *sr,
                        const Tensor *embed_tokens, const Tensor *lm_head_w);

/*
 * Mixed batch handler - processes both prefill and decode sequences.
 * Prefill sequences: have input_len > 1 (prompt tokens)
 * Decode sequences: have input_len = 1 (single new token)
 */
static int mixed_batch(Engine *eng,
                       ScheduleResult *sr,
                       const Tensor *embed_tokens,
                       const Tensor *lm_head_w) {
    int num_prefill = 0;
    int num_decode = 0;
    uint64_t start_ns;
    uint64_t elapsed_ns;

    /* Count prefill vs decode sequences */
    for (int i = 0; i < sr->num_seqs; i++) {
        if (sr->seq_input_lens[i] > 1) {
            num_prefill++;
        } else {
            num_decode++;
        }
    }

    /* Process prefill sequences first (they need full prompt embedding) */
    start_ns = (eng->profile.enabled && num_prefill > 0) ? engine_now_ns() : 0;
    CHECK(prefill_batch(eng, sr, embed_tokens, lm_head_w) >= 0,
          "prefill_batch failed");
    if (eng->profile.enabled && num_prefill > 0) {
        elapsed_ns = engine_now_ns() - start_ns;
        eng->profile.prefill_batches++;
        eng->profile.prefill_seqs += (uint64_t) num_prefill;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.prefill_ns_total,
                                       &eng->profile.prefill_ns_max);
        engine_profile_note_hist(eng->profile.prefill_batch_hist, num_prefill);
        engine_profile_note_pool(eng);
    }

    /* Process decode sequences together (if any) */
    if (num_decode > 0) {
        ScheduleResult decode_sr;
        decode_sr.seqs = malloc(num_decode * sizeof(Sequence *));
        decode_sr.seq_input_lens = malloc(num_decode * sizeof(int));
        decode_sr.seq_cached_lens = malloc(num_decode * sizeof(int));
        decode_sr.num_seqs = 0;
        decode_sr.type = STEP_DECODE;

        for (int i = 0; i < sr->num_seqs; i++) {
            if (sr->seq_input_lens[i] == 1) {
                decode_sr.seqs[decode_sr.num_seqs] = sr->seqs[i];
                decode_sr.seq_input_lens[decode_sr.num_seqs] = sr->seq_input_lens[i];
                decode_sr.seq_cached_lens[decode_sr.num_seqs] = sr->seq_cached_lens[i];
                decode_sr.num_seqs++;
            }
        }

        start_ns = eng->profile.enabled ? engine_now_ns() : 0;
        CHECK(decode_batch(eng, &decode_sr, embed_tokens, lm_head_w) == 0,
              "decode_batch failed");
        if (eng->profile.enabled) {
            elapsed_ns = engine_now_ns() - start_ns;
            eng->profile.decode_batches++;
            eng->profile.decode_tokens += (uint64_t) num_decode;
            engine_profile_record_duration(elapsed_ns,
                                           &eng->profile.decode_ns_total,
                                           &eng->profile.decode_ns_max);
            engine_profile_note_hist(eng->profile.decode_batch_hist, num_decode);
            engine_profile_note_pool(eng);
        }

        free(decode_sr.seqs);
        free(decode_sr.seq_input_lens);
        free(decode_sr.seq_cached_lens);
    }

    return 0;
}

static int decode_batch(Engine *eng,
                        ScheduleResult *sr,
                        const Tensor *embed_tokens,
                        const Tensor *lm_head_w) {
    Model *m = eng->model;
    cublasHandle_t handle = (cublasHandle_t)m->cublas;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    int num_seqs = sr->num_seqs;
    int *h_prev_ids;
    int *h_seq_lens;
    int *seq_indices;
    bf16_t **query_ptrs;
    BlockTable *batch_bts;
    uint64_t start_ns;
    uint64_t elapsed_ns;
    cudaError_t err;

    if (num_seqs <= 0) {
        return 0;
    }

    h_prev_ids = eng->h_decode_prev_ids;
    h_seq_lens = eng->h_decode_seq_lens;
    seq_indices = eng->h_decode_seq_indices;
    query_ptrs = eng->h_decode_query_ptrs;
    batch_bts = eng->h_decode_batch_bts;
    CHECK(h_prev_ids && h_seq_lens && seq_indices && query_ptrs && batch_bts,
          "Decode batch metadata buffers are not initialized");

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    for (int i = 0; i < num_seqs; i++) {
        Sequence *seq = sr->seqs[i];
        h_prev_ids[i] = seq->generated_ids[seq->n_generated - 1];
        h_seq_lens[i] = seq->total_seq_len;
        seq_indices[i] = seq->seq_idx;
        batch_bts[i] = seq->bt;
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.decode_prepare_ns_total,
                                       &eng->profile.decode_prepare_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    err = cudaMemcpy(eng->ws_d_prev_ids, h_prev_ids,
                     (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess, "decode prev_ids copy failed: %s", cudaGetErrorString(err));
    err = cudaMemcpy(eng->ws_d_seq_lens, h_seq_lens,
                     (size_t)num_seqs * sizeof(int), cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess, "decode seq_lens copy failed: %s", cudaGetErrorString(err));
    embedding_lookup(eng->ws_d_batch_q, (const bf16_t *)embed_tokens->data,
                     eng->ws_d_prev_ids, num_seqs, cfg->dec_hidden);
    for (int i = 0; i < num_seqs; i++) {
        query_ptrs[i] = eng->ws_d_batch_q + (size_t)i * cfg->dec_hidden;
    }
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.decode_h2d_ns_total,
                                       &eng->profile.decode_h2d_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    decoder_decode_step_batched(handle, ws, query_ptrs,
                                m->kv_pool, batch_bts,
                                eng->ws_d_seq_lens, h_seq_lens,
                                num_seqs, seq_indices,
                                (bf16_t *)m->dequant_buf.data, cfg);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.decode_kernel_ns_total,
                                       &eng->profile.decode_kernel_ns_max);
    }

    start_ns = eng->profile.enabled ? engine_now_ns() : 0;
    lm_head_argmax_batched(handle,
                           (const bf16_t *) eng->ws_d_batch_q,
                           (const bf16_t *) lm_head_w->data,
                           num_seqs, cfg->dec_hidden, cfg->vocab_size,
                           h_prev_ids);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.decode_sample_ns_total,
                                       &eng->profile.decode_sample_ns_max);
    }

    for (int i = 0; i < num_seqs; i++) {
        Sequence *seq = sr->seqs[i];
        int next_token = h_prev_ids[i];
        seq->generated_ids[seq->n_generated] = next_token;
        if (is_eos(next_token, cfg)) {
            seq->eos_reason = 1;
        } else if (seq->n_generated + 1 >= seq->max_new_tokens) {
            seq->eos_reason = 2;
        }
    }

    return 0;
}

static void finish_sequence(Engine *eng,
                            Sequence *seq,
                            EngineResult *results,
                            int *n_results,
                            int max_results) {
    char *queued_text;

    (void) results;
    (void) n_results;
    (void) max_results;

    if (seq->status == SEQ_FINISHED) {
        return;
    }

    if (!seq->output_text) {
        seq->output_text = decode_generated_text(eng, seq->generated_ids, seq->n_generated);
    }

    if (seq->audio_embeds) {
        engine_profile_note_audio_free(eng,
                                       engine_audio_embed_bytes(&eng->model->cfg, seq->num_audio_tokens));
        cudaFree(seq->audio_embeds);
        seq->audio_embeds = NULL;
    }
    if (eng->model && eng->model->kv_pool && seq->bt.num_blocks > 0) {
        block_table_free(&seq->bt, eng->model->kv_pool);
        block_table_upload(&seq->bt);
    }

    queued_text = seq->output_text;
    seq->output_text = NULL;
    seq->status = SEQ_FINISHED;
    scheduler_release(&eng->scheduler, seq);
    eng->sequences_done++;
    engine_queue_result(eng, seq->seq_id, queued_text, seq->n_generated);
    engine_profile_note_scheduler(eng);
    engine_profile_note_pool(eng);
}

static void compact_running_sequences(Engine *eng) {
    int dst = 0;
    Scheduler *sch;

    if (!eng) {
        return;
    }

    sch = &eng->scheduler;

    for (int src = 0; src < sch->running_len; src++) {
        Sequence *seq = sch->running[src];
        if (seq && seq->status != SEQ_FINISHED) {
            sch->running[dst++] = seq;
        } else if (seq != NULL) {
            engine_release_slot(eng, seq->slot_idx);
            seq_free(seq);
        }
    }
    for (int i = dst; i < sch->running_len; i++) {
        sch->running[i] = NULL;
    }
    sch->running_len = dst;
    engine_profile_note_scheduler(eng);
}

static int sequence_is_finished(const Sequence *seq, const ModelConfig *cfg) {
    if (!seq || seq->n_generated <= 0) {
        return 0;
    }

    if (seq->eos_reason != 0) {
        return 1;
    }
    if (seq->n_generated >= seq->max_new_tokens) {
        return 1;
    }
    return is_eos(seq->generated_ids[seq->n_generated - 1], cfg);
}

int engine_init(Engine *eng, const char *model_dir) {
    int i;
    int max_batched_tokens;

    memset(eng, 0, sizeof(*eng));

    eng->model = calloc(1, sizeof(Model));
    if (!eng->model) return -1;

    if (model_load(eng->model, model_dir) != 0) {
        free(eng->model);
        eng->model = NULL;
        return -1;
    }
    if (eng->model->kv_pool != NULL) {
        kv_pool_free(eng->model->kv_pool);
    }

    {
        char tok_path[512];
        snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
        eng->tok = tokenizer_load(tok_path);
        if (!eng->tok) {
            fprintf(stderr, "WARNING: Failed to load tokenizer, continuing anyway\n");
        }
    }

    memset(&eng->mel_filters, 0, sizeof(Tensor));
    eng->max_seqs = engine_max_sched_seqs();
    max_batched_tokens = engine_max_batched_tokens();
    eng->sequences = calloc((size_t)eng->max_seqs, sizeof(Sequence));
    eng->pending_results = calloc((size_t)eng->max_seqs, sizeof(EngineResult));
    eng->free_slots = calloc((size_t)eng->max_seqs, sizeof(int));
    if (!eng->sequences || !eng->pending_results || !eng->free_slots) {
        engine_free(eng);
        return -1;
    }

    eng->free_slot_count = eng->max_seqs;
    for (i = 0; i < eng->max_seqs; ++i) {
        eng->free_slots[i] = eng->max_seqs - 1 - i;
    }

    engine_alloc_workspace(eng);
    scheduler_init(&eng->scheduler, eng->model->kv_pool,
                   eng->max_seqs, max_batched_tokens);
    if (max_batched_tokens > 0) {
        fprintf(stderr,
                "Scheduler config: max_seqs=%d, max_batched_tokens=%d\n",
                eng->max_seqs, max_batched_tokens);
    } else {
        fprintf(stderr,
                "Scheduler config: max_seqs=%d, max_batched_tokens=disabled\n",
                eng->max_seqs);
    }
    eng->runtime_kv_prepared = 0;
    eng->profile.enabled = engine_profile_enabled_from_env();
    engine_profile_reset(eng);

    return 0;
}

void engine_free(Engine *eng) {
    if (!eng) {
        return;
    }

    engine_free_workspace(eng);

    if (eng->sequences) {
        for (int i = 0; i < eng->max_seqs; i++) {
            Sequence *seq = &eng->sequences[i];
            if (eng->model && eng->model->kv_pool && seq->bt.num_blocks > 0) {
                block_table_free(&seq->bt, eng->model->kv_pool);
            }
            if (seq->audio_embeds) {
                engine_profile_note_audio_free(eng,
                                               engine_audio_embed_bytes(&eng->model->cfg, seq->num_audio_tokens));
                cudaFree(seq->audio_embeds);
                seq->audio_embeds = NULL;
            }
            seq_free(seq);
        }
        free(eng->sequences);
        eng->sequences = NULL;
    }

    if (eng->pending_results) {
        for (int i = 0; i < eng->pending_result_count; ++i) {
            free(eng->pending_results[i].text);
            eng->pending_results[i].text = NULL;
        }
        free(eng->pending_results);
        eng->pending_results = NULL;
    }

    free(eng->free_slots);
    eng->free_slots = NULL;

    scheduler_free(&eng->scheduler);

    if (eng->model) {
        model_free(eng->model);
        free(eng->model);
        eng->model = NULL;
    }

    if (eng->tok) {
        tokenizer_free(eng->tok);
        eng->tok = NULL;
    }

    if (eng->mel_filters.data) {
        cudaFree(eng->mel_filters.data);
        memset(&eng->mel_filters, 0, sizeof(Tensor));
    }

    eng->max_seqs = 0;
    eng->next_request_id = 0;
    eng->free_slot_count = 0;
    eng->pending_result_count = 0;
    eng->sequences_ready = 0;
    eng->sequences_done = 0;
}

void engine_reset_batch(Engine *eng) {
    int i;
    if (!eng) return;
    eng->free_slot_count = eng->max_seqs;
    eng->pending_result_count = 0;
    eng->sequences_ready = 0;
    eng->sequences_done = 0;
    scheduler_reset(&eng->scheduler);
    for (i = 0; i < eng->max_seqs; i++) {
        if (eng->pending_results) {
            free(eng->pending_results[i].text);
            eng->pending_results[i].text = NULL;
            eng->pending_results[i].seq_id = 0;
        }
        if (eng->free_slots) {
            eng->free_slots[i] = eng->max_seqs - 1 - i;
        }
        if (eng->sequences[i].audio_embeds) {
            engine_profile_note_audio_free(eng,
                                           engine_audio_embed_bytes(&eng->model->cfg,
                                                                    eng->sequences[i].num_audio_tokens));
            cudaFree(eng->sequences[i].audio_embeds);
            eng->sequences[i].audio_embeds = NULL;
        }
        seq_free(&eng->sequences[i]);
    }
    if (eng->model && eng->model->kv_pool) {
        kv_pool_zero_cache(eng->model->kv_pool);
    }
    engine_profile_note_scheduler(eng);
    engine_profile_note_pool(eng);
}

void engine_profile_report_run(Engine *eng, const char *scope) {
    engine_profile_note_scheduler(eng);
    engine_profile_note_pool(eng);
    engine_profile_report(eng, scope);
}

void engine_profile_reset_run(Engine *eng) {
    engine_profile_reset(eng);
}

int engine_submit_with_prompt_ids(Engine *eng, const float *pcm, int num_samples,
                                  const int *prompt_token_ids, int num_prompt_tokens) {
    ModelConfig *cfg;
    Sequence *seq;
    int slot_idx;
    int estimated_audio_tokens;
    int estimated_prompt_tokens;
    int estimated_total_tokens;
    int estimated_total_blocks;
    int block_size;
    uint64_t start_ns = 0;
    uint64_t elapsed_ns;

    if (eng != NULL && eng->profile.enabled) {
        start_ns = engine_now_ns();
    }

    if (!eng || !eng->model || !eng->model->kv_pool || pcm == NULL || num_samples <= 0) {
        return -1;
    }
    if (!eng->runtime_kv_prepared && engine_prepare_runtime_kv_pool(eng) != 0) {
        return -1;
    }
    slot_idx = engine_acquire_slot(eng);
    if (slot_idx < 0) {
        return -1;
    }

    cfg = &eng->model->cfg;
    block_size = eng->model->kv_pool->block_size > 0 ? eng->model->kv_pool->block_size : KV_DEFAULT_BLOCK_SIZE;
    estimated_audio_tokens = estimate_audio_tokens_from_samples(num_samples);
    if (estimated_audio_tokens > ((cfg->max_audio_frames + 1) / 2 + 3) / 4) {
        estimated_audio_tokens = ((cfg->max_audio_frames + 1) / 2 + 3) / 4;
    }
    if (num_prompt_tokens < 0) {
        num_prompt_tokens = 0;
    }
    estimated_prompt_tokens = prompt_total_token_count(num_prompt_tokens) + estimated_audio_tokens;
    estimated_total_tokens = estimated_prompt_tokens + cfg->max_new_tokens;
    estimated_total_blocks = (estimated_prompt_tokens + block_size - 1) / block_size;
    if (estimated_total_blocks < 1) {
        estimated_total_blocks = 1;
    }

    seq = &eng->sequences[slot_idx];
    if (seq->audio_embeds) {
        cudaFree(seq->audio_embeds);
        seq->audio_embeds = NULL;
    }
    seq_free(seq);
    seq_init(seq, eng->next_request_id, slot_idx, cfg->max_new_tokens);
    seq->generated_ids = malloc((size_t)cfg->max_new_tokens * sizeof(int));
    if (!seq->generated_ids) {
        engine_release_slot(eng, slot_idx);
        return -1;
    }

    if (seq_set_audio(seq, pcm, num_samples,
                      estimated_audio_tokens,
                      estimated_prompt_tokens,
                      estimated_total_tokens,
                      estimated_total_blocks) != 0) {
        seq_free(seq);
        engine_release_slot(eng, slot_idx);
        return -1;
    }
    if (seq_set_user_prompt(seq, prompt_token_ids, num_prompt_tokens) != 0) {
        seq_free(seq);
        engine_release_slot(eng, slot_idx);
        return -1;
    }

    scheduler_add(&eng->scheduler, seq);
    eng->sequences_ready++;
    eng->next_request_id++;
    engine_profile_note_scheduler(eng);
    engine_profile_note_pool(eng);
    if (eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        eng->profile.admission_calls++;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.admission_ns_total,
                                       &eng->profile.admission_ns_max);
    }
    return seq->seq_id;
}

int engine_submit(Engine *eng, const float *pcm, int num_samples) {
    return engine_submit_with_prompt_ids(eng, pcm, num_samples, NULL, 0);
}

int engine_step(Engine *eng, EngineResult *results, int *n_results, int max_results) {
    Model *m;
    ModelConfig *cfg;
    Tensor *embed_tokens;
    Tensor *lm_head_w;
    ScheduleResult *sr;
    uint64_t start_ns = 0;
    uint64_t elapsed_ns;
    int finished;

    if (eng != NULL && eng->profile.enabled) {
        start_ns = engine_now_ns();
    }

    if (n_results) {
        *n_results = 0;
    }
    if (!eng || !eng->model) {
        return 1;
    }
    if (scheduler_is_finished(&eng->scheduler)) {
        if (results != NULL && n_results != NULL && max_results > 0) {
            engine_drain_results(eng, results, n_results, max_results);
        }
        finished = eng->pending_result_count == 0 ? 1 : 0;
        goto done;
    }

    m = eng->model;
    cfg = &m->cfg;
    embed_tokens = ws_get(&m->ws, "language_model.model.embed_tokens.weight");
    lm_head_w = ws_get(&m->ws, "language_model.lm_head.weight");
    CHECK(embed_tokens != NULL, "embed_tokens weight not found");
    CHECK(lm_head_w != NULL, "lm_head weight not found");

    sr = scheduler_schedule(&eng->scheduler);
    engine_profile_note_scheduler(eng);
    engine_profile_note_pool(eng);
    if (!sr || sr->num_seqs == 0) {
        if (results != NULL && n_results != NULL && max_results > 0) {
            engine_drain_results(eng, results, n_results, max_results);
        }
        finished = scheduler_is_finished(&eng->scheduler) && eng->pending_result_count == 0 ? 1 : 0;
        goto done;
    }

    /* Mixed batch: handle both prefill and decode sequences together */
    mixed_batch(eng, sr, embed_tokens, lm_head_w);

    scheduler_postprocess(&eng->scheduler, sr);

    for (int i = 0; i < eng->scheduler.running_len; i++) {
        Sequence *seq = eng->scheduler.running[i];
        if (sequence_is_finished(seq, cfg)) {
            finish_sequence(eng, seq, results, n_results, max_results);
        }
    }
    compact_running_sequences(eng);

    if (results != NULL && n_results != NULL && max_results > 0) {
        engine_drain_results(eng, results, n_results, max_results);
    }

    finished = scheduler_is_finished(&eng->scheduler) && eng->pending_result_count == 0 ? 1 : 0;

done:
    if (eng != NULL && eng->profile.enabled) {
        elapsed_ns = engine_now_ns() - start_ns;
        eng->profile.step_calls++;
        engine_profile_record_duration(elapsed_ns,
                                       &eng->profile.step_ns_total,
                                       &eng->profile.step_ns_max);
    }
    return finished;
}

char *engine_transcribe(Engine *eng, const float *pcm, int num_samples) {
    Model *m = eng->model;
    cublasHandle_t handle = (cublasHandle_t)m->cublas;
    ModelConfig *cfg = &m->cfg;
    WeightStore *ws = &m->ws;
    int actual_audio_tokens = 0;
    bf16_t *audio_embeds;
    int *h_input_ids = NULL;
    int total_seq = 0;
    int use_paged_kv;
    BlockTable bt;
    int first_token = 0;
    Tensor *embed_tokens;
    Tensor *lm_head_w;
    int *generated_tokens;
    int n_generated = 0;
    char *text;

    prepare_single_request_kv_pool(m);
    audio_embeds = encode_audio(eng, pcm, num_samples, &actual_audio_tokens);
    CHECK(build_prompt_input_ids(cfg, actual_audio_tokens, NULL, 0, &h_input_ids, &total_seq) == 0,
          "Failed to build prompt token IDs");

    engine_reserve_prefill_workspace(eng, (uint64_t)total_seq);
    cudaMemcpy(eng->ws_d_pf_ids, h_input_ids, (size_t)total_seq * sizeof(int), cudaMemcpyHostToDevice);
    free(h_input_ids);
    h_input_ids = NULL;

    embed_tokens = ws_get(ws, "language_model.model.embed_tokens.weight");
    CHECK(embed_tokens != NULL, "embed_tokens weight not found");

    embedding_lookup(eng->ws_d_pf_input, (const bf16_t *)embed_tokens->data,
                     eng->ws_d_pf_ids, total_seq, cfg->dec_hidden);

    for (int i = 0; i < actual_audio_tokens; i++) {
        cudaMemcpy(eng->ws_d_pf_input + (size_t)(3 + i) * cfg->dec_hidden,
                   audio_embeds + (size_t)i * cfg->dec_hidden,
                   (size_t)cfg->dec_hidden * sizeof(bf16_t),
                   cudaMemcpyDeviceToDevice);
    }
    cudaFree(audio_embeds);

    use_paged_kv = m->kv_pool != NULL;
    memset(&bt, 0, sizeof(bt));

    if (use_paged_kv) {
        if (m->kv.layers) {
            kv_cache_free(&m->kv);
            m->kv.layers = NULL;
            m->kv.seq_len = 0;
            m->kv.max_seq = 0;
        }
        {
            int total_capacity = total_seq + cfg->max_new_tokens;
            int n_blocks = (total_capacity + m->kv_pool->block_size - 1) / m->kv_pool->block_size;
            block_table_init(&bt, m->kv_pool, 0);
            CHECK(block_table_alloc(&bt, m->kv_pool, n_blocks) == 0,
                  "Failed to allocate %d KV blocks", n_blocks);
            block_table_upload(&bt);
            cudaMemcpy(eng->ws_d_sr_seq_lens, &total_seq, sizeof(int), cudaMemcpyHostToDevice);

            decoder_prefill_paged(handle, ws, eng->ws_d_pf_input, total_seq,
                                  m->kv_pool, &bt,
                                  (bf16_t *)m->dequant_buf.data, cfg);
        }
    } else {
        m->kv.seq_len = 0;
        decoder_prefill(handle, ws, eng->ws_d_pf_input, total_seq, &m->kv,
                        (bf16_t *)m->dequant_buf.data, cfg);
    }

    lm_head_w = ws_get(ws, "language_model.lm_head.weight");
    CHECK(lm_head_w != NULL, "lm_head weight not found");

    lm_head_argmax(eng->ws_d_pf_input + (size_t)(total_seq - 1) * cfg->dec_hidden,
                   (const bf16_t *)lm_head_w->data,
                   cfg->dec_hidden, cfg->vocab_size, &first_token);

    generated_tokens = malloc((size_t)cfg->max_new_tokens * sizeof(int));
    CHECK(generated_tokens != NULL, "Failed to allocate generated token buffer");
    generated_tokens[0] = first_token;
    n_generated = 1;

    if (!is_eos(first_token, cfg)) {
        for (int t = 1; t < cfg->max_new_tokens; t++) {
            int prev_id = generated_tokens[t - 1];

            cudaMemcpy(eng->ws_d_single_id, &prev_id, sizeof(int), cudaMemcpyHostToDevice);
            embedding_lookup(eng->ws_d_embed, (const bf16_t *)embed_tokens->data,
                             eng->ws_d_single_id, 1, cfg->dec_hidden);

            if (use_paged_kv) {
                decoder_decode_step_paged(handle, ws, eng->ws_d_embed,
                                          m->kv_pool, &bt, eng->ws_d_sr_seq_lens, 0,
                                          (bf16_t *)m->dequant_buf.data, cfg);
            } else {
                decoder_decode_step(handle, ws, eng->ws_d_embed, &m->kv,
                                    (bf16_t *)m->dequant_buf.data, cfg);
            }

            lm_head_argmax(eng->ws_d_embed,
                           (const bf16_t *)lm_head_w->data,
                           cfg->dec_hidden, cfg->vocab_size, &generated_tokens[t]);

            n_generated = t + 1;

            if (is_eos(generated_tokens[t], cfg)) {
                break;
            }
        }
    }

    if (bt.h_table) {
        block_table_free(&bt, m->kv_pool);
        free(bt.h_table);
    }

    text = decode_generated_text(eng, generated_tokens, n_generated);
    free(generated_tokens);
    return text;
}
