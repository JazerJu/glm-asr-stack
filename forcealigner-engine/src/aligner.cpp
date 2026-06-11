#include <cmath>
#include <clocale>
#include <cwctype>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "aligner.h"
#include "cuda_kernels.h"

namespace {

struct Utf8CodePoint {
    uint32_t value;
    int length;
};

static void ensure_utf8_locale() {
    static int locale_initialized = 0;
    if (!locale_initialized) {
        setlocale(LC_CTYPE, "");
        locale_initialized = 1;
    }
}

static Utf8CodePoint decode_utf8_codepoint(const char *s) {
    unsigned char c0 = (unsigned char)s[0];
    if ((c0 & 0x80u) == 0) {
        return {c0, 1};
    }
    if ((c0 & 0xE0u) == 0xC0u && (s[1] & 0xC0) == 0x80) {
        return {
            ((uint32_t)(c0 & 0x1Fu) << 6) |
            (uint32_t)((unsigned char)s[1] & 0x3Fu),
            2
        };
    }
    if ((c0 & 0xF0u) == 0xE0u && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        return {
            ((uint32_t)(c0 & 0x0Fu) << 12) |
            ((uint32_t)((unsigned char)s[1] & 0x3Fu) << 6) |
            (uint32_t)((unsigned char)s[2] & 0x3Fu),
            3
        };
    }
    if ((c0 & 0xF8u) == 0xF0u && (s[1] & 0xC0) == 0x80 &&
        (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
        return {
            ((uint32_t)(c0 & 0x07u) << 18) |
            ((uint32_t)((unsigned char)s[1] & 0x3Fu) << 12) |
            ((uint32_t)((unsigned char)s[2] & 0x3Fu) << 6) |
            (uint32_t)((unsigned char)s[3] & 0x3Fu),
            4
        };
    }
    return {c0, 1};
}

static int is_cjk_char(uint32_t code) {
    return (
        (0x4E00u <= code && code <= 0x9FFFu) ||
        (0x3400u <= code && code <= 0x4DBFu) ||
        (0x20000u <= code && code <= 0x2A6DFu) ||
        (0x3040u <= code && code <= 0x309Fu) ||
        (0x30A0u <= code && code <= 0x30FFu)
    );
}

static int is_kept_char(uint32_t code) {
    if (code == (uint32_t)'\'' || code == (uint32_t)'-') {
        return 1;
    }
    ensure_utf8_locale();
    return std::iswalnum((wint_t)code) ? 1 : 0;
}

static std::string clean_token(const std::string &token) {
    std::string cleaned;
    for (size_t i = 0; i < token.size();) {
        Utf8CodePoint cp = decode_utf8_codepoint(token.c_str() + i);
        if (is_kept_char(cp.value)) {
            cleaned.append(token, i, (size_t)cp.length);
        }
        i += (size_t)cp.length;
    }
    return cleaned;
}

static void split_segment_with_chinese(const std::string &segment,
                                       std::vector<std::string> *out_words) {
    std::string buf;
    for (size_t i = 0; i < segment.size();) {
        Utf8CodePoint cp = decode_utf8_codepoint(segment.c_str() + i);
        if (is_cjk_char(cp.value)) {
            if (!buf.empty()) {
                out_words->push_back(buf);
                buf.clear();
            }
            out_words->push_back(segment.substr(i, (size_t)cp.length));
        } else {
            buf.append(segment, i, (size_t)cp.length);
        }
        i += (size_t)cp.length;
    }
    if (!buf.empty()) {
        out_words->push_back(buf);
    }
}

static std::vector<std::string> tokenize_space_lang(const char *text) {
    std::vector<std::string> words;
    std::string segment;
    for (const unsigned char *p = (const unsigned char *)text; ; ++p) {
        if (*p != '\0' && !isspace(*p)) {
            segment.push_back((char)*p);
            continue;
        }
        if (!segment.empty()) {
            std::string cleaned = clean_token(segment);
            if (!cleaned.empty()) {
                split_segment_with_chinese(cleaned, &words);
            }
            segment.clear();
        }
        if (*p == '\0') {
            break;
        }
    }
    return words;
}

static int floordiv(int a, int b) {
    return a / b - (a % b != 0 && ((a ^ b) < 0));
}

static int get_audio_token_count(int feature_frames, int max_source_positions) {
    int input_lengths_leave = feature_frames % 100;
    int feat_lengths = floordiv(input_lengths_leave - 1, 2) + 1;
    int output_lengths = floordiv(floordiv(feat_lengths - 1, 2) + 1 - 1, 2) + 1 +
                         (feature_frames / 100) * 13;
    if (output_lengths < 0) {
        output_lengths = 0;
    }
    if (output_lengths > max_source_positions) {
        output_lengths = max_source_positions;
    }
    return output_lengths;
}

}  // namespace

int aligner_init(Aligner *a, const char *model_dir) {
    memset(a, 0, sizeof(*a));

    fprintf(stderr, "[aligner] Loading model from %s\n", model_dir);
    if (aligner_model_load(&a->model, model_dir) != 0) {
        fprintf(stderr, "ERROR: Failed to load model\n");
        return -1;
    }

    fprintf(stderr, "[aligner] Loading tokenizer\n");
    if (tokenizer_load(&a->tokenizer, model_dir) != 0) {
        fprintf(stderr, "ERROR: Failed to load tokenizer\n");
        return -1;
    }

    AlignerConfig *cfg = &a->model.cfg;
    int max_mel_frames = 3000;
    int max_frames = cfg->max_source_positions;
    int max_seq = 8192;

    size_t mel_bytes = (size_t)cfg->num_mel_bins * max_mel_frames * sizeof(bf16_t);
    size_t embed_bytes = (size_t)max_frames * cfg->enc_output_dim * sizeof(bf16_t);
    size_t input_bytes = (size_t)max_seq * cfg->dec_hidden * sizeof(bf16_t);
    size_t logits_bytes = (size_t)max_seq * cfg->classify_num * sizeof(bf16_t);

    fprintf(stderr, "[aligner] Allocating workspace: %.1f MB\n",
            (mel_bytes + embed_bytes + input_bytes + logits_bytes) / 1e6);

    cudaMalloc(&a->mel_buf, mel_bytes);
    cudaMalloc(&a->audio_embeds, embed_bytes);
    cudaMalloc(&a->input_embeds, input_bytes);
    cudaMalloc(&a->logits, logits_bytes);
    cudaMalloc(&a->input_ids_buf, max_seq * sizeof(int));
    cudaMalloc(&a->timestamp_buf, MAX_WORDS * 2 * sizeof(int));

    cudaStreamCreate(&a->stream);
    cublasSetStream((cublasHandle_t)a->model.cublas, a->stream);
    a->stream_created = 1;
    cudaEventCreate(&a->ev_start);
    cudaEventCreate(&a->ev_mel_done);
    cudaEventCreate(&a->ev_enc_done);
    cudaEventCreate(&a->ev_dec_done);
    cudaEventCreate(&a->ev_end);
    a->events_created = 1;

    a->initialized = 1;
    fprintf(stderr, "[aligner] Initialized successfully\n");
    return 0;
}

int aligner_align(Aligner *a, const char *audio_path,
                  const char *text, const char *language,
                  AlignWord *out_words, int max_words) {
    if (!a->initialized) return -1;

    AlignerConfig *cfg = &a->model.cfg;
    cudaStream_t stream = a->stream;

    /* Clear any stale async errors from previous invocation */
    cudaGetLastError();

    cudaEventRecord(a->ev_start, stream);

    /* 1. Load audio + 2. Compute mel spectrogram */
    fprintf(stderr, "[align] Loading audio: %s\n", audio_path);
    float *pcm = NULL;
    int audio_count = 0;
    if (load_audio_file(audio_path, &pcm, &audio_count) < 0 || audio_count == 0) {
        fprintf(stderr, "ERROR: Failed to load audio\n");
        return -1;
    }
    fprintf(stderr, "[align] Audio: %d samples (%.2fs)\n", audio_count, audio_count / 16000.0);

    /* Peak normalize: waveform / (max(|waveform|) + 1e-8) */
    {
        float peak = 0.0f;
        for (int i = 0; i < audio_count; i++) {
            float a = fabsf(pcm[i]);
            if (a > peak) peak = a;
        }
        if (peak > 1.0f) {
            float scale = 1.0f / peak;
            for (int i = 0; i < audio_count; i++) pcm[i] *= scale;
        }
        for (int i = 0; i < audio_count; i++) {
            if (pcm[i] > 1.0f) pcm[i] = 1.0f;
            else if (pcm[i] < -1.0f) pcm[i] = -1.0f;
        }
    }

    int mel_T = audio_count / 160;
    fprintf(stderr, "[align] Computing mel spectrogram: %d frames (%d samples)\n",
            mel_T, audio_count);

    if (audio_count > a->pcm_device_capacity) {
        if (a->pcm_device_buf) cudaFree(a->pcm_device_buf);
        cudaMalloc(&a->pcm_device_buf, (size_t)audio_count * sizeof(float));
        a->pcm_device_capacity = audio_count;
    }
    cudaMemcpy(a->pcm_device_buf, pcm, (size_t)audio_count * sizeof(float), cudaMemcpyHostToDevice);
    free(pcm);

    cudaMemset(a->mel_buf, 0, (size_t)cfg->num_mel_bins * 3000 * sizeof(bf16_t));
    cuda_mel_spectrogram((uint16_t *)a->mel_buf, a->pcm_device_buf, audio_count,
                         400, 160, cfg->num_mel_bins, 16000, stream);
    cudaEventRecord(a->ev_mel_done, stream);

    /* 3. Audio encoder forward */
    fprintf(stderr, "[align] Running audio encoder\n");
    cuda_audio_encoder_forward(
        a->audio_embeds, a->mel_buf, mel_T,
        &a->model.ws, cfg, a->model.cublas, stream
    );
    cudaEventRecord(a->ev_enc_done, stream);

    cublasSetStream((cublasHandle_t)a->model.cublas, stream);

    int audio_token_count = get_audio_token_count(mel_T, cfg->max_source_positions);
    fprintf(stderr, "[align] Audio tokens: %d\n", audio_token_count);

    /* If FA_DUMP_ENCODER_FULL is set, save full encoder output for cross-feed debugging */
    {
        const char *dump_dir = getenv("FA_DUMP_ENCODER_FULL");
        if (dump_dir && audio_token_count > 0) {
            cudaEventSynchronize(a->ev_enc_done);
            size_t feat_bytes = (size_t)audio_token_count * cfg->dec_hidden * sizeof(bf16_t);
            void *feat_host = malloc(feat_bytes);
            cudaMemcpy(feat_host, a->audio_embeds, feat_bytes, cudaMemcpyDeviceToHost);
            char fpath[512];
            snprintf(fpath, sizeof(fpath), "%s/encoder_out_%dtok.bf16", dump_dir, audio_token_count);
            FILE *f = fopen(fpath, "wb");
            if (f) { fwrite(feat_host, 1, feat_bytes, f); fclose(f); fprintf(stderr, "[dump] Saved encoder output: %s\n", fpath); }
            free(feat_host);
        }
    }

    std::vector<std::string> words = tokenize_space_lang(text);
    int num_words = (int)words.size();
    fprintf(stderr, "[align] Word tokens: %d\n", num_words);

    /* 6. Build input_ids: audio markers + [word BPE ids] + <timestamp> + <timestamp> for each word */
    int input_ids_host[MAX_PROMPT_IDS];
    int pos = 0;
    input_ids_host[pos++] = cfg->audio_start_id;
    for (int i = 0; i < audio_token_count; i++) {
        input_ids_host[pos++] = cfg->audio_pad_id;
    }
    input_ids_host[pos++] = cfg->audio_end_id;

    int total_bpe_tokens = 0;
    for (int w = 0; w < num_words; w++) {
        int word_bpe_ids[MAX_PROMPT_IDS];
        int word_bpe_count = tokenizer_encode(&a->tokenizer, words[w].c_str(), word_bpe_ids, MAX_PROMPT_IDS);
        if (word_bpe_count < 0) {
            fprintf(stderr, "ERROR: Failed to BPE-tokenize word: %s\n", words[w].c_str());
            cudaStreamDestroy(stream);
            return -1;
        }
        if (pos + word_bpe_count + 2 > MAX_PROMPT_IDS) {
            fprintf(stderr, "ERROR: Prompt too long (%d ids)\n", pos + word_bpe_count + 2);
            cudaStreamDestroy(stream);
            return -1;
        }
        for (int t = 0; t < word_bpe_count; t++) {
            input_ids_host[pos++] = word_bpe_ids[t];
        }
        input_ids_host[pos++] = cfg->timestamp_id;
        input_ids_host[pos++] = cfg->timestamp_id;
        total_bpe_tokens += word_bpe_count;
    }
    int seq_len = pos;
    int num_timestamps_expected = num_words * 2;
    fprintf(stderr, "[align] Sequence length: %d (text BPE tokens: %d)\n", seq_len, total_bpe_tokens);

    /* Upload input_ids to GPU */
    cudaMemcpy(a->input_ids_buf, input_ids_host, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    /* 6. Build input embeddings: token embeddings + inject audio embeds */
    Tensor *embed_w = ws_get(&a->model.ws, "thinker.model.embed_tokens.weight");
    if (!embed_w) {
        fprintf(stderr, "ERROR: embed_tokens weight not found\n");
        return -1;
    }

    cuda_embedding_lookup((uint16_t *)a->input_embeds,
                          (const uint16_t *)embed_w->data,
                          a->input_ids_buf, seq_len, cfg->dec_hidden, stream);

    /* Find audio_pad positions and scatter audio embeddings */
    int *h_ids = (int *)malloc(seq_len * sizeof(int));
    cudaMemcpy(h_ids, a->input_ids_buf, seq_len * sizeof(int), cudaMemcpyDeviceToHost);

    int *pad_indices_host = (int *)malloc(seq_len * sizeof(int));
    int pad_count = 0;
    for (int i = 0; i < seq_len; i++) {
        if (h_ids[i] == cfg->audio_pad_id) {
            pad_indices_host[pad_count++] = i;
        }
    }
    fprintf(stderr, "[align] Found %d audio pad positions\n", pad_count);

    if (pad_count > 0) {
        if (pad_count > a->pad_indices_capacity) {
            if (a->pad_indices_device_buf) cudaFree(a->pad_indices_device_buf);
            cudaMalloc(&a->pad_indices_device_buf, pad_count * sizeof(int));
            a->pad_indices_capacity = pad_count;
        }
        cudaMemcpy(a->pad_indices_device_buf, pad_indices_host, pad_count * sizeof(int), cudaMemcpyHostToDevice);

    cuda_scatter_copy((uint16_t *)a->input_embeds,
                          (const uint16_t *)a->audio_embeds,
                          a->pad_indices_device_buf, pad_count, cfg->dec_hidden, stream);
    }

    free(h_ids);
    free(pad_indices_host);

    /* 7. Text decoder forward (prefill only) */
    fprintf(stderr, "[align] Running text decoder\n");
    cuda_text_decoder_forward(
        a->logits, a->input_embeds, a->input_ids_buf,
        seq_len, &a->model.ws, cfg, a->model.cublas, stream
    );
    cudaEventRecord(a->ev_dec_done, stream);

    /* 8. Extract timestamps */
    int *timestamp_ms_host = (int *)malloc(MAX_WORDS * 2 * sizeof(int));
    int num_timestamps = extract_timestamps(
        a->logits, a->input_ids_buf, seq_len,
        cfg->timestamp_id, cfg->classify_num,
        timestamp_ms_host, MAX_WORDS * 2, stream
    );
    if (num_timestamps > 0) {
        fix_timestamps_lis(timestamp_ms_host, num_timestamps);
    }
    fprintf(stderr, "[align] Extracted %d timestamps (expected %d)\n",
            num_timestamps, num_timestamps_expected);

    int result_count = 0;

    {
        int out_count = 0;
        int paired_words = num_timestamps / 2;
        if (paired_words > num_words) {
            paired_words = num_words;
        }
        for (int i = 0; i < paired_words && out_count < max_words; i++) {
            strncpy(out_words[out_count].text, words[i].c_str(), sizeof(out_words[out_count].text) - 1);
            out_words[out_count].text[sizeof(out_words[out_count].text) - 1] = '\0';
            out_words[out_count].start_time = timestamp_ms_host[i * 2] / 1000.0;
            out_words[out_count].end_time = timestamp_ms_host[i * 2 + 1] / 1000.0;
            out_count++;
        }

        result_count = out_count;
    }

    cudaEventRecord(a->ev_end, stream);
    cudaEventSynchronize(a->ev_end);

    float ms_total, ms_mel, ms_enc, ms_dec;
    cudaEventElapsedTime(&ms_total, a->ev_start, a->ev_end);
    cudaEventElapsedTime(&ms_mel, a->ev_start, a->ev_mel_done);
    cudaEventElapsedTime(&ms_enc, a->ev_mel_done, a->ev_enc_done);
    cudaEventElapsedTime(&ms_dec, a->ev_enc_done, a->ev_dec_done);

    fprintf(stderr, "[bench] Total: %.1f ms  Mel: %.1f ms  Encoder: %.1f ms  Decoder: %.1f ms\n",
            ms_total, ms_mel, ms_enc, ms_dec);

    free(timestamp_ms_host);

    fprintf(stderr, "[align] Done: %d tokens aligned\n", result_count);
    return result_count;
}

int aligner_align_with_features(Aligner *a, const char *features_path,
                                int num_audio_tokens,
                                const char *text, const char *language,
                                AlignWord *out_words, int max_words) {
    if (!a->initialized) return -1;
    AlignerConfig *cfg = &a->model.cfg;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Load pre-computed audio features from binary file */
    fprintf(stderr, "[align] Loading audio features from %s (%d tokens)\n", features_path, num_audio_tokens);
    FILE *f = fopen(features_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: Cannot open %s\n", features_path); cudaStreamDestroy(stream); return -1; }
    size_t feat_bytes = (size_t)num_audio_tokens * cfg->dec_hidden * sizeof(bf16_t);
    void *feat_host = malloc(feat_bytes);
    if (fread(feat_host, 1, feat_bytes, f) != feat_bytes) {
        fprintf(stderr, "ERROR: Short read from %s\n", features_path);
        free(feat_host); fclose(f); cudaStreamDestroy(stream); return -1;
    }
    fclose(f);
    cudaMemcpy(a->audio_embeds, feat_host, feat_bytes, cudaMemcpyHostToDevice);
    free(feat_host);

    /* Word tokenization */
    std::vector<std::string> words = tokenize_space_lang(text);
    int num_words = (int)words.size();
    fprintf(stderr, "[align] Word tokens: %d\n", num_words);

    /* Build input_ids */
    int input_ids_host[MAX_PROMPT_IDS];
    int pos = 0;
    input_ids_host[pos++] = cfg->audio_start_id;
    for (int i = 0; i < num_audio_tokens; i++) input_ids_host[pos++] = cfg->audio_pad_id;
    input_ids_host[pos++] = cfg->audio_end_id;

    int total_bpe = 0;
    for (int w = 0; w < num_words; w++) {
        int wbpe[MAX_PROMPT_IDS];
        int nc = tokenizer_encode(&a->tokenizer, words[w].c_str(), wbpe, MAX_PROMPT_IDS);
        for (int t = 0; t < nc; t++) input_ids_host[pos++] = wbpe[t];
        input_ids_host[pos++] = cfg->timestamp_id;
        input_ids_host[pos++] = cfg->timestamp_id;
        total_bpe += nc;
    }
    int seq_len = pos;
    fprintf(stderr, "[align] Sequence length: %d (BPE: %d, audio: %d)\n", seq_len, total_bpe, num_audio_tokens);

    cudaMemcpy(a->input_ids_buf, input_ids_host, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    /* Embeddings */
    Tensor *embed_w = ws_get(&a->model.ws, "thinker.model.embed_tokens.weight");
    cuda_embedding_lookup((uint16_t *)a->input_embeds, (const uint16_t *)embed_w->data,
                          a->input_ids_buf, seq_len, cfg->dec_hidden, stream);

    /* Scatter audio embeddings */
    int *pad_idx = (int *)malloc(seq_len * sizeof(int));
    int pad_count = 0;
    for (int i = 0; i < seq_len; i++) {
        if (input_ids_host[i] == cfg->audio_pad_id) pad_idx[pad_count++] = i;
    }
    fprintf(stderr, "[align] Audio pad positions: %d\n", pad_count);
    if (pad_count > 0) {
        int *d_idx; cudaMalloc(&d_idx, pad_count * sizeof(int));
        cudaMemcpy(d_idx, pad_idx, pad_count * sizeof(int), cudaMemcpyHostToDevice);
        cuda_scatter_copy((uint16_t *)a->input_embeds, (const uint16_t *)a->audio_embeds,
                          d_idx, pad_count, cfg->dec_hidden, stream);
        cudaFree(d_idx);
    }
    free(pad_idx);

    /* Decoder */
    fprintf(stderr, "[align] Running text decoder\n");
    cuda_text_decoder_forward(a->logits, a->input_embeds, a->input_ids_buf,
                              seq_len, &a->model.ws, cfg, a->model.cublas, stream);

    /* Timestamps */
    int *ts_ms = (int *)malloc(MAX_WORDS * 2 * sizeof(int));
    int n_ts = extract_timestamps(a->logits, a->input_ids_buf, seq_len,
                                  cfg->timestamp_id, cfg->classify_num, ts_ms, MAX_WORDS * 2, stream);
    if (n_ts > 0) fix_timestamps_lis(ts_ms, n_ts);
    fprintf(stderr, "[align] Extracted %d timestamps (expected %d)\n", n_ts, num_words * 2);

    int result_count = 0;
    int paired = n_ts / 2;
    if (paired > num_words) paired = num_words;
    for (int i = 0; i < paired && result_count < max_words; i++) {
        strncpy(out_words[result_count].text, words[i].c_str(), 255);
        out_words[result_count].text[255] = '\0';
        out_words[result_count].start_time = ts_ms[i * 2] / 1000.0;
        out_words[result_count].end_time = ts_ms[i * 2 + 1] / 1000.0;
        result_count++;
    }

    free(ts_ms);
    cudaStreamDestroy(stream);
    fprintf(stderr, "[align] Done: %d tokens aligned\n", result_count);
    return result_count;
}

void aligner_free(Aligner *a) {
    if (!a->initialized) return;

    if (a->mel_buf) cudaFree(a->mel_buf);
    if (a->audio_embeds) cudaFree(a->audio_embeds);
    if (a->input_embeds) cudaFree(a->input_embeds);
    if (a->logits) cudaFree(a->logits);
    if (a->input_ids_buf) cudaFree(a->input_ids_buf);
    if (a->timestamp_buf) cudaFree(a->timestamp_buf);
    if (a->pcm_device_buf) cudaFree(a->pcm_device_buf);
    if (a->pad_indices_device_buf) cudaFree(a->pad_indices_device_buf);
    if (a->events_created) {
        cudaEventDestroy(a->ev_start);
        cudaEventDestroy(a->ev_mel_done);
        cudaEventDestroy(a->ev_enc_done);
        cudaEventDestroy(a->ev_dec_done);
        cudaEventDestroy(a->ev_end);
    }
    if (a->stream_created) cudaStreamDestroy(a->stream);
    if (a->pcm_buf) free(a->pcm_buf);

    aligner_model_free(&a->model);
    tokenizer_free(&a->tokenizer);

    a->initialized = 0;
}
