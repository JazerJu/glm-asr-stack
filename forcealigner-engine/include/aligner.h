#ifndef FA_ALIGNER_H
#define FA_ALIGNER_H

#include "types.h"
#include "model.h"
#include "tokenizer.h"
#include "audio_io.h"
#include "text_processor.h"
#include "timestamp.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    AlignerModel  model;
    Tokenizer     tokenizer;
    int           initialized;

    /* Workspace GPU buffers */
    bf16_t       *mel_buf;       /* [n_mels, max_frames] */
    bf16_t       *audio_embeds;  /* [max_source_positions, output_dim] */
    bf16_t       *input_embeds;  /* [max_seq, hidden] */
    bf16_t       *logits;        /* [max_seq, classify_num] */
    int          *input_ids_buf; /* [max_seq] */
    int          *timestamp_buf; /* [max_words * 2] */
    float        *pcm_device_buf;
    int           pcm_device_capacity;
    int          *pad_indices_device_buf;
    int           pad_indices_capacity;

    /* CPU buffers */
    float        *pcm_buf;
    int           pcm_buf_samples;

    /* Persistent CUDA stream */
    cudaStream_t  stream;
    int           stream_created;

    /* Persistent timing events */
    cudaEvent_t   ev_start;
    cudaEvent_t   ev_mel_done;
    cudaEvent_t   ev_enc_done;
    cudaEvent_t   ev_dec_done;
    cudaEvent_t   ev_end;
    int           events_created;
} Aligner;

/* Initialize aligner: load model, tokenizer, allocate workspace */
int aligner_init(Aligner *a, const char *model_dir);

/* Run forced alignment on audio + text.
 * audio_path: path to WAV/PCM file (mono 16kHz)
 * text: transcript text
 * language: "Chinese", "English", etc.
 * out_words: output aligned words (caller-allocated array)
 * max_words: max output words
 * Returns number of aligned words, or -1 on error. */
int aligner_align(Aligner *a, const char *audio_path,
                  const char *text, const char *language,
                  AlignWord *out_words, int max_words);

int aligner_align_with_features(Aligner *a, const char *features_path,
                                 int num_audio_tokens,
                                 const char *text, const char *language,
                                 AlignWord *out_words, int max_words);

void aligner_free(Aligner *a);

#ifdef __cplusplus
}
#endif

#endif /* FA_ALIGNER_H */
