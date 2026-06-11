#ifndef GLMASR_SEQUENCE_H
#define GLMASR_SEQUENCE_H

#include <stdint.h>
#include "types.h"
#include "kv_pool.h"

typedef enum {
    SEQ_WAITING,
    SEQ_RUNNING,
    SEQ_FINISHED
} SeqStatus;

typedef struct {
    int        seq_id;
    int        slot_idx;
    SeqStatus  status;

    float     *pcm;
    int        num_samples;
    int        estimated_audio_tokens;
    int        estimated_prompt_tokens;
    int        estimated_total_tokens;
    int        estimated_total_blocks;
    int       *user_prompt_token_ids;
    int        num_user_prompt_tokens;

    /* Prompt: text tokens + audio placeholder + audio embeds */
    int       *prompt_token_ids;   /* host: full prompt (text + pad placeholders) */
    int        num_prompt_tokens;
    int        num_audio_tokens;   /* audio tokens injected at positions [3, 3+N) */
    bf16_t    *audio_embeds;       /* GPU: [num_audio_tokens, dec_hidden] bf16 */

    /* Generation state */
    int        total_seq_len;      /* prompt_len + generated so far */
    int        max_new_tokens;
    int       *generated_ids;      /* host: generated token IDs */
    int        n_generated;
    int        eos_reason;         /* 0=running, 1=eos_hit, 2=max_tokens */

    /* KV blocks */
    BlockTable bt;                 /* block table in the KV pool */
    int        seq_idx;            /* which row in pool's block_tables */

    /* Result */
    char      *output_text;        /* decoded UTF-8 text (set when FINISHED) */
} Sequence;

void seq_init(Sequence *s, int seq_id, int slot_idx, int max_new_tokens);

int seq_set_audio(Sequence *s, const float *pcm, int num_samples,
                  int estimated_audio_tokens, int estimated_prompt_tokens,
                  int estimated_total_tokens, int estimated_total_blocks);

int seq_set_user_prompt(Sequence *s, const int *token_ids, int num_tokens);

void seq_set_prompt(Sequence *s, const int *token_ids, int num_prompt,
                     int num_audio_tokens, bf16_t *audio_embeds_gpu);

void seq_free(Sequence *s);

#endif
