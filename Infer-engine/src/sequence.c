#include "../include/sequence.h"

#include <stdlib.h>
#include <string.h>

void seq_init(Sequence *s, int seq_id, int slot_idx, int max_new_tokens) {
    if (s == NULL) {
        return;
    }

    memset(s, 0, sizeof(*s));
    s->seq_id = seq_id;
    s->slot_idx = slot_idx;
    s->seq_idx = slot_idx;
    s->max_new_tokens = max_new_tokens;
    s->status = SEQ_WAITING;
    s->eos_reason = 0;
}

int seq_set_audio(Sequence *s, const float *pcm, int num_samples,
                  int estimated_audio_tokens, int estimated_prompt_tokens,
                  int estimated_total_tokens, int estimated_total_blocks) {
    size_t pcm_bytes;

    if (s == NULL || pcm == NULL || num_samples <= 0) {
        return -1;
    }

    free(s->pcm);
    s->pcm = NULL;

    pcm_bytes = (size_t) num_samples * sizeof(float);
    s->pcm = (float *) malloc(pcm_bytes);
    if (s->pcm == NULL) {
        return -1;
    }

    memcpy(s->pcm, pcm, pcm_bytes);
    s->num_samples = num_samples;
    s->estimated_audio_tokens = estimated_audio_tokens;
    s->estimated_prompt_tokens = estimated_prompt_tokens;
    s->estimated_total_tokens = estimated_total_tokens;
    s->estimated_total_blocks = estimated_total_blocks;
    return 0;
}

int seq_set_user_prompt(Sequence *s, const int *token_ids, int num_tokens) {
    if (s == NULL) {
        return -1;
    }

    free(s->user_prompt_token_ids);
    s->user_prompt_token_ids = NULL;
    s->num_user_prompt_tokens = 0;

    if (token_ids == NULL || num_tokens <= 0) {
        return 0;
    }

    s->user_prompt_token_ids = (int *) malloc((size_t) num_tokens * sizeof(int));
    if (s->user_prompt_token_ids == NULL) {
        return -1;
    }

    memcpy(s->user_prompt_token_ids, token_ids, (size_t) num_tokens * sizeof(int));
    s->num_user_prompt_tokens = num_tokens;
    return 0;
}

void seq_set_prompt(Sequence *s, const int *token_ids, int num_prompt,
                    int num_audio_tokens, bf16_t *audio_embeds_gpu) {
    if (s == NULL) {
        return;
    }

    free(s->prompt_token_ids);
    s->prompt_token_ids = NULL;

    if (token_ids != NULL && num_prompt > 0) {
        s->prompt_token_ids = (int *) malloc((size_t) num_prompt * sizeof(int));
        if (s->prompt_token_ids != NULL) {
            memcpy(s->prompt_token_ids, token_ids, (size_t) num_prompt * sizeof(int));
        }
    }

    s->num_prompt_tokens = num_prompt;
    s->num_audio_tokens = num_audio_tokens;
    s->audio_embeds = audio_embeds_gpu;
    s->n_generated = 0;
    s->eos_reason = 0;
}

void seq_free(Sequence *s) {
    if (s == NULL) {
        return;
    }

    free(s->user_prompt_token_ids);
    free(s->prompt_token_ids);
    free(s->generated_ids);
    free(s->output_text);
    free(s->pcm);
    free(s->bt.h_table);

    memset(s, 0, sizeof(*s));
}
