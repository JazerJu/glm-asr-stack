#ifndef FA_TOKENIZER_H
#define FA_TOKENIZER_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_TOKENS 8192
#define MAX_VOCAB  200000
#define MAX_MERGES 300000

typedef struct {
    /* vocab: token_string -> token_id */
    char  **vocab_str;     /* [vocab_size] allocated strings */
    int    *vocab_id;      /* [vocab_size] parallel array of ids */
    int     vocab_count;

    /* merges: merge pairs */
    char   (*merge_left)[256];   /* [merge_count] */
    char   (*merge_right)[256];  /* [merge_count] */
    int     merge_count;

    /* id -> string lookup for decode */
    char  **id_to_str;     /* [max_id+1] */
    int     max_id;

    /* Special token IDs */
    int audio_start_id;
    int audio_end_id;
    int audio_pad_id;
    int timestamp_id;
} Tokenizer;

/* Load tokenizer from model directory (vocab.json + merges.txt + tokenizer_config.json) */
int tokenizer_load(Tokenizer *tok, const char *model_dir);

/* Encode text string to token IDs. Returns number of tokens. */
int tokenizer_encode(const Tokenizer *tok, const char *text, int *out_ids, int max_ids);

/* Decode a single token ID to its BPE string (byte-level encoded).
 * Returns pointer to internal static string, or "?" if unknown.
 * NOT thread-safe. */
const char *tokenizer_id_to_str(const Tokenizer *tok, int token_id);

/* Decode a single token ID to a human-readable UTF-8 string.
 * Applies byte-level BPE → UTF-8 decoding. NOT thread-safe. */
const char *tokenizer_id_to_display_str(const Tokenizer *tok, int token_id);

/* Check if a token starts a new word (Ġ prefix in byte-level BPE). */
int tokenizer_token_starts_word(const Tokenizer *tok, int token_id);

/* Free tokenizer resources */
void tokenizer_free(Tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* FA_TOKENIZER_H */
