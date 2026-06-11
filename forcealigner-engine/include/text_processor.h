#ifndef FA_TEXT_PROCESSOR_H
#define FA_TEXT_PROCESSOR_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_WORDS 4096
#define MAX_PROMPT_IDS 16384

/* Tokenize text into alignment units based on language.
 * For Chinese/CJK: each character is a unit.
 * For space-delimited languages: split on spaces, then split CJK chars.
 * Returns number of words, fills word_list. */
int tokenize_for_align(const char *text, const char *language,
                       char words[][256], int max_words);

/* Build the alignment prompt input_ids.
 * Format: [audio_start] [audio_pad × num_audio_tokens] [audio_end]
 *         + for each word: word_tokens... [timestamp]
 *         + final [timestamp]
 * Returns total sequence length. */
int build_alignment_prompt(const int *word_token_ids, const int *word_token_counts,
                          int num_words, int num_audio_tokens,
                          const AlignerConfig *cfg,
                          int *out_ids, int max_ids);

#ifdef __cplusplus
}
#endif

#endif /* FA_TEXT_PROCESSOR_H */
