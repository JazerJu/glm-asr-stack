#ifndef FA_TIMESTAMP_H
#define FA_TIMESTAMP_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Extract timestamps from model logits at timestamp token positions.
 * logits: [seq_len, classify_num] bf16 on GPU
 * input_ids: [seq_len] int32 on GPU
 * Returns number of timestamps extracted (should be 2 * num_words). */
int extract_timestamps(const bf16_t *logits, const int *input_ids,
                      int seq_len, int timestamp_id, int classify_num,
                      int *out_ms, int max_timestamps, void *stream);

/* Fix non-monotonic timestamps using LIS-based algorithm.
 * Modifies timestamps in-place. Returns 0 on success. */
int fix_timestamps_lis(int *timestamps, int count);

/* Convert raw timestamp pairs to word-level results.
 * timestamps: [2 * num_words] array of ms values
 * words: [num_words] array of word strings
 * Returns number of aligned words. */
int timestamps_to_words(const int *timestamps, int num_timestamps,
                        char words[][256], int num_words,
                        int timestamp_segment_ms,
                        AlignWord *out_words, int max_words);

/* Write aligned words to SRT format file.
 * Returns 0 on success. */
int write_srt(const AlignWord *words, int count, const char *output_path);

/* Print aligned words to stdout in tab-separated format. */
void print_aligned_words(const AlignWord *words, int count);

#ifdef __cplusplus
}
#endif

#endif /* FA_TIMESTAMP_H */
