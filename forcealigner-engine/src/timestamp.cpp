#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "timestamp.h"
#include "cuda_kernels.h"

static int *g_d_logit_rows = NULL;
static int *g_d_argmax = NULL;
static int g_timestamp_capacity = 0;

static int ensure_timestamp_buffers(int count) {
    if (count <= g_timestamp_capacity) {
        return 0;
    }
    if (g_d_logit_rows) {
        cudaFree(g_d_logit_rows);
        g_d_logit_rows = NULL;
    }
    if (g_d_argmax) {
        cudaFree(g_d_argmax);
        g_d_argmax = NULL;
    }
    if (cudaMalloc(&g_d_logit_rows, (size_t)count * sizeof(int)) != cudaSuccess) {
        return -1;
    }
    if (cudaMalloc(&g_d_argmax, (size_t)count * sizeof(int)) != cudaSuccess) {
        cudaFree(g_d_logit_rows);
        g_d_logit_rows = NULL;
        return -1;
    }
    g_timestamp_capacity = count;
    return 0;
}

int extract_timestamps(const bf16_t *logits, const int *input_ids,
                      int seq_len, int timestamp_id, int classify_num,
                      int *out_ms, int max_timestamps, void *stream) {
    cudaStream_t cuda_stream = (cudaStream_t)stream;

    if (!logits || !input_ids || !out_ms || seq_len <= 0 || classify_num <= 0 || max_timestamps <= 0) {
        return 0;
    }

    /* Find the <timestamp> token positions in the sequence.
       The LM head predicts the timestamp class from the hidden state
       at the <timestamp> token position itself. */
    int *h_input_ids = (int *)malloc(seq_len * sizeof(int));
    cudaMemcpy(h_input_ids, input_ids, seq_len * sizeof(int), cudaMemcpyDeviceToHost);

    int ts_count = 0;
    int *h_logit_rows = (int *)malloc(seq_len * sizeof(int));
    for (int i = 1; i < seq_len && ts_count < max_timestamps; i++) {
        if (h_input_ids[i] == timestamp_id) {
            h_logit_rows[ts_count++] = i;
        }
    }

    if (ts_count > 0) {
        if (ensure_timestamp_buffers(ts_count) != 0) {
            free(h_input_ids);
            free(h_logit_rows);
            return 0;
        }

        cudaMemcpyAsync(g_d_logit_rows, h_logit_rows, ts_count * sizeof(int),
                        cudaMemcpyHostToDevice, cuda_stream);
        cuda_argmax_indexed_last_dim(g_d_argmax, (const uint16_t *)logits,
                                     g_d_logit_rows, ts_count, classify_num, stream);

        /* Copy results to host and convert to ms */
        int *h_argmax = (int *)malloc(ts_count * sizeof(int));
        cudaMemcpyAsync(h_argmax, g_d_argmax, ts_count * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        for (int i = 0; i < ts_count; i++) {
            out_ms[i] = h_argmax[i] * 80;
        }

        free(h_argmax);
    }

    free(h_input_ids);
    free(h_logit_rows);

    return ts_count;
}

int fix_timestamps_lis(int *data, int n) {
    if (n == 0) return 0;

    int *dp = (int *)malloc(n * sizeof(int));
    int *parent = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) { dp[i] = 1; parent[i] = -1; }

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    int max_len = dp[0], max_idx = 0;
    for (int i = 1; i < n; i++) {
        if (dp[i] > max_len) { max_len = dp[i]; max_idx = i; }
    }

    int *lis = (int *)malloc(n * sizeof(int));
    int lis_count = 0;
    int idx = max_idx;
    while (idx != -1) { lis[lis_count++] = idx; idx = parent[idx]; }

    int *is_normal = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < lis_count; i++) is_normal[lis[i]] = 1;

    /* Reverse LIS to get forward order */
    for (int i = 0; i < lis_count / 2; i++) {
        int tmp = lis[i]; lis[i] = lis[lis_count - 1 - i]; lis[lis_count - 1 - i] = tmp;
    }

    int i = 0;
    while (i < n) {
        if (!is_normal[i]) {
            int j = i;
            while (j < n && !is_normal[j]) j++;
            int anomaly_count = j - i;

            int left_val = -1;
            for (int k = i - 1; k >= 0; k--) {
                if (is_normal[k]) { left_val = data[k]; break; }
            }
            int right_val = -1;
            for (int k = j; k < n; k++) {
                if (is_normal[k]) { right_val = data[k]; break; }
            }

            if (anomaly_count <= 2) {
                for (int k = i; k < j; k++) {
                    if (left_val < 0) data[k] = right_val;
                    else if (right_val < 0) data[k] = left_val;
                    else data[k] = ((k - (i - 1)) <= (j - k)) ? left_val : right_val;
                }
            } else {
                if (left_val >= 0 && right_val >= 0) {
                    double step = (double)(right_val - left_val) / (anomaly_count + 1);
                    for (int k = i; k < j; k++) {
                        data[k] = (int)(left_val + step * (k - i + 1));
                    }
                } else if (left_val >= 0) {
                    for (int k = i; k < j; k++) data[k] = left_val;
                } else if (right_val >= 0) {
                    for (int k = i; k < j; k++) data[k] = right_val;
                }
            }
            i = j;
        } else {
            i++;
        }
    }

    free(dp); free(parent); free(lis); free(is_normal);
    return 0;
}

int timestamps_to_words(const int *timestamps, int num_timestamps,
                        char words[][256], int num_words,
                        int timestamp_segment_ms,
                        AlignWord *out_words, int max_words) {
    (void)timestamp_segment_ms;
    int count = 0;
    for (int i = 0; i < num_words && count < max_words; i++) {
        if (2 * i + 1 >= num_timestamps) break;
        strncpy(out_words[count].text, words[i], 255);
        out_words[count].text[255] = '\0';
        out_words[count].start_time = timestamps[2 * i] / 1000.0;
        out_words[count].end_time = timestamps[2 * i + 1] / 1000.0;
        count++;
    }
    return count;
}

void print_aligned_words(const AlignWord *words, int count) {
    for (int i = 0; i < count; i++) {
        printf("%s\t%.3f\t%.3f\n", words[i].text, words[i].start_time, words[i].end_time);
    }
}

int write_srt(const AlignWord *words, int count, const char *output_path) {
    FILE *f = fopen(output_path, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open output file: %s\n", output_path);
        return -1;
    }

    int srt_idx = 1;
    /* Group words into subtitle segments (simple: every ~5 words or at sentence boundaries) */
    int i = 0;
    while (i < count) {
        int seg_start = i;
        int seg_end = i + 5;
        if (seg_end > count) seg_end = count;

        fprintf(f, "%d\n", srt_idx++);
        int h1 = (int)(words[seg_start].start_time / 3600);
        int m1 = (int)((words[seg_start].start_time - h1 * 3600) / 60);
        int s1 = (int)(words[seg_start].start_time) % 60;
        int ms1 = (int)((words[seg_start].start_time - (int)words[seg_start].start_time) * 1000);

        int h2 = (int)(words[seg_end - 1].end_time / 3600);
        int m2 = (int)((words[seg_end - 1].end_time - h2 * 3600) / 60);
        int s2 = (int)(words[seg_end - 1].end_time) % 60;
        int ms2 = (int)((words[seg_end - 1].end_time - (int)words[seg_end - 1].end_time) * 1000);

        fprintf(f, "%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n",
                h1, m1, s1, ms1, h2, m2, s2, ms2);

        for (int j = seg_start; j < seg_end; j++) {
            fprintf(f, "%s", words[j].text);
        }
        fprintf(f, "\n\n");

        i = seg_end;
    }

    fclose(f);
    return 0;
}
