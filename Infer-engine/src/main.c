#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#include "/usr/local/cuda-12.8/include/cuda_runtime.h"

#include "../include/audio_io.h"
#include "../include/engine.h"
#include "../include/types.h"

#define MAIN_MAX_RESULTS 64
#define LINE_BUF_SIZE 4096

typedef struct {
    char *path;
    int   seq_id;
    int   n_generated;
} AudioRequest;

static char *dup_cstr(const char *src);

static int set_request_text(char **texts_by_request, int index, const char *text) {
    char *copy = dup_cstr(text);

    if (!copy) return -1;

    free(texts_by_request[index]);
    texts_by_request[index] = copy;
    return 0;
}

static int find_request_index_by_seq_id(const AudioRequest *requests,
                                        int num_requests,
                                        int seq_id) {
    for (int i = 0; i < num_requests; i++) {
        if (requests[i].seq_id == seq_id) {
            return i;
        }
    }

    return -1;
}

static char *dup_cstr(const char *src) {
    size_t len;
    char *dst;

    if (!src) return NULL;

    len = strlen(src) + 1;
    dst = malloc(len);
    if (!dst) return NULL;

    memcpy(dst, src, len);
    return dst;
}

static void free_text_array(char **texts, int count) {
    if (!texts) return;

    for (int i = 0; i < count; i++) {
        free(texts[i]);
    }
}

static void free_path_array(char **paths, int count) {
    if (!paths) return;

    for (int i = 0; i < count; i++) {
        free(paths[i]);
        paths[i] = NULL;
    }
}

static int process_audio_paths_continuously(Engine *eng,
                                            const char **audio_paths,
                                            int num_audio_paths,
                                            const int *prompt_token_ids,
                                            int num_prompt_token_ids,
                                            AudioRequest *requests,
                                            char **texts_by_request,
                                            int *tokens_by_request) {
    int next_request = 0;
    int active_requests = 0;
    int completed_requests = 0;

    for (int i = 0; i < num_audio_paths; i++) {
        requests[i].path = (char *)audio_paths[i];
        requests[i].seq_id = -1;
        requests[i].n_generated = 0;
        if (tokens_by_request) {
            tokens_by_request[i] = 0;
        }
    }

    while (completed_requests < num_audio_paths) {
        int submitted_any = 0;

        while (next_request < num_audio_paths) {
            float *pcm = NULL;
            int num_samples = 0;
            int seq_id;

            if (load_audio_file(audio_paths[next_request], &pcm, &num_samples) != 0) {
                fprintf(stderr, "ERROR: Failed to load %s\n", audio_paths[next_request]);
                if (set_request_text(texts_by_request, next_request, "[ERROR]") != 0) {
                    fprintf(stderr, "ERROR: Failed to allocate error text\n");
                    return -1;
                }
                if (tokens_by_request) {
                    tokens_by_request[next_request] = 0;
                }
                completed_requests++;
                next_request++;
                continue;
            }

            seq_id = engine_submit_with_prompt_ids(eng, pcm, num_samples,
                                                   prompt_token_ids,
                                                   num_prompt_token_ids);
            free(pcm);

            if (seq_id < 0) {
                if (active_requests == 0) {
                    fprintf(stderr, "ERROR: Failed to submit %s\n", audio_paths[next_request]);
                    if (set_request_text(texts_by_request, next_request, "[ERROR]") != 0) {
                        fprintf(stderr, "ERROR: Failed to allocate error text\n");
                        return -1;
                    }
                    if (tokens_by_request) {
                        tokens_by_request[next_request] = 0;
                    }
                    completed_requests++;
                    next_request++;
                    continue;
                }
                break;
            }

            requests[next_request].seq_id = seq_id;
            next_request++;
            active_requests++;
            submitted_any = 1;
        }

        if (active_requests == 0) {
            if (next_request < num_audio_paths) {
                continue;
            }
            break;
        }

        if (active_requests > 0 && (next_request < num_audio_paths || !submitted_any)) {
            EngineResult results[MAIN_MAX_RESULTS];
            int n_results = 0;
            int finished = engine_step(eng, results, &n_results, MAIN_MAX_RESULTS);

            for (int i = 0; i < n_results; i++) {
                int request_index = find_request_index_by_seq_id(requests, num_audio_paths, results[i].seq_id);

                if (request_index >= 0) {
                    requests[request_index].seq_id = -1;
                    requests[request_index].n_generated = results[i].n_generated;
                    if (tokens_by_request) {
                        tokens_by_request[request_index] = results[i].n_generated;
                    }
                    free(texts_by_request[request_index]);
                    texts_by_request[request_index] = results[i].text;
                    active_requests--;
                    completed_requests++;
                } else {
                    free(results[i].text);
                }
            }
            (void)finished;
        }
    }

    for (int i = 0; i < num_audio_paths; i++) {
        if (!texts_by_request[i]) {
            if (set_request_text(texts_by_request, i, "[ERROR]") != 0) {
                fprintf(stderr, "ERROR: Failed to allocate fallback error text\n");
                return -1;
            }
            if (tokens_by_request) {
                tokens_by_request[i] = 0;
            }
        }
    }

    engine_reset_batch(eng);
    return 0;
}

static int ensure_request_capacity(AudioRequest **requests,
                                   char ***paths,
                                   int *capacity,
                                   int needed) {
    AudioRequest *new_requests;
    char **new_paths;

    if (needed <= *capacity) return 0;

    new_requests = realloc(*requests, (size_t)needed * sizeof(AudioRequest));
    if (!new_requests) return -1;
    *requests = new_requests;

    new_paths = realloc(*paths, (size_t)needed * sizeof(char *));
    if (!new_paths) return -1;
    *paths = new_paths;

    for (int i = *capacity; i < needed; i++) {
        (*requests)[i].path = NULL;
        (*requests)[i].seq_id = -1;
        (*paths)[i] = NULL;
    }

    *capacity = needed;
    return 0;
}

static int parse_prompt_ids_command(const char *line, int **out_ids, int *out_count) {
    const char *p = line;
    char *end = NULL;
    long count;
    int *ids = NULL;

    if (out_ids == NULL || out_count == NULL) {
        return -1;
    }
    *out_ids = NULL;
    *out_count = 0;

    while (*p == ' ') {
        p++;
    }
    count = strtol(p, &end, 10);
    if (end == p || count < 0 || count > 4096) {
        return -1;
    }
    p = end;
    if (count == 0) {
        return 0;
    }

    ids = (int *) malloc((size_t) count * sizeof(int));
    if (ids == NULL) {
        return -1;
    }
    for (long i = 0; i < count; i++) {
        long value;
        while (*p == ' ') {
            p++;
        }
        value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > 1000000) {
            free(ids);
            return -1;
        }
        ids[i] = (int) value;
        p = end;
    }

    *out_ids = ids;
    *out_count = (int) count;
    return 0;
}

static int load_mel_filters(Engine *eng, const char *mel_filters_path) {
    FILE *mf = fopen(mel_filters_path, "rb");
    long mf_len;
    int mel_rows;
    float *mel_host;

    if (!mf) {
        fprintf(stderr, "ERROR: Cannot open mel filters: %s\n", mel_filters_path);
        return -1;
    }

    fseek(mf, 0, SEEK_END);
    mf_len = ftell(mf);
    fseek(mf, 0, SEEK_SET);

    mel_rows = (int)(mf_len / (long)sizeof(float) / 128L);
    mel_host = malloc((size_t)mf_len);
    if (!mel_host) {
        fclose(mf);
        fprintf(stderr, "ERROR: Failed to allocate mel filter host buffer\n");
        return -1;
    }
    if (fread(mel_host, 1, (size_t)mf_len, mf) != (size_t)mf_len) {
        fclose(mf);
        free(mel_host);
        fprintf(stderr, "ERROR: Failed to read mel filters\n");
        return -1;
    }
    fclose(mf);

    cudaMalloc(&eng->mel_filters.data, (size_t)mf_len);
    cudaMemcpy(eng->mel_filters.data, mel_host, (size_t)mf_len, cudaMemcpyHostToDevice);
    free(mel_host);

    eng->mel_filters.dtype = DTYPE_FP32;
    eng->mel_filters.ndim = 2;
    eng->mel_filters.shape[0] = mel_rows;
    eng->mel_filters.shape[1] = 128;
    eng->mel_filters.nbytes = (size_t)mf_len;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  One-shot CLI mode (backward compatibility)                        */
/* ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  Daemon mode (stdin/stdout):\n");
    fprintf(stderr, "    %s --daemon\n", prog);
    fprintf(stderr, "  One-shot mode (CLI):\n");
    fprintf(stderr, "    %s --model DIR --audio FILE [--audio FILE ...] --mel-filters FILE\n", prog);
    exit(1);
}

static int run_oneshot(int argc, char *argv[]) {
    const char *model_dir = NULL;
    const char *mel_filters_path = NULL;
    const char **audio_paths = NULL;
    int num_audio_paths = 0;
    int audio_capacity = 0;
    Engine *eng = NULL;

    eng = calloc(1, sizeof(*eng));
    if (!eng) {
        fprintf(stderr, "ERROR: Failed to allocate engine\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            if (num_audio_paths == audio_capacity) {
                int new_capacity = audio_capacity ? audio_capacity * 2 : 4;
                const char **new_audio_paths = realloc(audio_paths, (size_t)new_capacity * sizeof(char *));
                if (!new_audio_paths) {
                    free(audio_paths);
                    fprintf(stderr, "ERROR: Failed to grow audio path list\n");
                    return 1;
                }
                audio_paths = new_audio_paths;
                audio_capacity = new_capacity;
            }
            audio_paths[num_audio_paths++] = argv[++i];
        } else if (strcmp(argv[i], "--mel-filters") == 0 && i + 1 < argc) {
            mel_filters_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            free(audio_paths);
            usage(argv[0]);
        }
    }

    if (!model_dir || num_audio_paths == 0 || !mel_filters_path) {
        free(audio_paths);
        usage(argv[0]);
    }

    if (engine_init(eng, model_dir) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize engine\n");
        free(audio_paths);
        free(eng);
        return 1;
    }

    if (load_mel_filters(eng, mel_filters_path) != 0) {
        engine_free(eng);
        free(eng);
        free(audio_paths);
        return 1;
    }
    if (engine_prepare_runtime_kv_pool(eng) != 0) {
        fprintf(stderr, "ERROR: engine runtime KV preparation failed\n");
        engine_free(eng);
        free(eng);
        free(audio_paths);
        return 1;
    }

    {
        struct timeval t_start, t_end;
        double infer_ms;
        AudioRequest *requests = calloc((size_t)num_audio_paths, sizeof(AudioRequest));
        char **texts_by_request = calloc((size_t)num_audio_paths, sizeof(char *));
        int *tokens_by_request = calloc((size_t)num_audio_paths, sizeof(int));

        if (!requests || !texts_by_request || !tokens_by_request) {
            fprintf(stderr, "ERROR: Failed to allocate multi-audio request state\n");
            free(requests);
            free(texts_by_request);
            free(tokens_by_request);
            engine_free(eng);
            free(eng);
            free(audio_paths);
            return 1;
        }

        gettimeofday(&t_start, NULL);

        if (process_audio_paths_continuously(eng, audio_paths, num_audio_paths,
                                             NULL, 0,
                                             requests, texts_by_request, tokens_by_request) != 0) {
            free_text_array(texts_by_request, num_audio_paths);
            free(requests);
            free(texts_by_request);
            free(tokens_by_request);
            engine_free(eng);
            free(eng);
            free(audio_paths);
            return 1;
        }

        engine_profile_report_run(eng, "oneshot_run");
        engine_profile_reset_run(eng);

        gettimeofday(&t_end, NULL);
        infer_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                   (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        fprintf(stderr, "INFER_TIME: %.1f ms (%d samples, %.2f ms/sample)\n",
                infer_ms, num_audio_paths, infer_ms / num_audio_paths);
        {
            int total_generated = 0;
            for (int i = 0; i < num_audio_paths; i++) {
                total_generated += tokens_by_request[i];
            }
            fprintf(stderr, "OUTPUT_TOKENS: %d (%.2f tok/s, %.2f tok/sample)\n",
                    total_generated,
                    infer_ms > 0.0 ? (double)total_generated * 1000.0 / infer_ms : 0.0,
                    num_audio_paths > 0 ? (double)total_generated / (double)num_audio_paths : 0.0);
        }

        for (int i = 0; i < num_audio_paths; i++) {
            const char *text = texts_by_request[i] ? texts_by_request[i] : "[ERROR]";
            printf("%s: %s\n", requests[i].path, text);
        }

        free_text_array(texts_by_request, num_audio_paths);
        free(requests);
        free(texts_by_request);
        free(tokens_by_request);
    }

    engine_free(eng);
    free(eng);
    free(audio_paths);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Daemon mode: load once, process batches via stdin/stdout          */
/*                                                                    */
/*  Protocol:                                                         */
/*    stdin  → LOAD <model_dir>                                       */
/*    stdin  → MEL <mel_filters_path>                                 */
/*    stdin  → PROMPT_IDS <n> <id0> ... <idN-1>  (optional)           */
/*    stdin  → BATCH <n_samples>                                      */
/*    stdin  → <path_0>  (one per line)                               */
/*    stdin  → ...                                                    */
/*    stdin  → RUN                                                    */
/*    stdout ← <path>: <text>  (one per sample, submission order)    */
/*    stdout ← INFER_TIME: xxx ms (n samples, xxx ms/sample)         */
/*    stdout ← READY                                                  */
/*    stdin  → QUIT                                                   */
/*                                                                    */
/*  Errors go to stderr, followed by READY on stdout.                 */
/* ------------------------------------------------------------------ */

static char *read_line(void) {
    static char buf[LINE_BUF_SIZE];
    if (!fgets(buf, sizeof(buf), stdin)) return NULL;
    /* strip trailing newline */
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
    if (len > 1 && buf[len - 2] == '\r') buf[len - 2] = '\0';
    return buf;
}

static int run_daemon(void) {
    Engine *eng = NULL;
    int engine_loaded = 0;
    char *line;

    AudioRequest *requests = NULL;
    char **paths = NULL;
    int batch_capacity = 0;
    int batch_count = 0;
    int stored_paths = 0;
    int *batch_prompt_ids = NULL;
    int batch_prompt_len = 0;

    eng = calloc(1, sizeof(*eng));
    if (!eng) {
        fprintf(stderr, "ERROR: Failed to allocate engine\n");
        return 1;
    }

    /* Flush stdout immediately for line-based protocol */
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    while ((line = read_line()) != NULL) {
        /* Skip empty lines */
        if (line[0] == '\0') continue;

        if (strncmp(line, "LOAD ", 5) == 0) {
            const char *model_dir = line + 5;
            if (engine_loaded) {
                engine_free(eng);
                memset(eng, 0, sizeof(*eng));
                engine_loaded = 0;
            }
            if (engine_init(eng, model_dir) != 0) {
                fprintf(stderr, "ERROR: engine_init failed for %s\n", model_dir);
                printf("READY\n");
                continue;
            }
            engine_loaded = 1;
            printf("READY\n");

        } else if (strncmp(line, "MEL ", 4) == 0) {
            if (!engine_loaded) {
                fprintf(stderr, "ERROR: LOAD must be called before MEL\n");
                printf("READY\n");
                continue;
            }
            if (load_mel_filters(eng, line + 4) != 0) {
                fprintf(stderr, "ERROR: load_mel_filters failed\n");
                printf("READY\n");
                continue;
            }
            if (engine_prepare_runtime_kv_pool(eng) != 0) {
                fprintf(stderr, "ERROR: engine runtime KV preparation failed\n");
                printf("READY\n");
                continue;
            }
            printf("READY\n");

        } else if (strncmp(line, "PROMPT_IDS ", 11) == 0) {
            int *new_prompt_ids = NULL;
            int new_prompt_len = 0;
            if (parse_prompt_ids_command(line + 11, &new_prompt_ids, &new_prompt_len) != 0) {
                fprintf(stderr, "ERROR: invalid PROMPT_IDS command\n");
                printf("READY\n");
                continue;
            }
            free(batch_prompt_ids);
            batch_prompt_ids = new_prompt_ids;
            batch_prompt_len = new_prompt_len;
            printf("READY\n");

        } else if (strncmp(line, "BATCH ", 6) == 0) {
            batch_count = atoi(line + 6);
            if (batch_count <= 0) {
                fprintf(stderr, "ERROR: BATCH count %d out of range [1, inf)\n",
                        batch_count);
                batch_count = 0;
                printf("READY\n");
                continue;
            }

            if (ensure_request_capacity(&requests, &paths, &batch_capacity, batch_count) != 0) {
                fprintf(stderr, "ERROR: Failed to grow batch storage to %d items\n", batch_count);
                batch_count = 0;
                printf("READY\n");
                continue;
            }

            free_path_array(paths, stored_paths);
            stored_paths = 0;

            /* Read exactly batch_count path lines */
            for (int i = 0; i < batch_count; i++) {
                char *path_line = read_line();
                if (!path_line) {
                    fprintf(stderr, "ERROR: EOF while reading batch paths\n");
                    free_path_array(paths, stored_paths);
                    stored_paths = 0;
                    batch_count = 0;
                    printf("READY\n");
                    goto next_cmd;
                }
                paths[i] = dup_cstr(path_line);
                if (!paths[i]) {
                    fprintf(stderr, "ERROR: Failed to duplicate path\n");
                    free_path_array(paths, stored_paths);
                    stored_paths = 0;
                    batch_count = 0;
                    printf("READY\n");
                    goto next_cmd;
                }
                stored_paths++;
            }
            /* batch_count paths stored, waiting for RUN */
            continue;

        } else if (strcmp(line, "RUN") == 0) {
            if (!engine_loaded || batch_count == 0) {
                fprintf(stderr, "ERROR: LOAD + BATCH required before RUN\n");
                printf("READY\n");
                continue;
            }

            {
                struct timeval t_start, t_end;
                double infer_ms;
                char **texts_by_request = calloc((size_t)batch_count, sizeof(char *));
                int *tokens_by_request = calloc((size_t)batch_count, sizeof(int));
                const char **input_paths = (const char **)paths;

                if (!texts_by_request || !tokens_by_request) {
                    fprintf(stderr, "ERROR: alloc failed\n");
                    free(texts_by_request);
                    free(tokens_by_request);
                    printf("READY\n");
                    batch_count = 0;
                    goto next_cmd;
                }

                gettimeofday(&t_start, NULL);

                if (process_audio_paths_continuously(eng, input_paths, batch_count,
                                                     batch_prompt_ids, batch_prompt_len,
                                                     requests, texts_by_request, tokens_by_request) != 0) {
                    free_text_array(texts_by_request, batch_count);
                    free(texts_by_request);
                    free(tokens_by_request);
                    printf("READY\n");
                    batch_count = 0;
                    goto next_cmd;
                }

                engine_profile_report_run(eng, "daemon_run");
                engine_profile_reset_run(eng);

                gettimeofday(&t_end, NULL);
                infer_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                           (t_end.tv_usec - t_start.tv_usec) / 1000.0;

                /* Output results in submission order */
                for (int i = 0; i < batch_count; i++) {
                    const char *text = texts_by_request[i] ? texts_by_request[i] : "[ERROR]";
                    printf("%s: %s\n", requests[i].path, text);
                }

                printf("INFER_TIME: %.1f ms (%d samples, %.2f ms/sample)\n",
                       infer_ms, batch_count, infer_ms / batch_count);
                {
                    int total_generated = 0;
                    for (int i = 0; i < batch_count; i++) {
                        total_generated += tokens_by_request[i];
                    }
                    printf("OUTPUT_TOKENS: %d (%.2f tok/s, %.2f tok/sample)\n",
                           total_generated,
                           infer_ms > 0.0 ? (double)total_generated * 1000.0 / infer_ms : 0.0,
                           batch_count > 0 ? (double)total_generated / (double)batch_count : 0.0);
                }
                printf("READY\n");

                free_text_array(texts_by_request, batch_count);
                free(texts_by_request);
                free(tokens_by_request);
            }

            free_path_array(paths, stored_paths);
            stored_paths = 0;
            batch_count = 0;

        } else if (strcmp(line, "QUIT") == 0) {
            break;

        } else {
            fprintf(stderr, "ERROR: Unknown command: %s\n", line);
            printf("READY\n");
        }

next_cmd:
        continue;
    }

    /* Cleanup */
    if (engine_loaded) engine_free(eng);
    free_path_array(paths, stored_paths);
    free(batch_prompt_ids);
    free(requests);
    free(paths);
    free(eng);

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Entry point                                                       */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--daemon") == 0) {
            return run_daemon();
        }
    }

    /* One-shot mode (CLI, default) */
    return run_oneshot(argc, argv);
}
