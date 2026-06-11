#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "aligner.h"

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model DIR [--audio FILE --text \"TEXT\" --lang LANGUAGE [--output FILE] | --daemon]\n", prog);
    fprintf(stderr, "  --model   Path to ForceAligner model directory\n");
    fprintf(stderr, "  --audio   Path to audio file (WAV, mono 16kHz)\n");
    fprintf(stderr, "  --text    Transcript text for alignment\n");
    fprintf(stderr, "  --lang    Language: Chinese, English, Japanese, Korean, etc.\n");
    fprintf(stderr, "  --output  Output SRT file path (optional, default: stdout)\n");
    fprintf(stderr, "  --daemon  Read AUDIO<TAB>TEXT<TAB>LANGUAGE requests from stdin\n");
}

static void strip_newline(char *s) {
    if (!s) return;
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static int run_oneshot(const char *model_dir, const char *audio_path,
                       const char *text, const char *language,
                       const char *output_path) {
    Aligner aligner;
    if (aligner_init(&aligner, model_dir) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize aligner\n");
        return 1;
    }

    AlignWord *words = (AlignWord *)malloc(MAX_WORDS * sizeof(AlignWord));
    int count = aligner_align(&aligner, audio_path, text, language, words, MAX_WORDS);

    if (count > 0) {
        if (output_path) {
            write_srt(words, count, output_path);
            fprintf(stderr, "Wrote %d words to %s\n", count, output_path);
        }
        print_aligned_words(words, count);
    } else {
        fprintf(stderr, "ERROR: Alignment failed\n");
    }

    free(words);
    aligner_free(&aligner);
    return count > 0 ? 0 : 1;
}

static int run_daemon(const char *model_dir) {
    char buf[32768];
    Aligner aligner;
    AlignWord *words = NULL;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    if (aligner_init(&aligner, model_dir) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize aligner\n");
        return 1;
    }

    words = (AlignWord *)malloc(MAX_WORDS * sizeof(AlignWord));
    if (!words) {
        fprintf(stderr, "ERROR: Failed to allocate output buffer\n");
        aligner_free(&aligner);
        return 1;
    }

    while (fgets(buf, sizeof(buf), stdin)) {
        strip_newline(buf);
        if (buf[0] == '\0') {
            break;
        }

        char *audio_path = buf;
        char *text = strchr(audio_path, '\t');
        if (!text) {
            fprintf(stderr, "ERROR: Invalid daemon request (expected AUDIO<TAB>TEXT<TAB>LANGUAGE)\n");
            fprintf(stdout, "\n");
            fprintf(stderr, "[daemon] END_REQUEST\n");
            fflush(stdout);
            fflush(stderr);
            continue;
        }
        *text++ = '\0';

        char *language = strchr(text, '\t');
        if (!language) {
            fprintf(stderr, "ERROR: Invalid daemon request (expected AUDIO<TAB>TEXT<TAB>LANGUAGE)\n");
            fprintf(stdout, "\n");
            fprintf(stderr, "[daemon] END_REQUEST\n");
            fflush(stdout);
            fflush(stderr);
            continue;
        }
        *language++ = '\0';

        int count = aligner_align(&aligner, audio_path, text, language, words, MAX_WORDS);
        if (count < 0) {
            fprintf(stderr, "ERROR: Alignment failed for %s\n", audio_path);
        } else {
            print_aligned_words(words, count);
        }
        fprintf(stdout, "\n");
        fprintf(stderr, "[daemon] END_REQUEST\n");
        fflush(stdout);
        fflush(stderr);
    }

    free(words);
    aligner_free(&aligner);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *audio_path = NULL;
    const char *text = NULL;
    const char *language = "Chinese";
    const char *output_path = NULL;
    int daemon_mode = 0;

    const char *audio_features_path = NULL;
    int audio_features_tokens = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (strcmp(argv[i], "--audio-features") == 0 && i + 1 < argc) {
            audio_features_path = argv[++i];
        } else if (strcmp(argv[i], "--audio-features-tokens") == 0 && i + 1 < argc) {
            audio_features_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--daemon") == 0) {
            daemon_mode = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    if (!model_dir) {
        usage(argv[0]);
        return 1;
    }

    if (daemon_mode) {
        return run_daemon(model_dir);
    }

    if (audio_features_path && text && audio_features_tokens > 0) {
        Aligner aligner;
        if (aligner_init(&aligner, model_dir) != 0) return 1;
        AlignWord *words = (AlignWord *)malloc(MAX_WORDS * sizeof(AlignWord));
        int count = aligner_align_with_features(&aligner, audio_features_path,
                                                 audio_features_tokens, text, language,
                                                 words, MAX_WORDS);
        if (count > 0) print_aligned_words(words, count);
        free(words);
        aligner_free(&aligner);
        return count > 0 ? 0 : 1;
    }

    if (!audio_path || !text) {
        usage(argv[0]);
        return 1;
    }

    return run_oneshot(model_dir, audio_path, text, language, output_path);
}
