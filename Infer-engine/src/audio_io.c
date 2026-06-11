#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "../include/audio_io.h"

#define DR_WAV_IMPLEMENTATION
#include "../third_party/dr_wav.h"

#define AUDIO_IO_EXPECTED_SAMPLE_RATE 16000

static int has_suffix(const char *path, const char *suffix) {
    size_t n = strlen(path);
    size_t m = strlen(suffix);
    return n >= m && strcmp(path + n - m, suffix) == 0;
}

static char *shell_quote(const char *s) {
    size_t len = 2;
    char *out;
    char *p;

    for (const char *c = s; *c; ++c) {
        len += (*c == '\'') ? 4 : 1;
    }

    out = malloc(len + 1);
    if (!out) {
        return NULL;
    }

    p = out;
    *p++ = '\'';
    for (const char *c = s; *c; ++c) {
        if (*c == '\'') {
            memcpy(p, "'\\''", 4);
            p += 4;
        } else {
            *p++ = *c;
        }
    }
    *p++ = '\'';
    *p = '\0';
    return out;
}

int load_pcm_f32_file(const char *audio_path, float **pcm_out, int *num_samples_out) {
    FILE *f = fopen(audio_path, "rb");
    long audio_len;
    int num_samples;
    float *pcm;

    if (!f) {
        fprintf(stderr, "ERROR: Cannot open audio file: %s\n", audio_path);
        return -1;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "ERROR: Failed to seek audio file: %s\n", audio_path);
        fclose(f);
        return -1;
    }
    audio_len = ftell(f);
    if (audio_len < 0) {
        fprintf(stderr, "ERROR: Failed to get audio size: %s\n", audio_path);
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: Failed to rewind audio file: %s\n", audio_path);
        fclose(f);
        return -1;
    }

    if ((audio_len % (long)sizeof(float)) != 0) {
        fprintf(stderr,
                "ERROR: Raw PCM file size is not a multiple of float32 samples: %s\n",
                audio_path);
        fclose(f);
        return -1;
    }
    if ((audio_len / (long)sizeof(float)) > (long)INT_MAX) {
        fprintf(stderr, "ERROR: Raw PCM file is too large: %s\n", audio_path);
        fclose(f);
        return -1;
    }

    num_samples = (int)(audio_len / (long)sizeof(float));
    pcm = malloc((size_t)num_samples * sizeof(float));
    if (!pcm) {
        fprintf(stderr, "ERROR: Failed to allocate PCM buffer\n");
        fclose(f);
        return -1;
    }

    if (fread(pcm, sizeof(float), (size_t)num_samples, f) != (size_t)num_samples) {
        fprintf(stderr, "ERROR: Failed to read PCM data\n");
        free(pcm);
        fclose(f);
        return -1;
    }
    fclose(f);

    *pcm_out = pcm;
    *num_samples_out = num_samples;
    return 0;
}

int load_wav_file(const char *audio_path, float **pcm_out, int *num_samples_out) {
    drwav wav;
    drwav_uint64 frame_count;
    drwav_uint64 frames_read;
    float *pcm;

    if (!drwav_init_file(&wav, audio_path, NULL)) {
        fprintf(stderr, "ERROR: Failed to open WAV file: %s\n", audio_path);
        return -1;
    }

    if (wav.channels != 1 || wav.sampleRate != AUDIO_IO_EXPECTED_SAMPLE_RATE) {
        fprintf(stderr,
                "ERROR: Unsupported WAV format for %s (channels=%u sample_rate=%u; expected mono %d Hz)\n",
                audio_path,
                wav.channels,
                wav.sampleRate,
                AUDIO_IO_EXPECTED_SAMPLE_RATE);
        drwav_uninit(&wav);
        return -1;
    }

    frame_count = wav.totalPCMFrameCount;
    if (frame_count > (drwav_uint64)INT_MAX) {
        fprintf(stderr, "ERROR: WAV file is too large: %s\n", audio_path);
        drwav_uninit(&wav);
        return -1;
    }

    pcm = malloc((size_t)frame_count * sizeof(float));
    if (!pcm) {
        fprintf(stderr, "ERROR: Failed to allocate WAV buffer\n");
        drwav_uninit(&wav);
        return -1;
    }

    frames_read = drwav_read_pcm_frames_f32(&wav, frame_count, pcm);
    drwav_uninit(&wav);
    if (frames_read != frame_count) {
        fprintf(stderr, "ERROR: Failed to read complete WAV payload: %s\n", audio_path);
        free(pcm);
        return -1;
    }

    *pcm_out = pcm;
    *num_samples_out = (int)frame_count;
    return 0;
}

static int load_audio_ffmpeg(const char *audio_path, float **pcm_out, int *num_samples_out) {
    char *quoted_path = NULL;
    char *cmd = NULL;
    FILE *pipe = NULL;
    float *pcm = NULL;
    size_t count = 0;
    size_t capacity = 262144;
    int status;
    int ret = -1;

    quoted_path = shell_quote(audio_path);
    if (!quoted_path) {
        fprintf(stderr, "ERROR: Failed to quote audio path\n");
        return -1;
    }

    cmd = malloc(strlen(quoted_path) + 160);
    if (!cmd) {
        fprintf(stderr, "ERROR: Failed to allocate ffmpeg command\n");
        goto cleanup;
    }
    sprintf(cmd,
            "ffmpeg -nostdin -hide_banner -loglevel error -i %s "
            "-f f32le -acodec pcm_f32le -ac 1 -ar %d pipe:1",
            quoted_path, AUDIO_IO_EXPECTED_SAMPLE_RATE);

    pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "ERROR: Failed to start ffmpeg for %s\n", audio_path);
        goto cleanup;
    }

    pcm = malloc(capacity * sizeof(*pcm));
    if (!pcm) {
        fprintf(stderr, "ERROR: Failed to allocate ffmpeg PCM buffer\n");
        goto cleanup;
    }

    for (;;) {
        size_t room = capacity - count;
        size_t nread;

        if (room == 0) {
            size_t new_capacity = capacity * 2;
            float *new_pcm;
            if (new_capacity <= capacity || new_capacity > (size_t)INT_MAX) {
                fprintf(stderr, "ERROR: Decoded audio is too large: %s\n", audio_path);
                goto cleanup;
            }
            new_pcm = realloc(pcm, new_capacity * sizeof(*pcm));
            if (!new_pcm) {
                fprintf(stderr, "ERROR: Failed to grow ffmpeg PCM buffer\n");
                goto cleanup;
            }
            pcm = new_pcm;
            capacity = new_capacity;
            room = capacity - count;
        }

        nread = fread(pcm + count, sizeof(*pcm), room, pipe);
        count += nread;
        if (nread < room) {
            if (ferror(pipe)) {
                fprintf(stderr, "ERROR: Failed to read ffmpeg output for %s\n", audio_path);
                goto cleanup;
            }
            break;
        }
    }

    status = pclose(pipe);
    pipe = NULL;
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "ERROR: ffmpeg failed for %s\n", audio_path);
        goto cleanup;
    }
    if (count == 0 || count > (size_t)INT_MAX) {
        fprintf(stderr, "ERROR: ffmpeg produced invalid sample count for %s\n", audio_path);
        goto cleanup;
    }

    *pcm_out = pcm;
    *num_samples_out = (int)count;
    pcm = NULL;
    ret = 0;

cleanup:
    if (pipe) {
        pclose(pipe);
    }
    free(pcm);
    free(cmd);
    free(quoted_path);
    return ret;
}

int load_audio_file(const char *audio_path, float **pcm_out, int *num_samples_out) {
    if (has_suffix(audio_path, ".wav")) {
        return load_wav_file(audio_path, pcm_out, num_samples_out);
    }
    if (has_suffix(audio_path, ".bin")) {
        return load_pcm_f32_file(audio_path, pcm_out, num_samples_out);
    }

    return load_audio_ffmpeg(audio_path, pcm_out, num_samples_out);
}
