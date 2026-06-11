#define DR_WAV_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_io.h"
#include "../third_party/dr_wav.h"

int load_wav_file(const char *path, float **out_samples, int *out_sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, path, NULL)) {
        fprintf(stderr, "ERROR: Failed to open WAV: %s\n", path);
        return -1;
    }

    if (wav.sampleRate != 16000) {
        fprintf(stderr, "WARNING: WAV sample rate %u != 16000, resampling not implemented\n", wav.sampleRate);
    }

    drwav_uint64 total_frames = wav.totalPCMFrameCount;
    float *raw = (float *)malloc((size_t)total_frames * wav.channels * sizeof(float));
    if (!raw) { drwav_uninit(&wav); return -1; }

    drwav_uint64 frames_read = drwav_read_pcm_frames_f32(&wav, total_frames, raw);
    drwav_uninit(&wav);

    float *mono = (float *)malloc(frames_read * sizeof(float));
    if (!mono) { free(raw); return -1; }

    if (wav.channels == 1) {
        memcpy(mono, raw, frames_read * sizeof(float));
    } else {
        for (drwav_uint64 i = 0; i < frames_read; i++) {
            float sum = 0.0f;
            for (int c = 0; c < wav.channels; c++) {
                sum += raw[i * wav.channels + c];
            }
            mono[i] = sum / (float)wav.channels;
        }
    }
    free(raw);

    float peak = 0.0f;
    for (drwav_uint64 i = 0; i < frames_read; i++) {
        float a = mono[i] < 0 ? -mono[i] : mono[i];
        if (a > peak) peak = a;
    }
    if (peak > 1.0f) {
        for (drwav_uint64 i = 0; i < frames_read; i++) {
            mono[i] /= peak;
        }
    }

    *out_samples = mono;
    *out_sample_rate = (int)wav.sampleRate;
    return (int)frames_read;
}

int load_pcm_file(const char *path, float **out_samples, int *out_count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open PCM file: %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int count = size / (int)sizeof(float);
    float *buf = (float *)malloc(size);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, sizeof(float), count, f) != (size_t)count) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_samples = buf;
    *out_count = count;
    return count;
}

int load_audio_file(const char *path, float **out_samples, int *out_count) {
    const char *ext = strrchr(path, '.');
    if (ext && (strcmp(ext, ".wav") == 0 || strcmp(ext, ".WAV") == 0)) {
        int sr = 0;
        int ret = load_wav_file(path, out_samples, &sr);
        if (ret < 0) return ret;
        *out_count = ret;
        return ret;
    }
    return load_pcm_file(path, out_samples, out_count);
}
