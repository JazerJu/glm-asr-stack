#ifndef FA_AUDIO_IO_H
#define FA_AUDIO_IO_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load WAV file, convert to mono 16kHz float32.
 * Returns number of samples, writes to *out_samples.
 * Caller must free *out_samples with free(). */
int load_wav_file(const char *path, float **out_samples, int *out_sample_rate);

/* Load raw float32 PCM file. */
int load_pcm_file(const char *path, float **out_samples, int *out_count);

/* Load audio file (auto-detect format by extension). */
int load_audio_file(const char *path, float **out_samples, int *out_count);

#ifdef __cplusplus
}
#endif

#endif /* FA_AUDIO_IO_H */
