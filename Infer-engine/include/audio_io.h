#ifndef GLMASR_AUDIO_IO_H
#define GLMASR_AUDIO_IO_H

int load_pcm_f32_file(const char *audio_path, float **pcm_out, int *num_samples_out);
int load_wav_file(const char *audio_path, float **pcm_out, int *num_samples_out);
int load_audio_file(const char *audio_path, float **pcm_out, int *num_samples_out);

#endif
