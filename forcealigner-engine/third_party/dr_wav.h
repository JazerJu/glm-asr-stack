#ifndef DR_WAV_H
#define DR_WAV_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef uint8_t  drwav_uint8;
typedef int8_t   drwav_int8;
typedef uint16_t drwav_uint16;
typedef int16_t  drwav_int16;
typedef uint32_t drwav_uint32;
typedef int32_t  drwav_int32;
typedef uint64_t drwav_uint64;
typedef int      drwav_bool32;

#define DRWAV_TRUE  1
#define DRWAV_FALSE 0

#define DR_WAVE_FORMAT_PCM        0x0001
#define DR_WAVE_FORMAT_IEEE_FLOAT 0x0003

typedef struct {
    FILE         *pFile;
    drwav_uint16  formatTag;
    drwav_uint16  channels;
    drwav_uint32  sampleRate;
    drwav_uint16  bitsPerSample;
    drwav_uint16  bytesPerFrame;
    drwav_uint64  totalPCMFrameCount;
    drwav_uint64  dataChunkDataPos;
    drwav_uint64  dataChunkDataSize;
    drwav_uint64  bytesRemaining;
} drwav;

drwav_bool32 drwav_init_file(drwav *pWav, const char *filename, const void *pAllocationCallbacks);
void drwav_uninit(drwav *pWav);
drwav_uint64 drwav_read_pcm_frames_f32(drwav *pWav, drwav_uint64 framesToRead, float *pBufferOut);

#ifdef DR_WAV_IMPLEMENTATION

static drwav_uint16 drwav__bytes_to_u16le(const drwav_uint8 *data) {
    return (drwav_uint16)(data[0] | (data[1] << 8));
}

static drwav_uint32 drwav__bytes_to_u32le(const drwav_uint8 *data) {
    return ((drwav_uint32)data[0]) |
           ((drwav_uint32)data[1] << 8) |
           ((drwav_uint32)data[2] << 16) |
           ((drwav_uint32)data[3] << 24);
}

static drwav_int32 drwav__s24_to_s32(const drwav_uint8 *data) {
    drwav_uint32 value = ((drwav_uint32)data[0]) |
                         ((drwav_uint32)data[1] << 8) |
                         ((drwav_uint32)data[2] << 16);
    if ((value & 0x00800000U) != 0) {
        value |= 0xFF000000U;
    }
    return (drwav_int32)value;
}

static float drwav__pcm_sample_to_f32(const drwav_uint8 *sample, drwav_uint16 formatTag, drwav_uint16 bitsPerSample) {
    if (formatTag == DR_WAVE_FORMAT_IEEE_FLOAT && bitsPerSample == 32) {
        float value;
        memcpy(&value, sample, sizeof(value));
        return value;
    }

    switch (bitsPerSample) {
        case 8: {
            return ((float)((int)sample[0] - 128)) / 128.0f;
        }
        case 16: {
            drwav_int16 value = (drwav_int16)drwav__bytes_to_u16le(sample);
            return (float)value / 32768.0f;
        }
        case 24: {
            drwav_int32 value = drwav__s24_to_s32(sample);
            return (float)value / 8388608.0f;
        }
        case 32: {
            drwav_int32 value = (drwav_int32)drwav__bytes_to_u32le(sample);
            return (float)value / 2147483648.0f;
        }
        default: {
            return 0.0f;
        }
    }
}

static drwav_bool32 drwav__skip_bytes(FILE *pFile, drwav_uint64 bytesToSkip) {
    if (bytesToSkip > 0x7fffffffULL) {
        while (bytesToSkip > 0x7fffffffULL) {
            if (fseek(pFile, 0x7fffffff, SEEK_CUR) != 0) {
                return DRWAV_FALSE;
            }
            bytesToSkip -= 0x7fffffffULL;
        }
    }
    return fseek(pFile, (long)bytesToSkip, SEEK_CUR) == 0 ? DRWAV_TRUE : DRWAV_FALSE;
}

drwav_bool32 drwav_init_file(drwav *pWav, const char *filename, const void *pAllocationCallbacks) {
    drwav_uint8 riff[12];
    drwav_bool32 foundFmt = DRWAV_FALSE;
    drwav_bool32 foundData = DRWAV_FALSE;

    (void)pAllocationCallbacks;

    if (pWav == NULL || filename == NULL) {
        return DRWAV_FALSE;
    }

    memset(pWav, 0, sizeof(*pWav));
    pWav->pFile = fopen(filename, "rb");
    if (pWav->pFile == NULL) {
        return DRWAV_FALSE;
    }

    if (fread(riff, 1, sizeof(riff), pWav->pFile) != sizeof(riff)) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    if (memcmp(riff + 0, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    while (!foundData) {
        drwav_uint8 header[8];
        drwav_uint32 chunkSize;
        drwav_uint64 chunkSizePadded;

        if (fread(header, 1, sizeof(header), pWav->pFile) != sizeof(header)) {
            break;
        }

        chunkSize = drwav__bytes_to_u32le(header + 4);
        chunkSizePadded = (drwav_uint64)chunkSize + (drwav_uint64)(chunkSize & 1U);

        if (memcmp(header + 0, "fmt ", 4) == 0) {
            drwav_uint8 fmt[16];
            if (chunkSize < sizeof(fmt)) {
                drwav_uninit(pWav);
                return DRWAV_FALSE;
            }
            if (fread(fmt, 1, sizeof(fmt), pWav->pFile) != sizeof(fmt)) {
                drwav_uninit(pWav);
                return DRWAV_FALSE;
            }

            pWav->formatTag = drwav__bytes_to_u16le(fmt + 0);
            pWav->channels = drwav__bytes_to_u16le(fmt + 2);
            pWav->sampleRate = drwav__bytes_to_u32le(fmt + 4);
            pWav->bytesPerFrame = drwav__bytes_to_u16le(fmt + 12);
            pWav->bitsPerSample = drwav__bytes_to_u16le(fmt + 14);
            foundFmt = DRWAV_TRUE;

            if (chunkSizePadded > sizeof(fmt)) {
                if (!drwav__skip_bytes(pWav->pFile, chunkSizePadded - sizeof(fmt))) {
                    drwav_uninit(pWav);
                    return DRWAV_FALSE;
                }
            }
        } else if (memcmp(header + 0, "data", 4) == 0) {
            long pos = ftell(pWav->pFile);
            if (pos < 0) {
                drwav_uninit(pWav);
                return DRWAV_FALSE;
            }
            pWav->dataChunkDataPos = (drwav_uint64)pos;
            pWav->dataChunkDataSize = (drwav_uint64)chunkSize;
            foundData = DRWAV_TRUE;
            if (!drwav__skip_bytes(pWav->pFile, chunkSizePadded)) {
                drwav_uninit(pWav);
                return DRWAV_FALSE;
            }
        } else {
            if (!drwav__skip_bytes(pWav->pFile, chunkSizePadded)) {
                drwav_uninit(pWav);
                return DRWAV_FALSE;
            }
        }
    }

    if (!foundFmt || !foundData || pWav->channels == 0 || pWav->bytesPerFrame == 0) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    if (pWav->formatTag != DR_WAVE_FORMAT_PCM && pWav->formatTag != DR_WAVE_FORMAT_IEEE_FLOAT) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    if (pWav->dataChunkDataSize % pWav->bytesPerFrame != 0) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    pWav->totalPCMFrameCount = pWav->dataChunkDataSize / pWav->bytesPerFrame;
    pWav->bytesRemaining = pWav->dataChunkDataSize;

    if (fseek(pWav->pFile, (long)pWav->dataChunkDataPos, SEEK_SET) != 0) {
        drwav_uninit(pWav);
        return DRWAV_FALSE;
    }

    return DRWAV_TRUE;
}

void drwav_uninit(drwav *pWav) {
    if (pWav == NULL) {
        return;
    }

    if (pWav->pFile != NULL) {
        fclose(pWav->pFile);
        pWav->pFile = NULL;
    }

    pWav->bytesRemaining = 0;
}

drwav_uint64 drwav_read_pcm_frames_f32(drwav *pWav, drwav_uint64 framesToRead, float *pBufferOut) {
    drwav_uint64 bytesToRead;
    drwav_uint64 sampleCount;
    drwav_uint8 *data;
    drwav_uint64 i;
    drwav_uint64 framesRead;
    drwav_uint32 bytesPerSample;

    if (pWav == NULL || pBufferOut == NULL || pWav->pFile == NULL || pWav->bytesPerFrame == 0) {
        return 0;
    }

    if (framesToRead > pWav->bytesRemaining / pWav->bytesPerFrame) {
        framesToRead = pWav->bytesRemaining / pWav->bytesPerFrame;
    }
    if (framesToRead == 0) {
        return 0;
    }

    bytesToRead = framesToRead * pWav->bytesPerFrame;
    bytesPerSample = (drwav_uint32)(pWav->bitsPerSample / 8);
    sampleCount = framesToRead * pWav->channels;

    if (bytesPerSample == 0) {
        return 0;
    }

    data = (drwav_uint8 *)malloc((size_t)bytesToRead);
    if (data == NULL) {
        return 0;
    }

    bytesToRead = (drwav_uint64)fread(data, 1, (size_t)bytesToRead, pWav->pFile);
    pWav->bytesRemaining -= bytesToRead;
    framesRead = bytesToRead / pWav->bytesPerFrame;
    sampleCount = framesRead * pWav->channels;

    for (i = 0; i < sampleCount; ++i) {
        pBufferOut[i] = drwav__pcm_sample_to_f32(data + i * bytesPerSample,
                                                  pWav->formatTag,
                                                  pWav->bitsPerSample);
    }

    free(data);
    return framesRead;
}

#endif

#endif
