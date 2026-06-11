#ifndef FA_SAFETENSORS_H
#define FA_SAFETENSORS_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Load safetensors file, parse header, mmap data, upload tensors to GPU.
 * Returns 0 on success, -1 on failure. */
int safetensors_load(const char *path, WeightStore *ws);

/* Free all weight store resources */
void safetensors_free(WeightStore *ws);

#ifdef __cplusplus
}
#endif

#endif /* FA_SAFETENSORS_H */
