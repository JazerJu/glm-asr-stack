#ifndef GLMASR_SAFETENSORS_H
#define GLMASR_SAFETENSORS_H

#include "types.h"

int  safetensors_load(const char *path, WeightStore *ws);
void safetensors_free(WeightStore *ws);

#endif
