#ifndef GLMASR_MODEL_H
#define GLMASR_MODEL_H

#include "types.h"
#include "kv_pool.h"

typedef struct {
    ModelConfig  cfg;
    WeightStore  ws;
    KVPool      *kv_pool;       /* paged KV block pool (NULL until warmup) */
    KVCache      kv;            /* legacy contiguous cache, kept for backward compat */
    void        *cublas;        /* cublasHandle_t, opaque in C */
    Tensor       dequant_buf;   /* reusable bf16 buffer for FP4 dequant */
    size_t       dequant_weight_scratch_elems;
    size_t       dequant_aux_scratch_elems;
} Model;

int   model_load(Model *m, const char *model_dir);
void  model_free(Model *m);

void  model_config_from_json(const char *path, ModelConfig *cfg);

#endif
