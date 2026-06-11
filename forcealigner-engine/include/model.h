#ifndef FA_MODEL_H
#define FA_MODEL_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    AlignerConfig cfg;
    WeightStore   ws;
    void         *cublas;    /* cublasHandle_t */
    Tensor        embed_weight;  /* [vocab, hidden] */
    Tensor        lm_head_weight; /* [classify_num, hidden] */
    /* Scratch buffers */
    Tensor        scratch;
    size_t        scratch_elems;
} AlignerModel;

/* Parse config.json and load model weights */
int aligner_model_load(AlignerModel *m, const char *model_dir);

/* Free model resources */
void aligner_model_free(AlignerModel *m);

#ifdef __cplusplus
}
#endif

#endif /* FA_MODEL_H */
