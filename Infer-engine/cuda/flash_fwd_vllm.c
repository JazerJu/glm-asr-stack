#include "flash_fwd_vllm.h"
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>

typedef int (*flash_fwd_bridge_fn)(void*, void*, void*, void*, int, int, int, float, CUstream);
typedef int (*flash_fwd_strided_fn)(void*, void*, void*, void*, int, int, int, float, int, int, int, CUstream);
typedef int (*thin_avail_fn)(void);
typedef int (*thin_launch_strided_fn)(void*, void*, void*, void*, int, int, int, float, int*, int, int, int, CUstream);
typedef int (*thin_gqa_avail_fn)(void);
typedef int (*thin_gqa_launch_fn)(void*, void*, void*, void*, int, int, int, int, int, int, int, int, int, int*, int, CUstream);
typedef int (*thin_varlen_launch_fn)(void*, void*, void*, void*, int, int, int, int, int, int, int, int, int*, CUstream);
typedef int (*thin_dense_tma_avail_fn)(void);
typedef int (*thin_dense_tma_launch_fn)(void*, void*, void*, void*, int, int, int, int, int, int, int, CUstream);

static void *g_vllm_handle = NULL;
static void *g_bridge_handle = NULL;
static flash_fwd_bridge_fn g_bridge_fn = NULL;
static flash_fwd_strided_fn g_strided_fn = NULL;
static thin_avail_fn g_thin_avail = NULL;
static thin_launch_strided_fn g_thin_launch = NULL;
static thin_gqa_avail_fn g_thin_gqa_avail = NULL;
static thin_gqa_launch_fn g_thin_gqa_launch = NULL;
static thin_varlen_launch_fn g_thin_varlen_launch = NULL;
static thin_dense_tma_avail_fn g_thin_dense_tma_avail = NULL;
static thin_dense_tma_launch_fn g_thin_dense_tma_launch = NULL;
static int g_init_done = 0;

static int *g_cu_seqlens = NULL;
static int g_cu_seqlens_size = 0;

static int ensure_cu_seqlens(int seqlen) {
    if (g_cu_seqlens && g_cu_seqlens_size >= (seqlen + 1)) return 0;
    if (g_cu_seqlens) { cudaFree(g_cu_seqlens); g_cu_seqlens = NULL; }
    g_cu_seqlens_size = seqlen + 1;
    int host_cu[2] = {0, seqlen};
    if (cudaMalloc((void**)&g_cu_seqlens, g_cu_seqlens_size * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "FA2: cu_seqlens alloc failed\n"); return -1;
    }
    if (cudaMemcpy(g_cu_seqlens, host_cu, 2 * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "FA2: cu_seqlens copy failed\n"); cudaFree(g_cu_seqlens); g_cu_seqlens = NULL; return -1;
    }
    return 0;
}

static void fa2_init(void) {
    if (g_init_done) return;
    g_init_done = 1;

    /* Try CuteDSL FA2 bridge first (our self-contained kernel) */
    void *cutedsl = dlopen("cutedsl_fa/fa2_bridge.so", RTLD_LAZY | RTLD_LOCAL);
    if (cutedsl) {
        thin_avail_fn enc_avail = (thin_avail_fn)dlsym(cutedsl, "flash_fwd_vllm_available");
        if (enc_avail && enc_avail()) {
            g_thin_avail  = enc_avail;
            g_thin_launch = (thin_launch_strided_fn)dlsym(cutedsl, "flash_fwd_vllm_launch");
            g_thin_gqa_avail = (thin_gqa_avail_fn)dlsym(cutedsl, "flash_fwd_vllm_gqa_available");
            g_thin_gqa_launch = (thin_gqa_launch_fn)dlsym(cutedsl, "flash_fwd_vllm_gqa_launch");
            g_thin_varlen_launch = (thin_varlen_launch_fn)dlsym(cutedsl, "flash_fwd_vllm_launch_varlen");
            g_thin_dense_tma_avail = (thin_dense_tma_avail_fn)dlsym(cutedsl, "flash_fwd_vllm_dense_tma_available");
            g_thin_dense_tma_launch = (thin_dense_tma_launch_fn)dlsym(cutedsl, "flash_fwd_vllm_launch_dense_tma");
            fprintf(stderr, "FA2 CuteDSL: loaded direct-link kernel\n");
            return;
        }
        fprintf(stderr, "FA2 CuteDSL: not available, trying vLLM...\n");
        dlclose(cutedsl);
    }

    const char *vp = "/data/fwsr/glm-asr/GLM-ASR/venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so";
    g_vllm_handle = dlopen(vp, RTLD_LAZY | RTLD_GLOBAL);
    if (!g_vllm_handle) { fprintf(stderr,"FA2: dlopen vllm.so: %s\n",dlerror()); return; }

    void *th = dlopen("build/libthin_fa2_wrapper.so", RTLD_LAZY | RTLD_LOCAL);
    if (th) {
        g_thin_avail = (thin_avail_fn)dlsym(th, "thin_wrapper_available");
        g_thin_launch = (thin_launch_strided_fn)dlsym(th, "thin_fa2_launch_strided");
        g_thin_gqa_avail = (thin_gqa_avail_fn)dlsym(th, "thin_fa2_gqa_available");
        g_thin_gqa_launch = (thin_gqa_launch_fn)dlsym(th, "thin_fa2_gqa_launch");
        g_thin_varlen_launch = (thin_varlen_launch_fn)dlsym(th, "thin_fa2_varlen_launch");
        g_thin_dense_tma_avail = NULL;
        g_thin_dense_tma_launch = NULL;
        if (g_thin_avail && g_thin_avail()) {
            fprintf(stderr, "FA2 thin: loaded (gqa=%d)\n", g_thin_gqa_avail ? g_thin_gqa_avail() : 0);
            return;
        }
        fprintf(stderr, "FA2 thin: not available, falling back to bridge\n");
        dlclose(th); g_thin_avail = NULL; g_thin_launch = NULL;
        g_thin_gqa_avail = NULL; g_thin_gqa_launch = NULL;
        g_thin_dense_tma_avail = NULL; g_thin_dense_tma_launch = NULL;
    }

    const char *bp = "build/libflash_fwd_bridge.so";
    g_bridge_handle = dlopen(bp, RTLD_LAZY | RTLD_LOCAL);
    if (!g_bridge_handle) { fprintf(stderr,"FA2: dlopen bridge: %s\n",dlerror()); dlclose(g_vllm_handle); g_vllm_handle=NULL; return; }
    g_bridge_fn = (flash_fwd_bridge_fn)dlsym(g_bridge_handle, "flash_fwd_launch_varlen");
    g_strided_fn = (flash_fwd_strided_fn)dlsym(g_bridge_handle, "flash_fwd_launch_varlen_strided");
    if (!g_bridge_fn) { fprintf(stderr,"FA2: dlsym: %s\n",dlerror()); dlclose(g_bridge_handle); dlclose(g_vllm_handle); g_bridge_handle=g_vllm_handle=NULL; return; }
    fprintf(stderr,"FA2 bridge: loaded\n");
}

int flash_fwd_bridge_available(void) { fa2_init(); return g_thin_launch != NULL || g_bridge_fn != NULL; }
int flash_fwd_vllm_available(void) { return flash_fwd_bridge_available(); }

int flash_fwd_bridge_launch(void *q, void *k, void *v, void *out, int seqlen, int nheads, int hdim, float scale) {
    if (seqlen<=0 || nheads<=0 || hdim!=64) return 0;
    if (scale <= 0.0f) scale = 1.0f/sqrtf((float)hdim);

    if (g_thin_launch) {
        if (ensure_cu_seqlens(seqlen) != 0) return 0;
        int row_stride = nheads * hdim;
        if (g_thin_launch(q,k,v,out,seqlen,nheads,hdim,scale,g_cu_seqlens,row_stride,row_stride,row_stride,(CUstream)0) != 0) {
            fprintf(stderr,"FA2 thin: launch failed\n"); return 0;
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr,"FA2 thin: %s\n",cudaGetErrorString(e)); return 0; }
        return 1;
    }

    if (!g_bridge_fn) return 0;
    if (g_bridge_fn(q,k,v,out,seqlen,nheads,hdim,scale,(CUstream)0) != 0) { fprintf(stderr,"FA2: launch failed\n"); return 0; }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr,"FA2: post-launch: %s\n",cudaGetErrorString(e)); return 0; }
    return 1;
}

int flash_fwd_vllm_launch(void *q, void *k, void *v, void *out, int seqlen, int hdim, int qs, int ks, int vs, int nh, int headdim) {
    if (seqlen<=0 || nh<=0 || headdim!=64) return 0;
    float scale = 1.0f/sqrtf((float)headdim);

    if (g_thin_launch) {
        if (ensure_cu_seqlens(seqlen) != 0) return 0;
        if (g_thin_launch(q,k,v,out,seqlen,nh,headdim,scale,g_cu_seqlens,qs,ks,vs,(CUstream)0) != 0) return 0;
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr,"FA2 thin: %s\n",cudaGetErrorString(e)); return 0; }
        return 1;
    }

    int std_hdim = nh * headdim;
    if (qs == std_hdim && ks == std_hdim && vs == std_hdim) {
        return flash_fwd_bridge_launch(q,k,v,out,seqlen,nh,headdim,scale);
    }

    if (!g_strided_fn) return 0;
    if (g_strided_fn(q,k,v,out,seqlen,nh,headdim,scale,qs,ks,vs,(CUstream)0) != 0) {
        fprintf(stderr,"FA2: strided launch failed\n"); return 0;
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr,"FA2: post-launch: %s\n",cudaGetErrorString(e)); return 0; }
    return 1;
}

int flash_fwd_vllm_gqa_available(void) {
    fa2_init();
    return (g_thin_gqa_avail && g_thin_gqa_avail()) ? 1 : 0;
}

int flash_fwd_vllm_gqa_launch(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int q_heads, int kv_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_row_stride,
    int num_seqs,
    int *d_cu_seqlens,
    int is_causal)
{
    fa2_init();
    if (!g_thin_gqa_launch) return 0;
    int rc = g_thin_gqa_launch(q, k, v, out, total_seqlen, max_seqlen,
                               q_heads, kv_heads, head_dim,
                               q_row_stride, k_row_stride, v_row_stride,
                               num_seqs, d_cu_seqlens, is_causal, (CUstream)0);
    return (rc == 0) ? 1 : 0;
}

int flash_fwd_vllm_launch_varlen(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int num_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_row_stride,
    int num_seqs,
    int *d_cu_seqlens)
{
    fa2_init();
    if (!g_thin_varlen_launch) return 0;
    int rc = g_thin_varlen_launch(q, k, v, out, total_seqlen, max_seqlen,
                                  num_heads, head_dim,
                                  q_row_stride, k_row_stride, v_row_stride,
                                  num_seqs, d_cu_seqlens, (CUstream)0);
    return (rc == 0) ? 1 : 0;
}

int flash_fwd_vllm_dense_tma_available(void) {
    fa2_init();
    return (g_thin_dense_tma_avail && g_thin_dense_tma_avail() && g_thin_dense_tma_launch) ? 1 : 0;
}

int flash_fwd_vllm_launch_dense_tma(
    void *q, void *k, void *v, void *out,
    int num_seqs, int seq_len,
    int num_heads, int head_dim,
    int q_row_stride, int k_row_stride, int v_row_stride)
{
    fa2_init();
    if (!g_thin_dense_tma_launch || head_dim != 64 || num_seqs <= 0 || seq_len <= 0) return 0;
    int rc = g_thin_dense_tma_launch(q, k, v, out, num_seqs, seq_len,
                                     num_heads, head_dim,
                                     q_row_stride, k_row_stride, v_row_stride,
                                     (CUstream)0);
    return (rc == 0) ? 1 : 0;
}
