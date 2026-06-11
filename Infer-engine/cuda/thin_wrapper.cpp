// thin_wrapper.cpp — Direct call to flash::run_mha_fwd_<bfloat16_t, Headdim, IsCausal>
// Bypasses torch dispatcher (~0.5µs/call vs ~190µs/call).
//
// Compile inside nvidia/cuda:12.9.1-devel-ubuntu20.04 (GCC 10.5.0):
//   g++ -shared -fPIC -O2 -o libthin_fa2_wrapper.so thin_wrapper.cpp \
//       -I/usr/local/cuda/include
//
// At runtime, dlsym's the run_mha_fwd symbol from vLLM's _vllm_fa2_C.abi3.so.
// The Flash_fwd_params struct ABI must exactly match what vLLM 0.18.0 expects.
//
// Exports two launch paths:
//   thin_fa2_launch_strided  — encoder FA2 (head_dim=64, MHA, non-causal)
//   thin_fa2_gqa_launch      — decoder FA2 (head_dim=128, GQA, causal)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <dlfcn.h>

// ---- ABI-exact struct definitions ----
// Source: vllm-project/flash-attention @ 1488682, csrc/flash_attn/src/flash.h
// PhiloxCudaState: pytorch v2.10.0, aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh

struct PhiloxCudaState {
    union Payload {
        uint64_t val;
        int64_t* ptr;
    };
    Payload seed_{};
    Payload offset_{};
    uint64_t offset_intragraph_ = 0;
    bool captured_ = false;
};

struct Qkv_params {
    using index_t = int64_t;
    void *q_ptr;
    void *k_ptr;
    void *v_ptr;
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    int h;
    int h_k;
    int h_h_k_ratio;
};

struct Flash_fwd_params : public Qkv_params {
    void *o_ptr;
    void *oaccum_ptr;
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    void *p_ptr;
    void *softmax_lse_ptr;
    void *softmax_lseaccum_ptr;

    int b;
    int seqlen_q;
    int seqlen_k;
    int seqlen_knew;
    int d;
    int seqlen_q_rounded;
    int seqlen_k_rounded;
    int d_rounded;
    int rotary_dim;
    int total_q;

    float scale_softmax;
    float scale_softmax_log2;

    int *cu_seqlens_q;
    int *cu_seqlens_k;
    int *leftpad_k;
    int *seqused_k;
    int *blockmask;

    void *knew_ptr;
    void *vnew_ptr;
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void *rotary_cos_ptr;
    void *rotary_sin_ptr;
    int *cache_batch_idx;

    int *block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    float scale_softmax_rp_dropout;

    int window_size_left;
    int window_size_right;
    float softcap;

    PhiloxCudaState philox_args;
    uint64_t *rng_state;

    bool is_bf16;
    bool is_causal;
    bool is_seqlens_k_cumulative;
    bool is_rotary_interleaved;
    int num_splits;

    void *alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;
    bool seqlenq_ngroups_swapped;
};

// ---- Cached softmax_lse buffer ----
// The FA2 kernel always writes log-sum-exp to softmax_lse_ptr.
// For b=1 varlen with unpadded_lse=true, shape is [nheads, seqlen_q] float32.
static float *g_lse_buf = nullptr;
static int g_lse_buf_size = 0;  // in elements

static float *ensure_lse_buf(int nheads, int seqlen) {
    int need = nheads * seqlen;
    if (g_lse_buf && g_lse_buf_size >= need) return g_lse_buf;
    if (g_lse_buf) cudaFree(g_lse_buf);
    cudaError_t err = cudaMalloc(&g_lse_buf, (size_t)need * sizeof(float));
    if (err != cudaSuccess) { g_lse_buf = nullptr; return nullptr; }
    g_lse_buf_size = need;
    return g_lse_buf;
}

// ---- Runtime symbol resolution ----

static void *g_vllm_handle = NULL;
using RunMhaFn = void(*)(Flash_fwd_params &, cudaStream_t);
static RunMhaFn g_run_mha_fn = nullptr;      // head_dim=64, non-causal (encoder)
static RunMhaFn g_run_mha_h128_fn = nullptr;  // head_dim=128, causal (decoder)
static int g_init = 0;

static const char *SYM_HDIM64 =
    "_ZN5flash12run_mha_fwd_IN7cutlass10bfloat16_tELi64ELb0EEEvRNS_16Flash_fwd_paramsEP11CUstream_st";
static const char *SYM_HDIM128_CAUSAL =
    "_ZN5flash12run_mha_fwd_IN7cutlass10bfloat16_tELi128ELb1EEEvRNS_16Flash_fwd_paramsEP11CUstream_st";
static const char *SYM_GENERIC =
    "_ZN5flash11run_mha_fwdERNS_16Flash_fwd_paramsEP11CUstream_stb";
static const char *VLLM_SO =
    "/data/fwsr/glm-asr/GLM-ASR/venv/lib/python3.12/site-packages/"
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so";

static void do_init() {
    if (g_init) return;
    g_init = 1;

    g_vllm_handle = dlopen(VLLM_SO, RTLD_LAZY | RTLD_GLOBAL);
    if (!g_vllm_handle) {
        fprintf(stderr, "thin_wrapper: dlopen: %s\n", dlerror());
        return;
    }

    void *sym64 = dlsym(g_vllm_handle, SYM_HDIM64);
    if (!sym64) sym64 = dlsym(g_vllm_handle, SYM_GENERIC);
    if (sym64) g_run_mha_fn = reinterpret_cast<RunMhaFn>(sym64);

    void *sym128 = dlsym(g_vllm_handle, SYM_HDIM128_CAUSAL);
    if (sym128) g_run_mha_h128_fn = reinterpret_cast<RunMhaFn>(sym128);

    if (!g_run_mha_fn && !g_run_mha_h128_fn) {
        fprintf(stderr, "thin_wrapper: no symbols found\n");
        dlclose(g_vllm_handle);
        g_vllm_handle = NULL;
        return;
    }
    fprintf(stderr, "thin_wrapper: loaded (h64=%d h128=%d)\n",
            g_run_mha_fn ? 1 : 0, g_run_mha_h128_fn ? 1 : 0);
}

// ---- C API ----

extern "C" {

int thin_wrapper_available() {
    do_init();
    return g_run_mha_fn != nullptr;
}

// Round up to nearest multiple of m
static inline int round_up(int x, int m) { return (x + m - 1) / m * m; }

int thin_fa2_launch_strided(
    void *q, void *k, void *v, void *out,
    int seqlen, int nheads, int headdim,
    float softmax_scale,
    int *d_cu_seqlens,
    int q_stride, int k_stride, int v_stride,
    cudaStream_t stream)
{
    do_init();
    if (!g_run_mha_fn || headdim != 64 || seqlen <= 0 || nheads <= 0) return -1;

    if (softmax_scale <= 0.0f)
        softmax_scale = 1.0f / sqrtf((float)headdim);

    const int hidden = nheads * headdim;

    // Allocate softmax_lse buffer: [nheads, seqlen] float32
    float *lse = ensure_lse_buf(nheads, seqlen);
    if (!lse) { fprintf(stderr, "thin_wrapper: lse alloc failed\n"); return -1; }

    // Zero-init the entire struct (matches set_params_fprop: "params = {};")
    Flash_fwd_params p{};

    // Pointers
    p.q_ptr = q;  p.k_ptr = k;  p.v_ptr = v;
    p.o_ptr = out;

    // Strides (all in elements, not bytes)
    // For varlen (cu_seqlens != null), batch strides are left as 0
    p.q_row_stride = q_stride;
    p.k_row_stride = k_stride;
    p.v_row_stride = v_stride;
    p.q_head_stride = headdim;
    p.k_head_stride = headdim;
    p.v_head_stride = headdim;
    // Output is always contiguous [seqlen, nheads, headdim]
    p.o_row_stride = hidden;   // = nheads * headdim
    p.o_head_stride = headdim;

    // softmax_lse MUST be valid — kernel always writes to it
    p.softmax_lse_ptr = lse;

    // Dimensions
    p.b = 1;
    p.h = nheads;
    p.h_k = nheads;          // MHA, not GQA
    p.h_h_k_ratio = 1;
    p.seqlen_q = seqlen;
    p.seqlen_k = seqlen;
    p.seqlen_q_rounded = round_up(seqlen, 128);  // kernel block alignment
    p.seqlen_k_rounded = round_up(seqlen, 128);
    p.d = headdim;
    p.d_rounded = round_up(headdim, 32);  // 64 → 64
    p.total_q = seqlen;

    // Scaling
    p.scale_softmax = softmax_scale;
    p.scale_softmax_log2 = softmax_scale * (float)M_LOG2E;

    // Varlen pointers
    p.cu_seqlens_q = d_cu_seqlens;
    p.cu_seqlens_k = d_cu_seqlens;

    // Dropout (none) — p_dropout is "probability of KEEPING"
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = softmax_scale;

    // Window (full attention)
    p.window_size_left = -1;
    p.window_size_right = -1;
    p.softcap = 0.0f;

    // Control flags — match varlen path exactly
    p.is_bf16 = true;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.is_rotary_interleaved = false;
    p.num_splits = 0;          // 0 or 1 → normal (non-split) kernel path
    p.page_block_size = 1;     // unused but match varlen default
    p.unpadded_lse = true;     // varlen uses unpadded LSE
    p.seqlenq_ngroups_swapped = false;

    g_run_mha_fn(p, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "thin_wrapper: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int thin_fa2_launch(
    void *q, void *k, void *v, void *out,
    int seqlen, int nheads, int headdim,
    float softmax_scale,
    int *d_cu_seqlens,
    cudaStream_t stream)
{
    const int row_stride = nheads * headdim;
    return thin_fa2_launch_strided(q, k, v, out, seqlen, nheads, headdim,
                                   softmax_scale, d_cu_seqlens,
                                   row_stride, row_stride, row_stride,
                                   stream);
}

int thin_fa2_gqa_available(void) {
    do_init();
    return g_run_mha_h128_fn != nullptr;
}

int thin_fa2_varlen_launch(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int nheads, int headdim,
    int q_row_stride, int k_row_stride, int v_row_stride,
    int num_seqs,
    int *d_cu_seqlens,
    cudaStream_t stream)
{
    do_init();
    if (!g_run_mha_fn || headdim != 64 || total_seqlen <= 0 || nheads <= 0)
        return -1;

    float scale = 1.0f / sqrtf((float)headdim);
    int hidden = nheads * headdim;

    float *lse = ensure_lse_buf(nheads, total_seqlen);
    if (!lse) { fprintf(stderr, "thin_wrapper: lse alloc failed\n"); return -1; }

    Flash_fwd_params p{};

    p.q_ptr = q;  p.k_ptr = k;  p.v_ptr = v;
    p.o_ptr = out;

    p.q_row_stride = q_row_stride;
    p.k_row_stride = k_row_stride;
    p.v_row_stride = v_row_stride;
    p.q_head_stride = headdim;
    p.k_head_stride = headdim;
    p.v_head_stride = headdim;
    p.o_row_stride = hidden;
    p.o_head_stride = headdim;

    p.softmax_lse_ptr = lse;

    p.b = num_seqs;
    p.h = nheads;
    p.h_k = nheads;
    p.h_h_k_ratio = 1;
    p.seqlen_q = max_seqlen;
    p.seqlen_k = max_seqlen;
    p.seqlen_q_rounded = round_up(max_seqlen, 128);
    p.seqlen_k_rounded = round_up(max_seqlen, 128);
    p.d = headdim;
    p.d_rounded = headdim;
    p.total_q = total_seqlen;

    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale * (float)M_LOG2E;

    p.cu_seqlens_q = d_cu_seqlens;
    p.cu_seqlens_k = d_cu_seqlens;

    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = scale;

    p.window_size_left = -1;
    p.window_size_right = -1;
    p.softcap = 0.0f;

    p.is_bf16 = true;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.is_rotary_interleaved = false;
    p.num_splits = 0;
    p.page_block_size = 1;
    p.unpadded_lse = true;
    p.seqlenq_ngroups_swapped = false;

    g_run_mha_fn(p, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "thin_wrapper varlen: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int thin_fa2_gqa_launch(
    void *q, void *k, void *v, void *out,
    int total_seqlen, int max_seqlen,
    int q_heads, int kv_heads, int headdim,
    int q_row_stride, int k_row_stride, int v_row_stride,
    int num_seqs,
    int *d_cu_seqlens,
    int is_causal,
    cudaStream_t stream)
{
    do_init();
    if (!g_run_mha_h128_fn || headdim != 128 || total_seqlen <= 0 || q_heads <= 0)
        return -1;

    float scale = 1.0f / sqrtf((float)headdim);
    int q_ratio = q_heads / kv_heads;
    int hidden = q_heads * headdim;

    float *lse = ensure_lse_buf(q_heads, total_seqlen);
    if (!lse) { fprintf(stderr, "thin_wrapper: lse alloc failed\n"); return -1; }

    Flash_fwd_params p{};

    p.q_ptr = q;  p.k_ptr = k;  p.v_ptr = v;
    p.o_ptr = out;

    p.q_row_stride = q_row_stride;
    p.k_row_stride = k_row_stride;
    p.v_row_stride = v_row_stride;
    p.q_head_stride = headdim;
    p.k_head_stride = headdim;
    p.v_head_stride = headdim;
    p.o_row_stride = hidden;
    p.o_head_stride = headdim;

    p.softmax_lse_ptr = lse;

    p.b = num_seqs;
    p.h = q_heads;
    p.h_k = kv_heads;
    p.h_h_k_ratio = q_ratio;
    p.seqlen_q = max_seqlen;
    p.seqlen_k = max_seqlen;
    p.seqlen_q_rounded = round_up(max_seqlen, 128);
    p.seqlen_k_rounded = round_up(max_seqlen, 128);
    p.d = headdim;
    p.d_rounded = round_up(headdim, 32);
    p.total_q = total_seqlen;

    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale * (float)M_LOG2E;

    p.cu_seqlens_q = d_cu_seqlens;
    p.cu_seqlens_k = d_cu_seqlens;

    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = scale;

    p.window_size_left = -1;
    p.window_size_right = is_causal ? 0 : -1;
    p.softcap = 0.0f;

    p.is_bf16 = true;
    p.is_causal = is_causal ? true : false;
    p.is_seqlens_k_cumulative = true;
    p.is_rotary_interleaved = false;
    p.num_splits = 0;
    p.page_block_size = 1;
    p.unpadded_lse = true;
    p.seqlenq_ngroups_swapped = false;

    g_run_mha_h128_fn(p, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "thin_wrapper gqa: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

}
