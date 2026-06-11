# cuda/

## OVERVIEW

CUDA C++ kernel implementations for mel spectrogram, encoder, decoder, paged KV attention, and optional NVFP4 CUTLASS GEMM.

## STRUCTURE

- `kernels.cu` , Core kernels: fp4_dequant, rmsnorm, layernorm, rope, gelu/silu, bf16_linear, fp4_linear, conv1d, softmax, embedding_lookup, concat_4frames, transpose_2d, residual_add.
- `kernels.cuh` , CUDA-internal header with typed declarations, `__nv_bfloat16*`, `cublasHandle_t`, `WeightStore*`, `ModelConfig*`.
- `mel.cu` , Mel spectrogram path: cuFFT R2C, Hann window, mel filterbank, persistent `__device__` filter buffer.
- `encoder.cu` , 32-layer bidirectional encoder with batched GEMM attention and FP4 weights.
- `decoder.cu` , 28-layer Llama decoder with GQA (16q/4kv heads), FP4, legacy KV, paged KV, and batched decode paths.
- `paged_kv.cu` , Paged KV kernels: `reshape_and_cache` for slot-mapped writes, `paged_decode_attention` for block-table scans.
- `nvfp4_cutlass.cu` , Native NVFP4 CUTLASS GEMM path, cuBLASLt FP4 GEMM, bf16→fp4 quantization kernel, opt-in.

## WHERE TO LOOK

- Add an elementwise kernel in `kernels.cu`, follow the `__global__` kernel plus host wrapper pattern.
- Add mel processing in `mel.cu`.
- Modify encoder attention in `encoder.cu`.
- Modify a decoder layer in `decoder.cu`, largest CUDA file, about 1134 lines.
- Tune paged attention in `paged_kv.cu`.
- Tune NVFP4 GEMM in `nvfp4_cutlass.cu`.

## CONVENTIONS

- Pair every host launcher with a device kernel: `kernel_name_kernel` on device, `kernel_name` on host.
- `#define WARP_SIZE 32` is repeated locally in `kernels.cu`, `mel.cu`, and `paged_kv.cu`.
- Shared memory is static-size only, used for reductions and scratch, no `extern __shared__` pattern here.
- Warp reductions use `__shfl_down_sync`, especially in `paged_kv.cu` and `kernels.cu`.
- `extern "C"` style varies by file: whole-file in `kernels.cu`, section block in `decoder.cu`, per-function in `mel.cu`, `paged_kv.cu`, `nvfp4_cutlass.cu`.
- `decoder.cu` keeps static GPU scratch buffers like `d_dec_q_buf`, `d_dec_k_buf`, allocated on first use.
- cuBLAS calls use `CUBLAS_COMPUTE_32F` with float `alpha` and `beta`.
- NVFP4 path stays behind `prefer_nvfp4_cutlass()`, which reads `GLMASR_USE_CUTLASS_NVFP4`.

## VERIFICATION

- After touching a CUDA entry point exported to C, compare the signature in `cuda/kernels.cuh` with `include/cuda_kernels.h`.
- After touching kernel math or memory layout, run `make` and prefer at least one inference smoke test on real weights.
- Use `compute-sanitizer --tool memcheck` when changing paged KV layout, slot mapping, or custom cache writes.
- NVFP4 path changes should be tested with and without `GLMASR_USE_CUTLASS_NVFP4=1`; optional one-shot comparison is behind `GLMASR_NVFP4_COMPARE_ONCE`.

## ANTI-PATTERNS

- Don't use device builtin `warpSize`, separate `.cu` compilation breaks linkage, use local `WARP_SIZE 32`.
- Don't move softmax or similar reduction loops to host code, watchdog timeouts follow, use `softmax_inplace`.
- Don't set cuBLAS compute type to `CUDA_R_16BF`, results go wrong, keep `CUBLAS_COMPUTE_32F`.
- Don't change an exported CUDA signature in only one header. C ABI drift here is easy to miss and hard to debug.
