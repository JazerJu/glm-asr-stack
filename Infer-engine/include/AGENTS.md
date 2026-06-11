# include/

## OVERVIEW

C ABI headers exposing CUDA engine types and functions to `.c` source files, plus the dual-ABI bridge.

## STRUCTURE

- `types.h` , core types (`Tensor`, `ModelConfig`, `WeightStore`, `KVCache`) plus `bf16_t` bridge and shared macros.
- `cuda_kernels.h` , C-compatible extern declarations for CUDA entry points, `uint16_t*` for bf16-facing buffers.
- `engine.h` , `Engine` struct and submit, step, transcribe API.
- `model.h` , `Model` struct, `model_load` and `model_free`, config parser surface.
- `scheduler.h` , `Scheduler` struct, `StepType` enum, `ScheduleResult`.
- `sequence.h` , `Sequence` struct, `SeqStatus` enum, lifecycle API.
- `kv_pool.h` , `KVPool` and `BlockTable` structs, block alloc/free and slot mapping.
- `tokenizer.h` , `Tokenizer` struct, load and decode API.
- `safetensors.h` , safetensors loader, header parse plus mmap-backed data access.
- `audio_io.h` , audio file loading: `load_pcm_f32_file` (raw .bin), `load_wav_file` (via dr_wav), `load_audio_file` (dispatcher by extension).

## WHERE TO LOOK

- Add a new shared type in `types.h`, mirror existing `Tensor`, `ModelConfig`, `WeightStore` layout style.
- Expose a new CUDA function to C in `cuda_kernels.h`, keep bf16 as `uint16_t*` and handles as `void*`.
- Add engine-level state in `engine.h`, keep orchestration-facing fields there instead of leaking model internals.
- Add scheduler fields in `scheduler.h`, next to `StepType` and `ScheduleResult` flow control types.
- Add audio format support in `audio_io.h`, currently supports raw PCM `.bin` and `.wav` (mono 16 kHz).

## CONVENTIONS

- Every header uses a `#ifndef GLMASR_*_H` include guard.
- `types.h` is the universal dependency, include it before relying on project types anywhere else.
- `kv_pool.h` wraps declarations in an `extern "C"` block because both `.c` and `.cu` code include it.
- Prefer minimal includes over a fixed global include order. Include the header that owns the symbol you need, and avoid pulling in unrelated engine headers.
- `bf16_t` lives in `types.h`, maps to `__nv_bfloat16` under `__CUDACC__`, `uint16_t` otherwise.
- Keep headers ABI-facing, not CUDA-implementation-facing. Implementation-only typing belongs in `cuda/kernels.cuh`.

## VERIFICATION

- If you change `types.h`, expect rebuild impact across both `src/` and `cuda/`.
- If you change `cuda_kernels.h`, compare it against `cuda/kernels.cuh` and the corresponding definitions in `cuda/*.cu`.
- For dual-ABI changes, verify that C callers compile without CUDA-only types and CUDA callers still get the strongly typed signatures they need.

## ANTI-PATTERNS

- Don't expose `__nv_bfloat16` or `cublasHandle_t` in `include/` headers. Use `uint16_t*` and `void*` in `cuda_kernels.h`.
- Don't add CUDA-only helpers to public headers when they are only called from `.cu` files.
- `decoder_decode_step_batched` is still missing from `cuda_kernels.h`, `engine.c` carries a manual `extern` as a workaround.
- Don't enforce a fake include-order rule that causes wider coupling or unnecessary header inclusion.
