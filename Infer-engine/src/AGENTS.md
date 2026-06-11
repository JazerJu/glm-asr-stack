# src

## OVERVIEW

C/CUDA-facing host implementation of model loading, scheduler, KV pool management, engine orchestration, and CLI/audio ingestion.

## STRUCTURE

- `main.c`: CLI entry, `--daemon` stdin/stdout protocol, `--oneshot`, batch inference, ffmpeg audio decode for non-WAV/PCM formats.
- `engine.c`: pipeline orchestration, `encode_audio` to `prefill_sequence` to `decode_batch` to `finish_sequence`.
- `model.c`: `config.json` parser with brace-depth tracking, model init, KV pool init, cuBLAS handle setup.
- `scheduler.c`: prefill-first scheduler with preemption, block table management, token budgeting.
- `kv_pool.c`: paged KV block pool, init/free/alloc_block, slot mapping, warmup with 800MB headroom.
- `sequence.c`: sequence lifecycle, init/set_prompt/free, prompt token and audio embed storage.
- `safetensors.c`: safetensors parser, JSON header extraction, mmap data section to GPU tensors.
- `tokenizer.c`: BPE tokenizer, `vocab.json`, `added_tokens`, byte-level fallback decode.
- `audio_io.c`: audio file loading, raw PCM `.bin` via `load_pcm_f32_file`, `.wav` via `load_wav_file` (dr_wav), dispatched by extension in `load_audio_file`.

## WHERE TO LOOK

- Pipeline flow changes: `engine.c`, especially `encode_audio`, `prefill_sequence`, `decode_batch`.
- Scheduler policy: `scheduler.c`, prefill-first path and preemption logic.
- KV memory management: `kv_pool.c`, block alloc/free and warmup probing.
- Config parsing: `model.c`, `model_config_from_json` with brace-depth tracking.
- Audio input handling: `audio_io.c` for `.wav` and raw PCM loading; `main.c` for ffmpeg fallback (`.mp3` etc.).

## CONVENTIONS

- `src/` is mostly C99, but some files include CUDA runtime or cuBLAS headers directly.
- Defensive NULL checks on pointer arguments, especially in `scheduler.c`, `kv_pool.c`, `sequence.c`.
- CUDA headers are included by absolute path in some translation units, for example `/usr/local/cuda-12.8/include/cuda_runtime.h` in `main.c` and `engine.c`.
- `fprintf(stderr, ...)` is the only logging path in `src/`.
- Fatal host-side failures exit through `CHECK()` or direct `fprintf` plus `exit`.
- Use the allocation style already present in the file you are editing. `main.c` already uses `realloc` for ffmpeg decode buffering.
- GPU allocations use direct `cudaMalloc` and `cudaFree`, no wrapper ownership layer.
- Include what the file actually needs. Do not add unrelated project headers just to satisfy a global include order.
- `audio_io.c` includes `../third_party/dr_wav.h` with `DR_WAV_IMPLEMENTATION` defined (single-header pattern).

## VERIFICATION

- `main.c`: if audio loading changed, test both raw PCM, `.wav`, and an `ffmpeg` path such as `.mp3`.
- `model.c`: validate on a real `config.json`; parser changes are easy to get wrong with nested sections.
- `engine.c`: prefer an end-to-end smoke test because the orchestration code touches most subsystems.

## ANTI-PATTERNS

- CPU softmax loops. Use the CUDA `softmax_inplace` path.
- `parse_nested_int` without brace tracking. Nested JSON exits early.
- Reusing one shared `char *` cursor across config sections. Each parser branch needs its own `const char *` cursor.
- Copying outdated compile commands from docs without checking file locations and CUDA header dependencies.
