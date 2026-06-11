# GLM-ASR Inference Engine

## OVERVIEW

Pure CUDA C inference engine for `GLM-ASR-Nano-2512` (speech-to-text). Loads BF16 safetensors weights, processes PCM/WAV audio through mel spectrogram → encoder → decoder pipeline, outputs transcribed text. Target: RTX 50 series (Blackwell SM120). Best benchmark results use BF16 weights (NVFP4 path available but slower without w4a16 quantization).

This file should stay focused on repo-wide guidance. Put directory-specific detail in `src/AGENTS.md`, `cuda/AGENTS.md`, and `include/AGENTS.md`.

## STRUCTURE

```
Infer-engine/
├── CMakeLists.txt        # CUDA 12.8 CMake build, GLMASR_CUDA_ARCH selects sm_120/sm_89/sm_86
├── transcribe_librispeech.py  # Public benchmark/transcription entry point
├── tools/
│   ├── eval_batch_test.py    # Internal mixed-length benchmark helper
│   └── eval_batch_pad30.py   # Internal pad30 benchmark/data-prep helper
├── third_party/
│   └── dr_wav.h          # Single-header WAV decoder (dr_wav implementation in audio_io.c)
├── include/              # C ABI headers + dual-ABI bridge (see include/AGENTS.md)
│   ├── types.h           # Core types: Tensor, ModelConfig, WeightStore, KVCache, bf16_t
│   ├── cuda_kernels.h    # C-compatible extern declarations (uint16_t* for bf16)
│   ├── engine.h          # Engine struct + submit/step/transcribe API
│   ├── model.h           # Model struct + load/free
│   ├── scheduler.h       # Prefill-first scheduler (nano-vllm style)
│   ├── sequence.h        # Per-request sequence state + block table
│   ├── kv_pool.h         # Paged KV cache pool (block-based, vLLM layout)
│   ├── tokenizer.h       # BPE tokenizer load/decode
│   ├── safetensors.h     # Safetensors loader API
│   └── audio_io.h        # Audio file loading: PCM .bin + WAV via dr_wav
├── src/                  # C implementation (see src/AGENTS.md)
│   ├── main.c            # CLI entry: --daemon mode + --oneshot mode + batched inference
│   ├── engine.c          # Pipeline orchestration: mel→conv→enc→concat→proj→prefill→decode
│   ├── model.c           # config.json parser, model + KV pool initialization
│   ├── scheduler.c       # Prefill-first scheduler with preemption
│   ├── kv_pool.c         # Paged KV block pool: alloc/free/slot mapping/warmup
│   ├── sequence.c        # Sequence lifecycle (init/set_prompt/free)
│   ├── safetensors.c     # Safetensors parser: JSON header + mmap data → GPU tensors
│   ├── tokenizer.c       # BPE tokenizer: vocab + added_tokens + byte-level decode
│   └── audio_io.c        # Audio loading: raw PCM + WAV (dr_wav), dispatches by extension
└── cuda/                 # CUDA kernels (see cuda/AGENTS.md)
    ├── kernels.cu        # Core kernels: fp4_dequant, rmsnorm, rope, gelu/silu, bf16/fp4_linear, conv1d
    ├── kernels.cuh       # CUDA-internal header (__nv_bfloat16*, cublasHandle_t typed)
    ├── mel.cu            # Mel spectrogram (cuFFT R2C, window, mel filterbank)
    ├── encoder.cu        # 32-layer bidirectional encoder (batched GEMM attention)
    ├── decoder.cu        # 28-layer Llama decoder (GQA, FP4, legacy + paged KV paths)
    ├── paged_kv.cu       # Paged KV reshape_and_cache + paged_decode_attention kernels
    └── nvfp4_cutlass.cu  # Native NVFP4 CUTLASS GEMM path (opt-in via env var)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add a new CUDA kernel | `cuda/kernels.cu` + declare in `cuda/kernels.cuh` + `include/cuda_kernels.h` | Must follow dual-ABI pattern |
| Modify model pipeline | `src/engine.c` | encode_audio → prefill_sequence → decode_batch |
| Change scheduler policy | `src/scheduler.c` | Prefill-first with preemption |
| Add/remove model weights | `src/model.c` | ws_get() lookup by name |
| Parse new config field | `src/model.c` → `model_config_from_json` | Brace-depth tracking required |
| Fix tokenizer issue | `src/tokenizer.c` | added_tokens (ids 59247-59263) still broken |
| Add paged attention optimization | `cuda/paged_kv.cu` | Block-based layout, warp shuffle reductions |
| Enable native NVFP4 GEMM | `cuda/nvfp4_cutlass.cu` | Set `GLMASR_USE_CUTLASS_NVFP4=1` |
| Change build flags/arch | `CMakeLists.txt` | CUDA 12.8, `GLMASR_CUDA_ARCH` cache variable |
| Add audio format support | `src/audio_io.c` + `include/audio_io.h` | WAV via dr_wav, raw PCM fallback |

## PREREQUISITES

- CUDA toolkit is expected at `/usr/local/cuda-12.8/`; `CMakeLists.txt` exposes this as `CUDA_ROOT`.
- Build target is `sm_120`; this repo is currently tuned for RTX 50 series / Blackwell.
- `ffmpeg` is only required when the audio input is not raw `.bin` / `.wav` / float PCM. `.mp3` and other formats go through `ffmpeg` in `src/main.c`; `.wav` files are decoded natively via `dr_wav`.
- Runtime validation that exercises CUDA paths needs a working NVIDIA driver plus a GPU that supports the compiled arch.

## CODE MAP

### Key Types (include/types.h)

| Symbol | Role |
|--------|------|
| `Tensor` | GPU buffer + dtype + shape |
| `ModelConfig` | All hyperparams (enc/dec dims, vocab, special tokens) |
| `WeightStore` | Name→Tensor lookup (linear scan, first-char fast reject) |
| `KVCache` | Legacy contiguous per-layer KV (backward compat) |
| `KVPool` | Paged block-based KV pool (vLLM-style) |
| `BlockTable` | Per-sequence block mapping into KVPool |
| `Sequence` | Per-request state (prompt, audio embeds, generation, block table) |
| `Scheduler` | Prefill-first batch scheduler with preemption |
| `Engine` | Top-level orchestrator: model + scheduler + workspace buffers |

### Key Functions (src/engine.c)

| Symbol | Role |
|--------|------|
| `engine_init` | Load model, tokenizer, mel filters, warmup KV pool |
| `engine_submit` | Queue audio for transcription |
| `engine_step` | Run one scheduler step (prefill or decode batch) |
| `engine_transcribe` | Convenience: full pipeline → text string |
| `engine_reset_batch` | Clear batch state between runs |
| `engine_free` | Release all GPU/host resources |
| `mixed_batch` | Internal: dispatch prefill vs decode per-sequence |
| `materialize_sequence_prompt` | Internal: encode audio + build prompt tokens |
| `decode_generated_text` | Internal: token IDs → string via tokenizer |

### Key CUDA Entry Points

| Symbol | File | Role |
|--------|------|------|
| `mel_spectrogram` | mel.cu | PCM → [128, T] bf16 mel |
| `conv1d_forward` | kernels.cu | cuBLAS im2col-based 1D conv |
| `encoder_forward` | encoder.cu | 32-layer bidirectional encoder |
| `decoder_prefill` | decoder.cu | Full-sequence decoder prefill |
| `decoder_decode_step` | decoder.cu | Single-token decode step |
| `decoder_prefill_paged` | decoder.cu | Prefill with paged KV |
| `decoder_decode_step_paged` | decoder.cu | Single-token decode with paged KV |
| `decoder_decode_step_batched` | decoder.cu | Multi-sequence batched decode |
| `paged_decode_attention` | paged_kv.cu | Block-table-based attention kernel |
| `reshape_and_cache` | paged_kv.cu | Write new KV to paged cache via slot mapping |
| `nvfp4_linear` | nvfp4_cutlass.cu | Native NVFP4 CUTLASS GEMM (opt-in) |
| `fp4_linear` | kernels.cu | Dequant-to-bf16 then cuBLAS fallback path |

## CONVENTIONS

### Dual-ABI Header Pattern (CRITICAL)
- **CUDA→CUDA calls**: use `cuda/kernels.cuh` (typed: `__nv_bfloat16*`, `cublasHandle_t`, `WeightStore*`)
- **C→CUDA calls**: use `include/cuda_kernels.h` (opaque: `uint16_t*`, `void*` handles)
- Bridge: `include/types.h` defines `bf16_t` → `__nv_bfloat16` under `__CUDACC__`, `uint16_t` otherwise

### Naming
- All project functions/kernels: `snake_case`
- No camelCase except CUDA/cuBLAS APIs
- Kernel suffix: `_kernel` (e.g., `fp4_dequant_kernel`)
- Host launchers: same name without suffix (e.g., `fp4_dequant`)

### Error Handling
- `CHECK(cond, fmt, ...)` — CPU assertion with fprintf+exit (types.h)
- `CUDA_CHECK(call)` — GPU error check with cudaGetErrorString+exit (CUDA only)
- Extensive `fprintf(stderr, ...)` logging throughout (58 call sites) — mix of errors, warnings, and debug

### Coding Style
- C99 for .c files, CUDA C++ for .cu files
- `#define WARP_SIZE 32` instead of device builtin `warpSize` (linkage issue workaround)
- `extern "C"` wrapping required for all CUDA functions called from C
- Three extern "C" patterns: whole-file block, section block, per-function
- No `.clang-format` or `.editorconfig` — follow surrounding code in each file
- Prefer rules that match the current code over aspirational style rules. If a local `AGENTS.md` conflicts with the code, verify the code before propagating the rule.

## ANTI-PATTERNS (THIS PROJECT)

1. **CPU softmax loops** — Caused GPU watchdog timeouts. Always use `softmax_inplace` CUDA kernel.
2. **cuBLAS compute type `CUDA_R_16BF`** — Use `CUBLAS_COMPUTE_32F` with float alpha/beta.
3. **Naive conv1d triple loops** — Watchdog timeout. Use im2col + cuBLAS GEMM (already done).
4. **`warpSize` device builtin** — Linkage error across separate compilation. Use `#define WARP_SIZE 32`.
5. **`parse_nested_int` without brace tracking** — Exits early on nested JSON objects. Track depth.
6. **Modifying shared pointers in config parser** — Each section needs its own `const char *` cursor.
7. **Hardcoded CUDA paths** — prefer `CMakeLists.txt` cache variables such as `CUDA_ROOT`.

## KNOWN ISSUES

- `tokenizer.c`: `parse_added_tokens` fails — 17 special tokens (ids 59247-59263) missing
- `include/cuda_kernels.h`: Missing `decoder_decode_step_batched` declaration (engine.c uses manual extern)
- `cuda/kernels.cuh`: Missing `mel_cleanup` declaration (exists in cuda_kernels.h + mel.cu)
- NVFP4 CUTLASS path is experimental — controlled by `GLMASR_USE_CUTLASS_NVFP4` env var

## LOW-ROI / REVERTED EXPERIMENTS

- Small operator fusions that were tried and did **not** produce meaningful end-to-end gains:
  - `Conv1d + GELU` fusion in the audio front-end
    - roughly `192.0x -> 192.1x` (noise-level; effectively no gain)
  - `RMSNorm + Residual Add` fusion on decoder hot paths
    - roughly `295.4x -> 294.0x` (no gain; slight regression)
  - Moving decoder `SiLU` into the linear epilogue
    - roughly `288.3x -> 287.7x` (no useful gain)
  - Letting decoder `down_proj` read fused packed buffers directly via strided input
    - roughly `301.5x -> 300.4x` (regression)
  - Letting batched prefill `down_proj` read fused packed buffers directly via strided input
    - roughly `301.5x -> 299.8x` (regression)
- Medium-granularity projection/dataflow rewrites that were tried and then reverted:
  - `QKV` triple same-input projection fusion that writes a packed buffer and then splits it back to Q/K/V
    - roughly `300.1x -> 296.4x` (clear regression)
  - Async stream overlap for paged-KV cache writes
    - roughly `295.4x -> 232.5x`, and output stability also worsened
- First encoder "flash attention" prototype was also reverted:
  - it used a naive online-softmax fused kernel
  - it was functionally acceptable but much slower than the cuBLAS-based encoder attention path
  - roughly `301.5x -> 107.1x` on the full benchmark
- Practical guidance:
  - prefer keeping these as historical notes rather than reopening them early
  - remaining high-value work is in larger structural changes, especially full attention-kernel upgrades and BF16/FP4 path unification

## ENVIRONMENT VARIABLES

| Variable | Default | Effect |
|----------|---------|--------|
| `GLMASR_USE_CUTLASS_NVFP4` | disabled | Enable native NVFP4 CUTLASS GEMM (vs bf16 dequant fallback) |
| `GLMASR_NVFP4_COMPARE_ONCE` | disabled | Compare CUTLASS vs fallback output once, log diffs |
| `GLMASR_BF16_LINEAR_LT_BIAS` | `1` | Use cuBLASLt bias epilogue for `bf16_linear(..., bias)`; set `0` to fall back to cuBLAS GEMM plus separate bias kernel |
| `GLMASR_BF16_LINEAR_TENSOR_OP` | `0` | Diagnostic: force `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for bf16 cuBLAS GEMMs; A/B showed no measurable gain over default |
| `GLMASR_ENCODER_QKV_FUSED` | `1` | Encoder Q32_K32 path: fuse Q/K/V projection into one packed `[rows, 3H]` GEMM, apply batched packed RoPE once per layer, and let FA read packed Q/K/V directly via row stride; set `0` to fall back to separate Q/K/V projections |
| `GLMASR_ENCODER_FA_ENABLE` | `1` | Master switch for encoder FA path |
| `GLMASR_ENCODER_FA_FORCE_FALLBACK` | `0` | Force encoder attention to use cuBLAS fallback |
| `GLMASR_ENCODER_FA_MIN_SEQ` | `1500` | Legacy FA min-seq gate; used as medium gate default |
| `GLMASR_ENCODER_FA_MEDIUM_MIN_SEQ` | inherits `GLMASR_ENCODER_FA_MIN_SEQ` | Seq threshold for FA-medium (simple kernel) |
| `GLMASR_ENCODER_FA_LONG_MIN_SEQ` | `1500` | Seq threshold for FA-long (pipelined kernel), clamped >= medium |
| `GLMASR_ENCODER_FA_PIPELINED` | `0` | Enable FA-long (pipelined dynamic-smem kernel) |
| `GLMASR_ENCODER_FA_MMA_SCORE` | `0` | Experimental FA-long variant: Tensor Core MMA score + cp.async K/V staging |
| `GLMASR_ENCODER_FA_WS_TMA` | `0` | Experimental FA-long variant: warp-specialized producer/consumer pipeline (QK MMA producer + online-softmax/PV consumers) |
| `GLMASR_ENCODER_FA_PV_MMA` | `0` | Experimental FA-long variant: QK and PV both use Tensor Core WMMA; keeps output as unnormalized online-softmax sum until final divide |
| `GLMASR_ENCODER_FA_PV_MMA_Q32` | `0` | Experimental FA-long variant: 32-row Q tile, 8-warps/CTA, QK and PV both on Tensor Core WMMA |
| `GLMASR_ENCODER_FA_PV_MMA_Q32_K16_REG` | `0` | Experimental FA-long variant: Q32 PV-MMA with K/V tile=16 and register-resident PV accumulators; retained as an occupancy comparison point |
| `GLMASR_ENCODER_FA_PV_MMA_Q32_K32` | `0` | Experimental FA-long variant: Q32 PV-MMA with K/V tile=32, cp.async staging, register-resident Q/PV fragments, no shared score staging, and direct fragment writeback; current best broad-coverage FA path |
| `GLMASR_ENCODER_FA_PV_MMA_Q32_TMA` | `0` | Experimental FA-long variant: Q32 PV-MMA plus real TMA K/V producer warp and consumer-only barriers; currently slower than Q32 baseline, kept for TMA-WS profiling |
| `GLMASR_ENCODER_FA_FA2` | `0` | Experimental FA-long variant: FA2-style 64-row Q tile with 128-row K/V tile, 8-warps/CTA, WMMA QK+PV, cp.async staging, and strided packed-QKV support for long audio |
| `GLMASR_ENCODER_FA_PV_MMA_2PASS` | `0` | Negative experiment hook: two-pass final-m/l PV-MMA; kept off because it is slower than online PV-MMA |
| `GLMASR_ENCODER_FA_COMPARE_ONCE` | `0` | One-shot FA vs cuBLAS numerical diff check |

## COMMANDS

```bash
cmake -S . -B build_cmake -DCUDA_ROOT=/usr/local/cuda-12.8 -DGLMASR_CUDA_ARCH=sm_120
cmake --build build_cmake -j
./glm_asr_infer --daemon              # Daemon mode (stdin/stdout protocol)
./glm_asr_infer --model DIR --audio FILE --mel-filters FILE  # One-shot mode
compute-sanitizer --tool memcheck ./glm_asr_infer ...  # CUDA memory check
```

Canonical batch benchmark commands (run from repo root `Infer-engine/`):

```bash
# Model path (BF16 — best performing)
MODEL_BF16=/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots/61ba4e0b3309b6656edea3e93e419f7bd5c61957

# Full 2611-sample benchmark with FA2 thin wrapper (best: 45.9s / RTFx 417x)
GLMASR_VRAM_UTIL=0.9 \
GLMASR_MAX_SEQS=256 \
GLMASR_ENCODER_FA_PIPELINED=1 \
GLMASR_ENCODER_FA_VLLM=1 \
GLMASR_ENCODER_FA_LONG_MIN_SEQ=1 \
GLMASR_ENCODER_FA_MIN_SEQ=1 \
python3 tools/eval_batch_test.py \
  --engine ./glm_asr_infer \
  --model $MODEL_BF16 \
  --mel /data/fwsr/glm-asr/output/mel_filters.bin \
  --start 9 --num 2611

# Pad30s encoder-only benchmark (best: 11.2s / RTFx 686x vs vLLM 12.0s / 639x)
GLMASR_VRAM_UTIL=0.9 \
GLMASR_MAX_SEQS=256 \
GLMASR_MAX_NEW_TOKENS=1 \
GLMASR_ENCODER_FA_PIPELINED=1 \
GLMASR_ENCODER_FA_VLLM=1 \
GLMASR_ENCODER_FA_LONG_MIN_SEQ=1 \
GLMASR_ENCODER_FA_MIN_SEQ=1 \
python3 tools/eval_batch_pad30.py --model $MODEL_BF16

# Quick smoke test
python3 tools/eval_batch_test.py \
  --engine ./glm_asr_infer \
  --model $MODEL_BF16 \
  --mel /data/fwsr/glm-asr/output/mel_filters.bin \
  --start 9 --num 4

```

## ORCHESTRATOR / TASK QUEUE / LIVE SESSIONS

The production-facing orchestration script lives outside this repo at
`/data/fwsr/glm-asr/dir/orchestrator.py`. It owns file-level work around the
C engine:

- decode arbitrary input audio to mono 16 kHz WAV
- split long audio into <=30s chunks
- batch chunks through the `Infer-engine` daemon
- optionally run `forcealigner-engine` after ASR
- write transcript TSV plus word-level or sentence-level SRT

The C engine does **not** split audio by itself. For files longer than the
model's 30s window, use the orchestrator or an external VAD/segmenter.

### Offline one-shot

```bash
MODEL_BF16=/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots/61ba4e0b3309b6656edea3e93e419f7bd5c61957
FA_MODEL=/path/to/forcealigner/model

python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --audio /path/to/input.mp3 \
  --model "$MODEL_BF16" \
  --mel /data/fwsr/glm-asr/output/mel_filters.bin \
  --fa-model "$FA_MODEL" \
  --subtitle-mode sentence \
  --output /data/fwsr/glm-asr/dir/output/input_sentence.srt
```

Use `--subtitle-mode word` for word-level SRT, `--subtitle-mode sentence` for
sentence cues, and `--asr-only` to skip ForceAligner and only write the ASR
transcript TSV.

### Persistent Task Queue

Submit tasks without immediately running them:

```bash
python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --submit \
  --audio /path/to/input.mp3 \
  --model "$MODEL_BF16" \
  --mel /data/fwsr/glm-asr/output/mel_filters.bin \
  --fa-model "$FA_MODEL" \
  --subtitle-mode sentence \
  --output /data/fwsr/glm-asr/dir/output/input_sentence.srt
```

Run queued work:

```bash
python3 /data/fwsr/glm-asr/dir/orchestrator.py --run-queue
python3 /data/fwsr/glm-asr/dir/orchestrator.py --status <task_id>
```

Queue behavior:

- task directories live under `/data/fwsr/glm-asr/dir/tasks/<task_id>/`
- per-task progress is in `status.json`
- daemon stderr/logs are written under each task's `logs/` directory and
  `/data/fwsr/glm-asr/dir/tasks/worker_logs/`
- `--run-queue` is ASR-first: it drains queued ASR windows across tasks before
  starting ForceAligner
- `--asr-window-chunks` controls how many chunks the Python orchestrator submits
  before re-scanning the queue; it is separate from the engine's
  `GLMASR_MAX_SEQS=256`

### Live Sessions

For VAD/live-stream use, do not create one task directory per segment. Create a
single live session, append VAD chunks to it, then run or watch the session.
Session data lives under `/data/fwsr/glm-asr/dir/live_sessions/<session_id>/`.

```bash
# Create one appendable live session.
python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --live-create lecture_live \
  --model "$MODEL_BF16" \
  --mel /data/fwsr/glm-asr/output/mel_filters.bin \
  --subtitle-mode sentence \
  --output /data/fwsr/glm-asr/dir/output/lecture_live.srt

# Append VAD chunks. If --start-sec is omitted, it starts at the previous end.
python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --live-append lecture_live \
  --audio /tmp/vad_chunk_000.wav \
  --start-sec 0.0

python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --live-append lecture_live \
  --audio /tmp/vad_chunk_001.wav

# Process pending chunks once, or keep polling with --watch.
python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --run-live lecture_live \
  --live-batch-size 32

python3 /data/fwsr/glm-asr/dir/orchestrator.py \
  --run-live lecture_live \
  --watch \
  --poll-sec 0.5

python3 /data/fwsr/glm-asr/dir/orchestrator.py --live-status lecture_live
```

Live mode is ASR-only by design. It writes:

- `segments.jsonl`: appended VAD segment manifest
- `transcript.tsv`: per-segment ASR text
- the configured `.srt` output path
- `logs/orchestrator.log` and `logs/infer.stderr.log`

Live SRT timing uses the VAD segment start/end times. Sentence and word cues are
approximate within each segment; use the offline ForceAligner path when precise
word timestamps are required.

### NSYS profiling (pad30s encoder-only)

```bash
nsys profile --trace=cuda,nvtx --output=/tmp/ie_pad30_enc --force-overwrite=true \
  bash -c 'GLMASR_VRAM_UTIL=0.9 GLMASR_MAX_SEQS=256 GLMASR_MAX_NEW_TOKENS=1 \
    GLMASR_ENCODER_FA_PIPELINED=1 GLMASR_ENCODER_FA_VLLM=1 \
    GLMASR_ENCODER_FA_LONG_MIN_SEQ=1 GLMASR_ENCODER_FA_MIN_SEQ=1 \
    python3 tools/eval_batch_pad30.py --model '"$MODEL_BF16"

# Export kernel CSV
nsys stats --report cuda_gpu_kern_sum /tmp/ie_pad30_enc.nsys-rep
```

#### Encoder kernel total time comparison (pad30s, 256 audios, encoder-only)

| | IE FA2 batched (final) | IE FA2 per-seq | vLLM |
|---|---|---|---|
| FA2 kernel | 1,062 ms (416 launches) | 1,267 ms (8192 launches) | 1,100 ms (1568 launches) |
| Main MatMul | 6,450 ms | 8,168 ms | 8,506 ms |
| GPU total | ~10,700 ms | ~12,117 ms | ~10,730 ms |
| Wall time | 11.2s (RTFx 686x) |  | 12.0s (RTFx 639x) |

Key findings:
- FA2 batched varlen (commit a9b98a6): 8192 -> 416 kernel calls, saved 205ms
- GELU fusion (commit 744e614): fused into MatMul epilogue, saved ~300ms
- IE MatMul 24% faster than vLLM (6,450 vs 8,506 ms)
- IE GPU total ~= vLLM (10,700 vs 10,730 ms), IE wall time 7% faster

NSYS files archived in `nsys_rec/` and in `/tmp/`.

### Nsight Compute profiling (per-kernel stall/spill analysis)

```bash
MODEL_BF16=/data/.cache/huggingface/hub/models--zai-org--GLM-ASR-Nano-2512/snapshots/61ba4e0b3309b6656edea3e93e419f7bd5c61957
NCU=/usr/local/NVIDIA-Nsight-Compute-2025.4/ncu

# Profile varlen FA2 kernel (requires thin wrapper path — temporarily move fa2_bridge.so)
mv cutedsl_fa/fa2_bridge.so cutedsl_fa/fa2_bridge.so.bak
$NCU --kernel-name "regex:flash_fwd_kernel" --launch-skip 0 --launch-count 1 \
  --set full --csv -o nsys_rec/ncu_varlen_fa2 -f -- \
  bash -c "cd /data/fwsr/glm-asr/Infer-engine && \
    GLMASR_VRAM_UTIL=0.9 GLMASR_MAX_SEQS=256 GLMASR_MAX_NEW_TOKENS=1 \
    GLMASR_ENCODER_FA_PIPELINED=1 GLMASR_ENCODER_FA_VLLM=1 \
    GLMASR_ENCODER_FA_LONG_MIN_SEQ=1 GLMASR_ENCODER_FA_MIN_SEQ=1 \
    python3 tools/eval_batch_pad30.py --model $MODEL_BF16"
mv cutedsl_fa/fa2_bridge.so.bak cutedsl_fa/fa2_bridge.so

# Profile CuteDSL TMA kernel
$NCU --kernel-name "regex:.*TMAOptimized.*" --launch-skip 0 --launch-count 1 \
  --set basic --csv -o nsys_rec/ncu_tma_basic -f -- \
  bash -c "...same env with GLMASR_ENCODER_FA_DENSE_TMA=1..."

# Full stall/spill analysis on TMA kernel
$NCU --kernel-name "regex:.*TMAOptimized.*" --launch-skip 0 --launch-count 1 \
  --set full --csv -o nsys_rec/ncu_tma_full -f -- \
  bash -c "...same env..."

# Readable metrics summary
$NCU --import nsys_rec/ncu_varlen_fa2.ncu-rep --page details
$NCU --import nsys_rec/ncu_tma_full.ncu-rep --page details
```

**Note**: `ncu` cannot profile kernels loaded via `cuLibraryLoadData` (CuteDSL).
To profile varlen FA2, temporarily rename `fa2_bridge.so` so the binary falls
back to the thin wrapper path (which uses `cuModuleLoad`).
CuteDSL TMA kernel (`fa2_enc_dense_tma_fwd`) IS profileable via `--set basic/full`
when loaded from `fa2_bridge.so` (name filter: `regex:.*TMAOptimized.*`).

#### Profiling key findings (pad30, 256x30s, RTX 5070 Ti, 2026-05-26)

| Metric | varlen FA2 (thin) | TMA 64×80 (CuteDSL) |
|--------|-------------------|---------------------|
| Duration/launch | 3.94 ms | 3.93 ms |
| Grid | 6,000 blocks (128t) | 12,000 blocks (160t) |
| Registers/thread | 255 (19% spill to L1) | 128 (near-zero spill) |
| SM Compute | 95.78% | 94.33% |
| Memory Throughput | 22.03% | 39.23% |
| DRAM Throughput | 10.90% | 10.77% |
| L1/TEX Throughput | 15.44% | 24.72% |

Stall breakdown (warps/SM, per-issue-active cycle):

| Stall reason | varlen FA2 (8 warps/SM) | TMA (15 warps/SM) |
|-------------|------------------------|-------------------|
| math_pipe_throttle | 8.97 (59.8%) | 6.70 (44.7%) |
| long_scoreboard (mem) | 0.32 (2.1%) | 0.47 (3.1%) |
| short_scoreboard (shm) | 0.17 (1.1%) | 0.46 (3.1%) |
| selected (active) | 1.00 (6.7%) | 1.00 (6.7%) |
| sleeping (producer idle) | — | 2.50 (16.6%) |
| wait (barrier) | 2.15 (14.3%) | 2.07 (13.8%) |

Conclusions:
- Both kernels are **compute-bound** (SM 94-96%). Copy is NOT the bottleneck
  (DRAM 11%, long_scoreboard <4% of warp time).
- TMA trades register spill (varlen's 19% → near-zero) for warp-specialization
  overhead (~30% warps sleeping/waiting). Net: identical duration (3.93ms).
- Warp-specialized TMA provides zero advantage over flat-scheduled cp.async for
  this encoder shape (N=20 heads, D=64 head_dim, S=1499 sequence).
- fa2_kernel.py (custom CuteDSL varlen) ties TMA at 10.8s pad30 wall time;
  both beat the upstream vLLM thin wrapper (11.0s).
- The only remaining high-ROI optimization for fa2_kernel.py: replace
  `warp.MmaF16BF16Op` (WMMA) with hand-written `mma.sync` PTX to reduce
  register pressure (255→~180 regs, spill 19%→~5%), estimated 3-5% gain.
- Archived .ncu-rep files in `nsys_rec/`; note they are large (28MB each)
  and git-ignored.

#### MMA register-pressure experiments (2026-05-26)

Reducing `tile_n` from 128 to 64 in the custom CuteDSL FA2 kernel
(`FlashAttention2VarlenSm80`) halves the per-warp score accumulator, reducing
register pressure and eliminating spill. Comparison (both ncu-profiled on the
SAME custom kernel, not the thin wrapper):

| Config (custom fa2_kernel.py) | Regs/thr | Spill | Duration(ncu) | SM Comp |
|-------------------------------|---------|-------|---------------|---------|
| num_warps=4, tile_n=128 (orig) | 255 | 3.8M | 4.07ms | 93.01% |
| num_warps=4, tile_n=64 | 238 | **0** | 3.90ms | 96.67% |
| Delta | -7% | -100% | **-4.2%** | +3.7% |

tile_n=64 eliminates spill, improves kernel duration 4.2%, and raises SM compute
utilization to 96.67%. However, end-to-end pad30 wall time is unchanged (10.8-10.9s)
because the FA2 kernel accounts for only ~10% of total GPU time; a 4.2% kernel
improvement translates to ~0.4% end-to-end, buried in measurement noise.

Note: earlier comparison incorrectly compared the custom kernel (tile_n=64) against
the upstream vLLM thin-wrapper kernel (tile_n=128, a different binary). The data
above corrects this — both rows are from the same custom `fa2_kernel.py` binary.

#### Reproducing the builds

```bash
# Rebuild CuteDSL TMA kernel (adjust GLMASR_TMA_* env vars for config)
cd cutedsl_fa
source /data/fwsr/glm-asr/GLM-ASR/venv/bin/activate
GLMASR_TMA_TILE_M=64 GLMASR_TMA_TILE_N=80 GLMASR_TMA_NUM_STAGES=2 \
  GLMASR_TMA_USE_TMA_Q=1 python3 fa_tma.py

# Rebuild CuteDSL varlen FA2 kernel
python3 fa2_kernel.py

# Rebuild fa2_bridge.so — CRITICAL: must include libtvm_ffi.so rpath
DLPACK=$(python3 -c "import tvm_ffi; print(tvm_ffi.__path__[0])")/include
TVMFFI_DIR=$(python3 -c "import tvm_ffi; print(tvm_ffi.__path__[0])")/lib
gcc -shared -fPIC -O2 -o fa2_bridge.so \
  fa2_bridge.c override_init.o \
  fa2_enc_varlen_fwd.o fa2_dec_varlen_fwd.o fa2_enc_dense_tma_fwd.o \
  -I/usr/local/cuda-12.8/include -I$DLPACK \
  -L/usr/local/cuda-12.8/lib64 $TVMFFI_DIR/libtvm_ffi.so libpatched3.a \
  -lcuda -lcudart -ldl -lm -lpthread \
  -Wl,-rpath,$TVMFFI_DIR
```

**If `fa2_bridge.so` loads but CuteDSL falls back to thin wrapper or vLLM bridge,**
check that `-Wl,-rpath,$TVMFFI_DIR` is present — missing rpath causes silent
`libtvm_ffi.so` load failure at runtime.

## VERIFICATION

- Build system / headers: run `cmake -S . -B build_cmake ... && cmake --build build_cmake -j`.
- CUDA kernel or memory-layout changes: run the CMake build, then prefer a targeted inference smoke test; use `compute-sanitizer --tool memcheck` for suspicious memory issues.
- CLI / audio-loading changes: test one raw PCM input and one `ffmpeg`-decoded input path if the change touched `src/main.c`.
- Header ABI changes: verify both C callers in `src/` and CUDA callers in `cuda/` still compile; dual-ABI mismatches are a common failure mode here.
- Audio I/O changes: test both `.wav` and raw `.bin` PCM paths through `load_audio_file`.

## NOTES

- Weight store uses linear scan O(n) with first-char fast reject — fine for 1361 tensors
- Legacy `KVCache` (contiguous) kept alongside paged `KVPool` for backward compat
- Model requires ~2.5GB BF16 weights (fits 16GB VRAM); NVFP4 variant available but slower without w4a16 quantization
- Encoder handles up to 1500 tokens (30s audio → 3000 mel → conv stride 2)
- Max decoder sequence: 19 (text prompt) + audio_tokens + 500 (max_new_tokens)
- `.wav` files decoded natively via `dr_wav` (mono 16 kHz required); `.bin` treated as raw float32 PCM
- No CI/CD pipeline — all builds and tests are manual/local
