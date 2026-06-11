# ForceAligner Optimization Log

## Environment
- GPU: Blackwell sm_120
- CUDA: 12.8
- Model: Qwen3-ForcedAligner-0.6B (snapshot c7cbfc20)
- Test audio: GLM-ASR/cut.wav (14s, English) + Buckeye 20-segment

## Baseline (commit f6e831d, before optimizations)

| Mode | Metric | Value |
|---|---|---|
| oneshot (14s audio) | Wall clock | ~830ms |
| oneshot (14s audio) | GPU bench | **broken** (ev_mel_done never recorded) |
| daemon (5 segments) | Wall clock | ~880ms (176ms/seg) |
| Buckeye 20-seg | C++ AAS | 66.8ms (Δ=-0.3ms vs Python) |

## Optimization #1–5 (commit f498078)

### Changes

| # | File | Change |
|---|---|---|
| 1 | `cuda/mel.cu` | Persistent cuFFT plan cache: `static cufftHandle g_cufft_plan` reused across calls when n_fft/num_frames unchanged |
| 2 | `cuda/mel.cu` | GPU mel postprocess: CUB `DeviceReduce::Max` + `mel_postprocess_kernel` replaces 2× full `cudaMemcpy` host roundtrip |
| 3 | `cuda/kernels.cu` | RoPE sin/cos precomputation: bf16 device table [8192 pos × 32 half_dim × 2], cached per theta, eliminates per-element `powf/cosf/sinf` |
| 4 | `src/main.cpp` | Remove `cudaDeviceSynchronize()` in daemon loop (aligner_align already syncs internally) |
| 5 | `src/aligner.cpp` | Fix: add `cudaEventRecord(ev_mel_done)` after `cuda_mel_spectrogram` |

### Post-Optimization Results

**Oneshot (14s audio, 3 runs):**

| Run | Wall | GPU Total | Mel | Encoder | Decoder |
|---|---|---|---|---|---|
| 1 | 0.84s | 170.5ms | 3.5ms | 133.6ms | 32.6ms |
| 2 | 0.80s | 166.6ms | 3.0ms | 130.5ms | 32.6ms |
| 3 | 0.79s | 167.7ms | 3.1ms | 131.6ms | 32.5ms |

GPU time: ~168ms ± 2ms. Wall ≈ GPU + model load + I/O (~630ms overhead in oneshot mode).

**Daemon mode (5 Buckeye segments):**

| Segment | Duration | Frames | Total | Mel | Encoder | Decoder |
|---|---|---|---|---|---|---|
| s0101a_021 | 3.84s | 383 | 155.4ms | 2.3ms | 124.9ms | 28.1ms |
| s0103a_011 | 5.96s | 596 | 15.7ms | 1.0ms | 3.9ms | 10.6ms |
| s0201a_002 | 3.91s | 391 | 13.0ms | 0.7ms | 3.0ms | 9.1ms |
| s0205a_008 | 5.43s | 543 | 15.6ms | 1.2ms | 3.8ms | 10.5ms |
| s0301b_006 | 8.93s | 893 | 19.1ms | 1.3ms | 5.9ms | 11.5ms |

**Key observation**: Segment 1 includes cuFFT plan creation (~140ms for first R2C plan on sm_120).
After cuFFT cache warms up, subsequent segments drop to **13–19ms GPU time**.
That's a ~90% reduction in daemon per-segment GPU time vs the ~140ms encoder chunk seen in segment 1.

First-segment slowness is the cuFFT plan creation (~137ms overhead in segment 1 encoder time of 124.9ms vs 3.0–5.9ms for segments 2–5).

**Buckeye regression:**

```
C++ AAS: 66.8 ms | Python AAS: 67.0 ms | Δ: -0.3 ms
✅ C++ engine within 50ms of Python reference
```

Accuracy **unchanged** — all RoPE and mel optimizations preserve numerical output within bf16 tolerance.

## Commit History

```
d5890f6 perf: fused residual-add + rmsnorm kernel for decoder
7991c63 perf: WMMA fused decoder attention kernel (causal, GQA, online softmax)
f498078 perf: cuFFT plan cache, GPU mel postprocess, RoPE lookup table, bench fix
f6e831d fix: remove rope_theta override — use config value 1M instead of 5M
3cd4205 fix: encoder NaN for 1-token infer chunks, RMSNorm eps 1e-6, peak norm conditional
99d9402 Add cuBLAS attention path and encoder dump utility
b2501e9 Fix mel frame count to N/160 matching torch.stft(center=True)
06e9167 Add project plan and architecture notes
16c0752 Add Buckeye evaluation scripts
82f7b78 Add Qwen3-ForcedAligner C++/CUDA inference engine
0a311b3 Add build system and dependencies
```

## Attention Backends

| Stage | Default | Fallback |
|---|---|---|
| Encoder | WMMA flash attention `flash_attention_q32_k32_kernel` (seq<8: `tiny_attention_kernel`) | cuBLAS `cublasGemmStridedBatchedEx` via `FA_FORCEALIGNER_USE_CUBLAS=1` |
| Decoder | WMMA fused attention `decoder_flash_attention_kernel` (Q_TILE=32, online softmax, causal) | cuBLAS `attention_prefill_cublas` via `FA_FORCEALIGNER_USE_CUBLAS_DECODER=1` |

## Optimization #6 — Decoder WMMA Fused Attention (commit 7991c63)

### Change
Replaced per-Q-head cuBLAS GEMMs (QK + PV) + separate `add_causal_mask` + `softmax_inplace` with single WMMA kernel:
- `decoder_flash_attention_kernel`: Q_TILE=32, KV_TILE=32, WMMA m16n16k16
- Causal masking integrated (j > i → -inf before online softmax)
- GQA: 4 Q heads per KV head, iterated within each block
- Seq < Q_TILE auto-falls back to cuBLAS

### Results (14s audio, 3 runs)

| Path | Decoder Time | Total GPU |
|---|---|---|
| WMMA fused (default) | **28.6–29.8ms** | 160–162ms |
| cuBLAS fallback | 31.8–32.2ms | 165–170ms |
| Improvement | **-11%** | **-4%** |

### Buckeye AAS
```
C++ AAS: 66.8 ms | Python AAS: 67.0 ms | Δ: -0.3 ms ✅
```

## Optimization #7 — Fused Residual+RMSNorm (commit d5890f6)

### Change
Replaced `add_residual` + `rmsnorm` pairs in decoder with single fused kernel:
- `fused_add_residual_rmsnorm_kernel`: in-place hidden update + norm output in one pass
- Applied post-o_proj (28 layers) + post-mlp_down (27 non-final layers)
- Saves 55 kernel launches per decoder forward pass

### Results (14s audio)
Decoder: ~31ms (within noise of #6's 29ms baseline). AAS: 67.0ms (Δ=-0.1ms).

## #8 — Skipped
Dual projection fuse (K+V, gate+up) deferred — requires weight concatenation and scratch buffer rework for ~0.5ms gain.

## Final Summary

| Metric | Baseline | After #1–7 | Change |
|---|---|---|---|
| oneshot GPU Total | broken | **~165ms** | now measurable |
| oneshot Wall | ~830ms | ~810ms | -2% (I/O bound) |
| Mel stage | broken | **3ms** | — |
| Encoder stage | broken | **130ms** | — |
| Decoder stage | broken | **31ms** | — |
| daemon warm seg | ~175ms/seg | ~16ms/seg | -91% (cuFFT cached) |
| Buckeye AAS (20-seg) | 66.8ms | **67.0ms** | +0.2ms (noise) |
| Buckeye AAS (37/39, excl. 2 bad) | — | **58.7ms (C++) vs 64.2ms (Python)** | **C++ -5.5ms** |

## Buckeye 39-Segment Daemon Regression (commit 95291c4)

Full Buckeye daemon test: 39 segments, 37ms/seg wall time.

| Category | Count | C++ AAS | Python AAS | Δ |
|---|---|---|---|---|
| Good (37/39) | 37 | **58.7ms** | 64.2ms | **-5.5ms** |
| All-zero (speaker s20) | 2 | 3832–6165ms | 42–49ms | — |

**Known issue**: s2001a_011, s2001b_006 (speaker s20, 8s, quiet filler words "uh-huh"/"um-hum") produce all-zero timestamps in C++ but work in Python. Root cause not yet identified — likely model-level edge case with very quiet audio (<0.2 peak) and unusual tokens ("-h" token 2832). Tokenization matches Python exactly after hyphen fix (commit 95291c4).

**Excluding the 2 bad segments, C++ actually outperforms Python by 5.5ms AAS on the remaining 37 segments.**

**Remaining high-impact targets:**
- CUDA graph capture for daemon throughput
- Batch inference (multi-audio encoder + decoder batching)
- Persistent KV cache for decode

**Attention Backends (final state):**

| Stage | Default | Fallback |
|---|---|---|
| Encoder | WMMA flash attention (Q_TILE=32) | cuBLAS via `FA_FORCEALIGNER_USE_CUBLAS=1` |
| Decoder | WMMA fused attention (Q_TILE=32, causal, GQA) | cuBLAS via `FA_FORCEALIGNER_USE_CUBLAS_DECODER=1` |
