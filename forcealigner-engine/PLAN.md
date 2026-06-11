# ForceAligner C++/CUDA Inference Engine

## Overview

Pure CUDA C++ inference engine for `Qwen3-ForcedAligner-0.6B`. Loads bf16 safetensors weights,
processes PCM/WAV audio through mel spectrogram → audio encoder → text decoder → timestamp
classification pipeline, outputs word-level timestamps as SRT.

## Architecture

```
Audio (16kHz mono float32)
  → Mel Spectrogram (CUDA cuFFT, n_fft=400, hop=160, n_mels=128)
  → Audio Encoder (24-layer Whisper-style, Conv2D×3 + Transformer)
      → [seq_len, 1024] bf16 embeddings
  → Build input_ids: [audio_start] + [audio_pad × N] + [audio_end] + word<ts>word<ts>...
  → Token Embedding (152064 vocab, 1024 hidden) → replace audio_pad positions with audio embeds
  → Text Decoder (28-layer Qwen-style, GQA, RoPE, prefill-only)
      → logits [seq_len, 5000]
  → Extract timestamp logits → argmax → class × 80ms → LIS fix → (start, end) pairs
  → Output SRT
```

## Model Specs

| Component | Param |
|-----------|-------|
| Audio encoder layers | 24 |
| Audio d_model | 1024 |
| Audio attention heads | 16 |
| Audio FFN dim | 4096 |
| Audio activation | GELU |
| Audio norm | LayerNorm |
| Conv2D channels | 480 |
| Conv2D stride | 2 (×3 layers) |
| Max audio positions | 1500 |
| Text decoder layers | 28 |
| Text hidden | 1024 |
| Text FFN | 3072 |
| Text heads | 16 Q, 8 KV |
| Text head_dim | 128 |
| Text activation | SiLU (gated MLP) |
| Text norm | RMSNorm |
| Vocab | 152064 |
| Timestamp classes | 5000 |
| Timestamp resolution | 80ms |
| Weight dtype | bf16 |

### Special Tokens
| Name | ID |
|------|----|
| audio_start | 151669 |
| audio_end | 151670 |
| audio_pad | 151676 |
| timestamp | 151705 |

## Project Structure

```
asr-aligner-engine/
├── Makefile                    # nvcc build, CUDA 12.8, sm_120
├── include/
│   ├── types.h                 # Core types: Tensor, WeightStore, AlignerConfig, bf16_t
│   ├── safetensors.h           # Safetensors loader API
│   ├── tokenizer.h             # BPE tokenizer (Qwen2-style)
│   ├── audio_io.h              # WAV/PCM audio loader
│   ├── aligner.h               # Main aligner API
│   └── cuda_kernels.h          # C-compatible CUDA declarations
├── src/
│   ├── main.cpp                # CLI entry: load model, process audio+text, output SRT
│   ├── aligner.cpp             # Pipeline orchestration: mel→encode→decode→timestamps
│   ├── model.cpp               # Config parser + model init + weight loading
│   ├── safetensors.cpp         # Safetensors parser (mmap + GPU upload)
│   ├── tokenizer.cpp           # BPE tokenizer (vocab.json + merges.txt + added_tokens)
│   ├── audio_io.cpp            # WAV loader (dr_wav), PCM loader
│   ├── text_processor.cpp      # CJK tokenization, prompt building, LIS fix
│   └── timestamp.cpp           # Timestamp extraction, SRT output
├── cuda/
│   ├── kernels.cuh             # CUDA-internal typed declarations
│   ├── kernels.cu              # Core kernels: rmsnorm, layernorm, rope, gelu, silu, bf16_linear
│   ├── mel.cu                  # Mel spectrogram (cuFFT R2C + mel filterbank)
│   ├── audio_encoder.cu        # 24-layer audio encoder (Conv2D + attention + FFN)
│   ├── text_decoder.cu         # 28-layer text decoder (GQA, RoPE, MLP, prefill-only)
│   └── conv2d.cu               # Conv2D forward (im2col + GEMM)
├── third_party/
│   └── dr_wav.h                # Single-header WAV decoder
├── force-aligner/              # Optimization experiments
│   └── (future: int4, kernel fusions, etc.)
└── test/
    └── (test scripts, reference outputs)
```

## Build

```bash
make                                    # Build
make run                               # Run with default test
./force_aligner --model MODEL_DIR --audio AUDIO --text "TEXT" --lang Chinese
```

## Key Design Decisions

1. **Prefill-only**: ForceAligner does NOT autoregress. Single forward pass through text decoder.
   No KV cache, no decode step, no scheduler needed.

2. **Reuse from Infer-engine**:
   - Safetensors loader (adapted for bf16-only, no FP4)
   - Mel spectrogram (same params)
   - Kernel utilities (rmsnorm, rope, bf16_linear)
   - Dual-ABI header pattern

3. **New code needed**:
   - Conv2D frontend (ASR uses Conv1D)
   - Audio encoder with LayerNorm+GELU (ASR uses RMSNorm+SiLU)
   - Text decoder prefill-only path (no decode step needed)
   - LM head → 5000-class classifier
   - CJK text processor + prompt builder
   - Timestamp extraction + LIS fix

4. **Optimization targets** (in force-aligner/):
   - INT4 quantization for decoder weights
   - Fused attention kernels
   - Conv2D kernel optimization
   - Batched inference for multiple segments

## Verification

Test with `/data/fwsr/glm-asr/GLM-ASR/cut.wav` (14s mono 16kHz):
1. Run official Python pipeline to get reference timestamps
2. Run C++ engine, compare output timestamps
3. Verify max deviation < 20ms per word
