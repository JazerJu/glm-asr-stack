# GLM-ASR InferEngine

<p align="center">
  <img src="assets/miner.png" width="128" alt="GLM-ASR InferEngine" />
</p>

<p align="center">
  English Version | <a href="README.md">中文版</a>
</p>

`GLM-ASR InferEngine` is a CUDA C/C++ high-performance inference engine for `GLM-ASR-Nano-2512`. This repository focuses on one thing only: **running GLM-ASR inference as fast as possible on consumer NVIDIA GPUs, with reproducible benchmarks**.

The full subtitle/task/Docker/ForceAligner stack lives in a separate project:

> https://github.com/JazerJu/glm-asr-stack

## Build `glm_asr_infer` From Source

```bash
cmake -S . -B build_cmake \
  -DCUDA_ROOT=/usr/local/cuda-12.8 \
  -DGLMASR_CUDA_ARCH=sm_120 \
  -DGLMASR_PYTHON_EXECUTABLE="$(which python3)"
cmake --build build_cmake -j
cmake --build build_cmake --target cutedsl_fa2_bridge -j
```

Build outputs:

```bash
./glm_asr_infer
cutedsl_fa/fa2_bridge.so
```

`glm_asr_infer` is the C/CUDA inference engine. `fa2_bridge.so` is the CuteDSL FA2/TMA performance plugin loaded at runtime with `dlopen`. The `cutedsl_fa2_bridge` target selects the proper CuteDSL kernel source from `GLMASR_CUDA_ARCH`. The output path is always:

```bash
cutedsl_fa/fa2_bridge.so
```

Common architecture targets:

| GPU | CMake argument | CuteDSL source |
|---|---|---|
| RTX 50 series | `-DGLMASR_CUDA_ARCH=sm_120` | `cutedsl_fa/fa2_kernel.py` + `cutedsl_fa/fa_tma.py` |
| RTX 40 series | `-DGLMASR_CUDA_ARCH=sm_89` | `cutedsl_fa/fa2_kernel_sm89.py` |
| RTX 30 series | `-DGLMASR_CUDA_ARCH=sm_86` | `cutedsl_fa/fa2_kernel_sm89.py`, compiled with `GLMASR_CUTEDSL_SM89_ARCH=sm_86` |

## Infer-engine vs vLLM Benchmark

The most important benchmark for this repository is the same-data comparison against vLLM, not an isolated RTFx number.

Setup:

- Model: `zai-org/GLM-ASR-Nano-2512` BF16
- Data: the same HuggingFace samples converted into 256 x 30s padded audio clips
- Generation: `max_new_tokens=1`
- vLLM: `vllm==0.18.0`
- Warmup: 16 separate pad30 samples, not included in timed results
- vLLM prefix cache: disabled
- GPU: RTX 5070 Ti, local warm runs

| Engine | Input | Wall / infer window | Notes |
|---|---|---|---|
| **Infer-engine** | 256 x 30s pad30 raw PCM | **10.6s** | C/CUDA engine, batched daemon |
| **vLLM 0.18.0** | Same 256 x 30s pad30 FLAC clips | **11.65s** | Offline Python API, FlashAttention |

Reproduce:

```bash
pip install -r requirements.txt
cmake -S . -B build_cmake \
  -DCUDA_ROOT=/usr/local/cuda-12.8 \
  -DGLMASR_CUDA_ARCH=sm_120 \
  -DGLMASR_PYTHON_EXECUTABLE="$(which python3)"
cmake --build build_cmake -j
cmake --build build_cmake --target cutedsl_fa2_bridge -j
python3 bench_vllm_pad30.py --download-models
```

`bench_vllm_pad30.py` prepares two pad30 views from the same HuggingFace samples: raw PCM `.bin` files for Infer-engine and `.flac` files for vLLM. It then runs the freshly built `./glm_asr_infer` and benchmarks vLLM through the offline Python API in the same Python process. By default it uses 16 extra samples for independent warmup, and the timed 256 samples never overlap with warmup. The vLLM offline path explicitly disables prefix caching. Driver version, GPU model, power limit, vLLM version, and batch settings can change the result.

Run only one side:

```bash
# Infer-engine only
python3 bench_vllm_pad30.py --download-models --skip-vllm

# vLLM offline only; prepares the same pad30 FLAC files
python3 bench_vllm_pad30.py --download-models --skip-infer

# To reproduce the OpenAI server path explicitly
python3 bench_vllm_pad30.py --download-models --skip-infer --vllm-backend server

# If a vLLM server is already running, only submit requests
python3 bench_vllm_pad30.py --download-models --skip-infer --vllm-backend server --keep-vllm-server
```

## RTX 50 / 40 / 30 Series 11h Two-Stage Run

This table comes from the same 11h audio file with ASR followed by ForceAligner. ForceAligner belongs to the full stack; this repository owns the ASR engine. See `glm-asr-stack` for the end-to-end subtitle pipeline.

| GPU | Arch | ASR infer | ForceAligner | Two-stage core time | Notes |
|---|---|---|---|---|---|
| RTX 5070 Ti | SM120 | ~95s | ~65s | ~160s | Local 11h two-stage runs, several versions in this range |
| RTX 4090 | SM89 | **71.71s** | **52.89s** | **124.60s** | Remote 4090, SM89 CuteDSL FA2 |
| RTX 3090 Ti | SM86 | **105.35s** | **64.45s** | **169.80s** | First rented 3090 Ti run, SM86 CuteDSL FA2 |

Summary: 4090 is fastest, 5070 Ti is in the middle, and 3090 Ti is slowest. Compared with 3090 Ti, 4090 is about `1.47x` faster on ASR, `1.22x` faster on ForceAligner, and `1.36x` faster on the two-stage core time.

## 2611 Mixed-Length Benchmark

This table compares Infer-engine only. The dataset is the 2611 mixed-length HuggingFace audio clips with the default `start=9, num=2611`.

| GPU | Arch | Infer time | RTFx | Notes |
|---|---|---|---|---|
| RTX 5070 Ti | SM120 | **37.1s** | **516.5x** | Local release benchmark |

Reproduce the 2611 run on one machine:

```bash
python3 transcribe_librispeech.py --download-models
```

## Scope

- Loads BF16 `safetensors` weights directly
- Does not use PyTorch or vLLM as the Infer-engine runtime
- Implements mel, encoder, decoder, paged KV cache, and scheduler in CUDA C/C++
- Release numbers currently use RTX 5070 Ti / SM120; the RTX 40 series SM89 CuTeDSL FA2 source branch is kept, and RTX 30/40 series runtime packaging belongs to `glm-asr-stack`
- Benchmark numbers in this README are meant to be reproducible from scripts

## Benchmark Environment

Following the nano-vLLM style, benchmark dependencies are listed in the repo:

```bash
pip install -r requirements.txt
```

This includes `vllm==0.18.0` for the pad30 comparison.

## Benchmark Scripts

```bash
# First table: pad30 comparison, Infer-engine and vLLM offline on the same samples
python3 bench_vllm_pad30.py --download-models

# Second table: 2611 mixed-length LibriSpeech, Infer-engine only
python3 transcribe_librispeech.py --download-models
```

Default dataset:

```text
dataset: hf-audio/esb-datasets-test-only-sorted
config:  librispeech
split:   test.clean
```

If the local HuggingFace Arrow cache exists, it is used first. Otherwise the script falls back to `datasets.load_dataset()`.

Default vLLM pad30 settings:

```bash
VLLM_BENCH_BACKEND=offline
VLLM_MAX_BATCHED_TOKENS=6000
VLLM_MAX_NUM_SEQS=256
VLLM_GPU_MEMORY_UTILIZATION=0.9
GLMASR_BENCH_WARMUP=16
```

Override them with environment variables as needed.

## Dependencies

Build:

- Linux x86_64
- NVIDIA GPU
- CUDA Toolkit 12.8 at `/usr/local/cuda-12.8`
- `gcc`, `cmake`

Python benchmark dependencies:

```bash
pip install -r requirements.txt
```

## Runtime Settings

Infer-engine benchmark defaults:

```bash
GLMASR_VRAM_UTIL=0.9
GLMASR_MAX_SEQS=256
GLMASR_MAX_BATCHED_TOKENS=10000
GLMASR_ENCODER_FA_PIPELINED=1
GLMASR_ENCODER_FA_VLLM=1
GLMASR_ENCODER_FA_LONG_MIN_SEQ=1
GLMASR_ENCODER_FA_MIN_SEQ=1
python3 transcribe_librispeech.py --download-models
```

For 8GB VRAM cards, start with:

```bash
GLMASR_MAX_SEQS=64
GLMASR_MAX_BATCHED_TOKENS=4000
GLMASR_VRAM_UTIL=0.85
python3 transcribe_librispeech.py --download-models
```

## Layout

```text
Infer-engine/
├── CMakeLists.txt     # standard CMake build entry
├── cuda/              # CUDA kernels: mel, encoder, decoder, paged KV, FA paths
├── include/           # C ABI headers and shared types
├── src/               # engine, scheduler, model loader, tokenizer, audio I/O
├── cutedsl_fa/        # CuTeDSL FA2 kernels and bridge code
├── tools/             # internal benchmark helpers used by transcribe_librispeech.py
├── transcribe_librispeech.py  # LibriSpeech transcription and comparison entry point
└── AGENTS.md          # detailed engineering notes and profiling history
```

## Models and Data

This repository does not redistribute model weights or benchmark audio. The default command downloads:

- Model: `zai-org/GLM-ASR-Nano-2512` -> `./models/GLM-ASR-Nano-2512`
- Mel filters: tracked at `./resources/mel_filters.bin`
- Benchmark data: default `hf-audio/esb-datasets-test-only-sorted/librispeech`

If a fully fixed benchmark dataset is needed later, create a HuggingFace dataset and switch inputs with:

```bash
GLMASR_BENCH_DATASET=JazerJu/your-dataset
GLMASR_BENCH_DATASET_CONFIG=default
GLMASR_BENCH_SPLIT=test
```

## Out of Scope

The following belong to `glm-asr-stack`:

- Docker runtime image
- Model download UI / model path management
- Video-to-subtitle pipeline
- ForceAligner alignment
- live session / task queue
- End-user one-line transcription commands

This repository keeps only the engine source, kernels, scheduler, and performance reproduction scripts.
