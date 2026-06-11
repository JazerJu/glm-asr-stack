# glm-asr-stack

<p align="center">
  English | <a href="README.md">中文</a>
</p>

`glm-asr-stack` is a packaged runtime for long-form audio transcription, video subtitles, and batch ASR. It combines a custom CUDA C `Infer-engine`, Qwen3 ForceAligner, and Python orchestration for chunking, queues, live sessions, and subtitle generation.

The default path is Docker:

- input an audio or video file
- convert it to 16 kHz audio
- split it into 30-second ASR windows
- transcribe with GLM-ASR-Nano-2512
- optionally align with ForceAligner
- write `.srt` subtitles and per-chunk transcripts

## Support Matrix

| Target | Status |
|---|---|
| Linux x86_64 + NVIDIA GPU | Supported |
| RTX 30 / 40 / 50 series | Supported, with bundled `sm86` / `sm89` / `sm120` binaries |
| Docker | Recommended |
| Native Linux runtime | Supported with installed CUDA runtime libraries |
| Windows | Not supported; WSL2 + NVIDIA Container Toolkit may work |

Model weights are not included in the image or release archives. Download them on first use or mount an existing HuggingFace cache/model directory.

## Quick Start

### 1. Get the runtime image

Use the published image:

```bash
docker pull jaceju68/glm-asr-stack:cuda12.8-runtime
```

Or build it from source:

```bash
cd ./glm-asr-stack

docker build --network=host \
  -f docker/Dockerfile.cuda12.8-runtime \
  -t glm-asr-stack:cuda12.8-runtime .
```

If your machine needs a proxy for CUDA / Python package downloads:

```bash
docker build --network=host \
  --build-arg http_proxy=http://127.0.0.1:7890 \
  --build-arg https_proxy=http://127.0.0.1:7890 \
  -f docker/Dockerfile.cuda12.8-runtime \
  -t glm-asr-stack:cuda12.8-runtime .
```

### 2. Download models

```bash
mkdir -p /data/glm-asr-models

docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  download-model all
```

This downloads:

- `zai-org/GLM-ASR-Nano-2512`
- `Qwen/Qwen3-ForcedAligner-0.6B`

### 3. Check the runtime

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  doctor
```

`doctor` checks the GPU, CUDA shared libraries, FA2 bridge libraries, model files, and `ffmpeg`.

### 4. Transcribe audio or video

Sentence-level subtitles:

```bash
mkdir -p /data/glm-asr-output

docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/video.mp4:/input/video.mp4:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  transcribe /input/video.mp4 \
    --subtitle-mode sentence \
    --output /output/video.srt
```

Word-level timestamps:

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  transcribe /input/audio.mp3 \
    --subtitle-mode word \
    --output /output/audio_words.srt
```

ASR only, without ForceAligner:

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  asr-only /input/audio.mp3 \
    --output /output/transcript.tsv
```

## Recommended Settings for 8GB GPUs

The default settings target 16GB-class GPUs:

```text
GLMASR_VRAM_UTIL=0.9
GLMASR_MAX_SEQS=256
GLMASR_MAX_BATCHED_TOKENS=10000
GLMASR_MAX_NEW_TOKENS=256
```

For 8GB GPUs, reduce both concurrency and the prefill token budget:

```bash
docker run --rm --gpus all \
  -e GLMASR_VRAM_UTIL=0.75 \
  -e GLMASR_MAX_SEQS=64 \
  -e GLMASR_MAX_BATCHED_TOKENS=4000 \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  transcribe /input/audio.mp3 \
    --subtitle-mode sentence \
    --output /output/audio.srt
```

`GLMASR_MAX_BATCHED_TOKENS` controls how many prefill tokens are sent to the GPU in one step, while `GLMASR_MAX_SEQS` controls the maximum decode concurrency. If an 8GB GPU still runs out of memory, lower them further:

```text
GLMASR_MAX_SEQS=32
GLMASR_MAX_BATCHED_TOKENS=2500
```

## Hot Words / Custom Prompt

Use `--prompt` for short context or hot-word hints:

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.wav:/input/audio.wav:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  transcribe /input/audio.wav \
    --prompt "Hot words: Claude Code, CUDA, FlashAttention." \
    --subtitle-mode sentence \
    --output /output/audio.srt
```

The prompt is intended for short context or hot words, not long documents.

## Verified Result

Local RTX 5070 Ti, Docker runtime image, input `/data/fwsr/glm-asr/e4419ff7cdea02a39bc38c268273023d_16k.mp3`:

| Audio | Chunks | ASR infer | ForceAligner | Output |
|---|---:|---:|---:|---:|
| 11:55:10 MP3 | 1431 | 94.71s | 62.19s | 7901 sentence cues |

Small sample:

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-runtime \
  transcribe /opt/glm-asr-stack/samples/smoke_120s.wav \
    --subtitle-mode sentence \
    --output /output/smoke_120s.srt
```

## Native Linux Runtime

Without Docker, you need:

- a working NVIDIA driver
- CUDA 12.x runtime libraries visible to the dynamic linker
- `ffmpeg`
- Python packages: `huggingface_hub`, `transformers`, `tokenizers`, `sentencepiece`

Then run from the repository root:

```bash
bin/glm-asr doctor
bin/glm-asr download-model all
bin/glm-asr transcribe /path/to/input.mp4 \
  --subtitle-mode sentence \
  --output runtime/output/input.srt
```

## Layout

```text
bin/
  glm-asr              # user entrypoint
  orchestrator.py      # chunking, queue, ASR/Align orchestration
  run-infer-engine     # selects the C engine binary for the GPU architecture
  run-forcealigner     # selects the ForceAligner binary for the GPU architecture
Infer-engine/
  glm_asr_infer_sm86
  glm_asr_infer_sm89
  glm_asr_infer_sm120
forcealigner-engine/
  force_aligner_sm86
  force_aligner_sm89
  force_aligner_sm120
cutedsl_fa/
  fa2_bridge_sm86.so
  fa2_bridge_sm89.so
  fa2_bridge_sm120.so
lib/
  libtvm_ffi.so
resources/
  mel_filters.bin
models/
  glm-asr-bf16/
  qwen3-forcealigner-0.6b/
runtime/
  output/
  work_dir/
  tasks/
  live_sessions/
```

## Source and Rebuild

`glm-asr-stack` ships ready-to-run binaries and `.so` files, and also includes the core engine source:

- `Infer-engine/`: GLM-ASR CUDA C inference engine source and prebuilt binaries
- `forcealigner-engine/`: ForceAligner CUDA/C++ engine source and prebuilt binaries

Rebuild directly from this repository:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
scripts/build_infer_arch.sh sm120

CUDA_HOME=/usr/local/cuda-12.8 \
scripts/build_forcealigner_arch.sh sm120
```

Supported architecture names:

```text
sm86   # RTX 30 series
sm89   # RTX 40 series
sm120  # RTX 50 series
```

The CuteDSL FA2 bridge source and build details are maintained in the Infer-engine repository. Normal users should not need to rebuild it.

## Related Projects

- Infer-engine performance core: <https://github.com/JazerJu/GlmAsr-InferEngine>
- GLM-ASR model: <https://huggingface.co/zai-org/GLM-ASR-Nano-2512>
- ForceAligner model: <https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
