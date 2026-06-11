# glm-asr-stack

<p align="center">
  <a href="README.en.md">English</a> | 中文版
</p>

`glm-asr-stack` 是一个面向长音频、视频字幕和批量转录的 GLM-ASR 推理栈。它把自研 CUDA C `Infer-engine`、Qwen3 ForceAligner、切片/队列/字幕编排脚本打包在一起，默认通过 Docker 运行。

核心目标很简单：

- 输入一个音频或视频文件
- 自动转成 16 kHz 音频并按 30s 窗口切片
- 用 GLM-ASR-Nano-2512 批量转录
- 可选用 ForceAligner 生成句子级或逐词时间戳
- 输出 `.srt` 字幕和逐段转录结果

## 支持范围

| 项目 | 状态 |
|---|---|
| Linux x86_64 + NVIDIA GPU | 支持 |
| RTX 30 系 / 40 系 / 50 系 | 支持，内置 `sm86` / `sm89` / `sm120` 二进制 |
| Docker | 推荐 |
| 裸机运行 | 支持，需要安装 CUDA 运行库和模型路径 |
| Windows | 暂不支持；可尝试 WSL2 + NVIDIA Container Toolkit |

模型权重不放进镜像或 release 包，需要首次运行时下载，或者挂载已有 HuggingFace cache。

## 快速开始

### 1. 获取运行镜像

直接使用已发布镜像：

```bash
docker pull jaceju68/glm-asr-stack:cuda12.8-slim
```

也可以从源码构建：

```bash
cd ./glm-asr-stack

docker build --network=host \
  -f docker/Dockerfile.cuda12.8-slim \
  -t glm-asr-stack:cuda12.8-slim .
```

如果你在本机用代理下载 CUDA / Python 包：

```bash
docker build --network=host \
  --build-arg http_proxy=http://127.0.0.1:7890 \
  --build-arg https_proxy=http://127.0.0.1:7890 \
  -f docker/Dockerfile.cuda12.8-slim \
  -t glm-asr-stack:cuda12.8-slim .
```

默认使用 slim 镜像，约 1.9GB。若你的环境需要完整 NVIDIA CUDA runtime，也可以改用
`jaceju68/glm-asr-stack:cuda12.8-runtime`。

### 2. 下载模型

```bash
mkdir -p /data/glm-asr-models

docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  download-model all
```

会下载：

- `zai-org/GLM-ASR-Nano-2512`
- `Qwen/Qwen3-ForcedAligner-0.6B`

### 3. 检查环境

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  doctor
```

`doctor` 会检查 GPU、CUDA 动态库、FA2 bridge、模型文件和 `ffmpeg`。

### 4. 转录视频或音频

生成句子级字幕：

```bash
mkdir -p /data/glm-asr-output

docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/video.mp4:/input/video.mp4:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  transcribe /input/video.mp4 \
    --subtitle-mode sentence \
    --output /output/video.srt
```

生成逐词时间戳字幕：

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  transcribe /input/audio.mp3 \
    --subtitle-mode word \
    --output /output/audio_words.srt
```

只转文字，不跑 ForceAligner：

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  asr-only /input/audio.mp3 \
    --output /output/transcript.tsv
```

## 8GB 显存推荐参数

默认参数面向 16GB 级别显卡：

```text
GLMASR_VRAM_UTIL=0.9
GLMASR_MAX_SEQS=256
GLMASR_MAX_BATCHED_TOKENS=10000
GLMASR_MAX_NEW_TOKENS=256
```

8GB 显卡建议降低并发和 prefill token 上限：

```bash
docker run --rm --gpus all \
  -e GLMASR_VRAM_UTIL=0.75 \
  -e GLMASR_MAX_SEQS=64 \
  -e GLMASR_MAX_BATCHED_TOKENS=4000 \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.mp3:/input/audio.mp3:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  transcribe /input/audio.mp3 \
    --subtitle-mode sentence \
    --output /output/audio.srt
```

其中 `GLMASR_MAX_BATCHED_TOKENS` 控制一次 prefill 喂给 GPU 的音频窗口总 token 数，`GLMASR_MAX_SEQS` 控制 decode 阶段最多同时跑多少条序列。8GB 卡如果仍然 OOM，可以继续降到：

```text
GLMASR_MAX_SEQS=32
GLMASR_MAX_BATCHED_TOKENS=2500
```

## 热词 / 自定义 Prompt

可以通过 `--prompt` 给 ASR 添加短 prompt 或热词提示：

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /path/to/audio.wav:/input/audio.wav:ro \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  transcribe /input/audio.wav \
    --prompt "Hot words: Claude Code, CUDA, FlashAttention." \
    --subtitle-mode sentence \
    --output /output/audio.srt
```

当前 prompt 适合放短热词或上下文提示，不建议放长篇文本。

## 已验证结果

本地 RTX 5070 Ti，Docker runtime 镜像，输入 `/data/fwsr/glm-asr/e4419ff7cdea02a39bc38c268273023d_16k.mp3`：

| 音频 | 切片 | ASR infer | ForceAligner | 输出 |
|---|---:|---:|---:|---:|
| 11:55:10 MP3 | 1431 | 94.71s | 62.19s | 7901 sentence cues |



小样例：

```bash
docker run --rm --gpus all \
  -v /data/glm-asr-models:/opt/glm-asr-stack/models \
  -v /data/glm-asr-output:/output \
  jaceju68/glm-asr-stack:cuda12.8-slim \
  transcribe /opt/glm-asr-stack/samples/smoke_120s.wav \
    --subtitle-mode sentence \
    --output /output/smoke_120s.srt
```

## 裸机运行

如果不使用 Docker，需要保证：

- NVIDIA driver 可用
- CUDA 12.x 运行库可被动态链接器找到
- `ffmpeg` 可执行
- Python 依赖：`huggingface_hub`、`transformers`、`tokenizers`、`sentencepiece`

然后在仓库根目录运行：

```bash
bin/glm-asr doctor
bin/glm-asr download-model all
bin/glm-asr transcribe /path/to/input.mp4 \
  --subtitle-mode sentence \
  --output runtime/output/input.srt
```

## 目录结构

```text
bin/
  glm-asr              # 用户入口
  orchestrator.py      # 切片、队列、ASR/Align 编排
  run-infer-engine     # 按 GPU 架构选择 C engine
  run-forcealigner     # 按 GPU 架构选择 ForceAligner
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

## 源码与重新编译

`glm-asr-stack` 默认发布的是可直接运行的二进制和 `.so`，同时也带有核心引擎源码：

- `Infer-engine/`：GLM-ASR CUDA C 推理引擎源码与预编译二进制
- `forcealigner-engine/`：ForceAligner CUDA/C++ 引擎源码与预编译二进制

直接从当前 repo 重新编译：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
scripts/build_infer_arch.sh sm120

CUDA_HOME=/usr/local/cuda-12.8 \
scripts/build_forcealigner_arch.sh sm120
```

支持的架构名：

```text
sm86   # RTX 30 系
sm89   # RTX 40 系
sm120  # RTX 50 系
```

CuteDSL FA2 bridge 的源码和编译细节放在 Infer-engine 仓库中维护；普通用户不需要重新编译它。

## 相关项目

- Infer-engine 性能核心：<https://github.com/JazerJu/GlmAsr-InferEngine>
- GLM-ASR 模型：<https://huggingface.co/zai-org/GLM-ASR-Nano-2512>
- ForceAligner 模型：<https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B>
